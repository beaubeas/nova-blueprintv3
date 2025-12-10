import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import json
import time
import bittensor as bt
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import pandas as pd
from pathlib import Path
import nova_ph2
from itertools import combinations

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening
from molecules import (
    generate_valid_random_molecules_batch,
    select_diverse_elites,
    build_component_weights,
    compute_tanimoto_similarity_to_pool,
    sample_random_valid_molecules,
    compute_maccs_entropy
)

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")


target_models = []
antitarget_models = []

def get_config(input_file: str = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


def initialize_models(config: dict):
    """Initialize separate model instances for each target and antitarget sequence."""
    global target_models, antitarget_models
    target_models = []
    antitarget_models = []
    
    for seq in config["target_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        target_models.append(wrapper)
    
    for seq in config["antitarget_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        antitarget_models.append(wrapper)


# ---------- scoring helpers (reuse pre-initialized models) ----------
def target_score_from_data(data: pd.Series):
    """Score molecules against all target models. target_sequence parameter kept for compatibility but not used."""
    global target_models, antitarget_models
    try:
        target_scores = []
        smiles_list = data.tolist()
        for target_model in target_models:
            scores = target_model.score_molecules(smiles_list)
            for antitarget_model in antitarget_models:
                antitarget_model.smiles_list = smiles_list
                antitarget_model.smiles_dict = target_model.smiles_dict

            scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
            target_scores.append(scores["target"])
        # Average across all targets
        target_series = pd.DataFrame(target_scores).mean(axis=0)
        return target_series
    except Exception as e:
        bt.logging.error(f"Target scoring error: {e}")
        return pd.Series(dtype=float)


def antitarget_scores():
    """Score molecules against all antitarget models. antitarget_sequence parameter kept for compatibility but not used."""
    
    global antitarget_models
    try:
        antitarget_scores = []
        for i, antitarget_model in enumerate(antitarget_models):
            antitarget_model.create_screen_loader(antitarget_model.protein_dict, antitarget_model.smiles_dict)
            antitarget_model.screen_df = virtual_screening(antitarget_model.screen_df, 
                                            antitarget_model.model, 
                                            antitarget_model.screen_loader,
                                            os.getcwd(),
                                            save_interpret=False,
                                            ligand_dict=antitarget_model.smiles_dict, 
                                            device=antitarget_model.device,
                                            save_cluster=False,
                                            )
            scores = antitarget_model.screen_df[['predicted_binding_affinity']]
            scores.rename(columns={'predicted_binding_affinity': f"anti_{i}"}, inplace=True)
            antitarget_scores.append(scores[f"anti_{i}"])
        
        if not antitarget_scores:
            return pd.Series(dtype=float)
        
        # average across antitargets
        anti_series = pd.DataFrame(antitarget_scores).mean(axis=0)
        return anti_series
    except Exception as e:
        bt.logging.error(f"Antitarget scoring error: {e}")
        return pd.Series(dtype=float)


def _cpu_random_candidates_with_similarity(
    iteration: int,
    n_samples: int,
    subnet_config: dict,
    top_pool_df: pd.DataFrame,
    avoid_inchikeys: set[str] | None = None,
    thresh: float = 0.8
) -> pd.DataFrame:
    """
    CPU-side helper:
    - draws a random batch of valid molecules (independent of the GPU batch),
    - computes Tanimoto similarity vs. current top_pool,
    - returns a DataFrame with name, smiles, InChIKey, tanimoto_similarity.
    """
    try:
        random_df = sample_random_valid_molecules(
            n_samples=n_samples,
            subnet_config=subnet_config,
            avoid_inchikeys=avoid_inchikeys,
            focus_neighborhood_of=top_pool_df
        )
        if random_df.empty or top_pool_df.empty:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

        sims = compute_tanimoto_similarity_to_pool(
            candidate_smiles=random_df["smiles"],
            pool_smiles=top_pool_df["smiles"],
        )
        random_df = random_df.copy()
        random_df["tanimoto_similarity"] = sims.reindex(random_df.index).fillna(0.0)
        random_df =random_df.sort_values(by="tanimoto_similarity", ascending=False)
        random_df_filtered = random_df[random_df["tanimoto_similarity"] >= thresh]
            
        if random_df_filtered.empty:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])
            
        random_df_filtered = random_df_filtered.reset_index(drop=True)
        return random_df_filtered[["name", "smiles", "InChIKey"]]
    except Exception as e:
        bt.logging.warning(f"[Miner] _cpu_random_candidates_with_similarity failed: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

def select_diverse_subset(pool, top_95_smiles, subset_size=5, entropy_threshold=0.1):
    smiles_list = pool["smiles"].tolist()
    for combination in combinations(smiles_list, subset_size):
        test_subset = top_95_smiles + list(combination)
        entropy = compute_maccs_entropy(test_subset)
        if entropy >= entropy_threshold:
            print(f"Entropy Threshold Met: {entropy:.4f}")
            return pool[pool["smiles"].isin(combination)]

    print("No combination exceeded the given entropy threshold.")
    return pd.DataFrame()


def main(config: dict):
    n_samples = config["num_molecules"] * 5
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score", "Target", "Anti"])
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    iteration = 0
    mutation_prob = 0.1
    elite_frac = 0.25
    seen_inchikeys = set()
    seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])
    start = time.time()
    prev_avg_score = None
    current_avg_score = None
    score_improvement_rate = 0.0
    

    n_samples_first_iteration = n_samples if config["allowed_reaction"] == "rxn:5" else n_samples * 4
    with ProcessPoolExecutor(max_workers=1) as cpu_executor:
        while time.time() - start < 1800:
            iteration += 1
            iter_start_time = time.time()
            remaining_time = 1800 - (time.time() - start)

            adjust_for_entropy = False
            if remaining_time <= 60:
                adjust_for_entropy = True

            component_weights = build_component_weights(top_pool, rxn_id) if not top_pool.empty else None
            elite_df = select_diverse_elites(top_pool, min(100, len(top_pool))) if not top_pool.empty else pd.DataFrame()
            elite_names = elite_df["name"].tolist() if not elite_df.empty else None

            data = generate_valid_random_molecules_batch(
                rxn_id,
                n_samples=n_samples_first_iteration if iteration == 1 else n_samples,
                db_path=DB_PATH,
                subnet_config=config,
                batch_size=300,
                elite_names=elite_names,
                elite_frac=elite_frac,
                mutation_prob=mutation_prob,
                avoid_inchikeys=seen_inchikeys,
                component_weights=component_weights,
            )

            gen_time = time.time() - iter_start_time
            bt.logging.info(
                f"[Miner] Iteration {iteration}: {len(data)} Samples Generated in ~{gen_time:.2f}s (pre-score)"
            )

            if data.empty:
                bt.logging.warning(f"[Miner] Iteration {iteration}: No valid molecules produced; continuing")
                continue
            
            if not seed_df.empty:
                data = pd.concat([data, seed_df])
                data  = data.drop_duplicates(subset=["InChIKey"], keep="first")

            try:
                filterd_data = data[~data["InChIKey"].isin(seen_inchikeys)]
                if len(filterd_data) < len(data):
                    bt.logging.warning(
                        f"[Miner] Iteration {iteration}: {len(data) - len(filterd_data)} molecules were previously seen; continuing with unseen only"
                    )

                dup_ratio = (len(data) - len(filterd_data)) / max(1, len(data))
                if dup_ratio > 0.6:
                    mutation_prob = min(0.5, mutation_prob * 1.5)
                    elite_frac = max(0.2, elite_frac * 0.8)
                elif dup_ratio < 0.2 and not top_pool.empty:
                    mutation_prob = max(0.05, mutation_prob * 0.9)
                    elite_frac = min(0.8, elite_frac * 1.1)

                data = filterd_data

            except Exception as e:
                bt.logging.warning(f"[Miner] Pre-score deduplication failed; proceeding unfiltered: {e}")

            data = data.reset_index(drop=True)

            cpu_future = None
            if not top_pool.empty and (score_improvement_rate<0.01 and iteration>1):
                cpu_future = cpu_executor.submit(
                    _cpu_random_candidates_with_similarity,
                    iteration,
                    100,
                    config,
                    top_pool.head(5)[["name", "smiles", "InChIKey"]],
                    seen_inchikeys,
                    0.9
                )
            gpu_start_time = time.time()
            data["Target"] = target_score_from_data(data["smiles"])
            data["Anti"] = antitarget_scores()
            data["score"] = data["Target"] - (config["antitarget_weight"] * data["Anti"])
            
            gpu_time = time.time() - gpu_start_time
            bt.logging.info(f"[Miner] Iteration {iteration}: GPU scoring time ~{gpu_time:.2f}s")
            if cpu_future is not None:
                try:
                    cpu_df = cpu_future.result(timeout=0)
                    if not cpu_df.empty:
                        seed_df = cpu_df.copy()
                except TimeoutError:
                    bt.logging.info(f"[Miner] Iteration {iteration}: CPU similarity still running — continuing without it this iteration")
                except Exception as e:
                    bt.logging.warning(f"[Miner] CPU random/similarity computation failed; proceeding without it: {e}")
            seen_inchikeys.update([k for k in data["InChIKey"].tolist() if k])
            total_data = data[["name", "smiles", "InChIKey", "score", "Target", "Anti"]]
            prev_avg_score = top_pool['score'].mean() if not top_pool.empty else None
            top_pool = pd.concat([top_pool, total_data])
            top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
            top_pool = top_pool.sort_values(by="score", ascending=False)

            if adjust_for_entropy:
                try:
                    top_95 = top_pool.iloc[:95]
                    remaining_pool = top_pool.iloc[95:]  # Remaining molecules after the top 95
                    additional_5 = select_diverse_subset(remaining_pool, top_95["smiles"].tolist(), subset_size=5, entropy_threshold=config['entropy_min_threshold'])
                    if not additional_5.empty:
                        top_pool = pd.concat([top_95, additional_5]).reset_index(drop=True)
                        entropy = compute_maccs_entropy(top_pool['smiles'].to_list())
                        bt.logging.info(f"[Miner] Iteration {iteration}: New Entropy = {entropy:.4f}")
                    else:
                        top_pool = top_pool.head(config["num_molecules"])
                        entropy = compute_maccs_entropy(top_pool['smiles'].to_list())
                        bt.logging.info(f"[Miner] Iteration {iteration}: New Entropy = {entropy:.4f}")
                
                except Exception as e:
                    bt.logging.warning(f"[Miner] Entropy handling failed: {e}")
            else:
                top_pool = top_pool.head(config["num_molecules"])
            
            current_avg_score = top_pool['score'].mean() if not top_pool.empty else None

            if current_avg_score is not None:
                if prev_avg_score is not None:
                    score_improvement_rate = (current_avg_score - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
                prev_avg_score = current_avg_score
            iter_total_time = time.time() - iter_start_time
            top_entries = {"molecules": top_pool["name"].tolist()}
            bt.logging.info(
                    f"Iteration {iteration} || Time: {iter_total_time:.2f}s | "
                    f"Avg: {top_pool['score'].mean():.4f} | Max: {top_pool['score'].max():.4f} | "
                    f"Min: {top_pool['score'].min():.4f} | Elite frac: {elite_frac:.2f} | "
                    f"Mute: {mutation_prob:.2f} | "
                    f"Improve: {score_improvement_rate:.4f}"
                )

            with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    config = get_config()
    start_time_1 = time.time()
    initialize_models(config)
    bt.logging.info(f"{time.time() - start_time_1} seconds for model initialization")
    main(config)
