import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import json
import time
import bittensor as bt
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict
import nova_ph2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening

from molecules import generate_valid_random_molecules_batch

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


def build_component_weights(top_pool: pd.DataFrame, rxn_id: int) -> Dict[str, Dict[int, float]]:
    """
    Build component weights based on scores of molecules containing them.
    Returns dict with 'A', 'B', 'C' keys mapping to {component_id: weight}
    """
    weights = {'A': defaultdict(float), 'B': defaultdict(float), 'C': defaultdict(float)}
    counts = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}
    
    if top_pool.empty:
        return weights
    
    # Extract component IDs and scores
    for _, row in top_pool.iterrows():
        name = row['name']
        score = row['score']
        parts = name.split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                weights['A'][A_id] += max(0, score)  # Only positive contributions
                weights['B'][B_id] += max(0, score)
                counts['A'][A_id] += 1
                counts['B'][B_id] += 1
                
                if len(parts) > 4:
                    C_id = int(parts[4])
                    weights['C'][C_id] += max(0, score)
                    counts['C'][C_id] += 1
            except (ValueError, IndexError):
                continue
    
    # Normalize by count and add smoothing
    for role in ['A', 'B', 'C']:
        for comp_id in weights[role]:
            if counts[role][comp_id] > 0:
                weights[role][comp_id] = weights[role][comp_id] / counts[role][comp_id] + 0.1  # Smoothing
    
    return weights

def select_diverse_elites(top_pool: pd.DataFrame, n_elites: int, min_score_ratio: float = 0.7) -> pd.DataFrame:
    """
    Select diverse elite molecules: top by score, but ensure diversity in component space.
    """
    if top_pool.empty or n_elites <= 0:
        return pd.DataFrame()
    
    # Take top candidates (more than needed for diversity filtering)
    top_candidates = top_pool.head(min(len(top_pool), n_elites * 3))
    if len(top_candidates) <= n_elites:
        return top_candidates
    
    # Score threshold: at least min_score_ratio of max score
    max_score = top_candidates['score'].max()
    threshold = max_score * min_score_ratio
    candidates = top_candidates[top_candidates['score'] >= threshold]
    
    # Select diverse set: prefer molecules with different components
    selected = []
    used_components = {'A': set(), 'B': set(), 'C': set()}
    
    # First, add top scorer
    if not candidates.empty:
        top_idx = candidates.index[0]
        top_row = candidates.iloc[0]
        selected.append(top_idx)
        parts = top_row['name'].split(":")
        if len(parts) >= 4:
            try:
                used_components['A'].add(int(parts[2]))
                used_components['B'].add(int(parts[3]))
                if len(parts) > 4:
                    used_components['C'].add(int(parts[4]))
            except (ValueError, IndexError):
                pass
    
    # Then add diverse molecules
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx in selected:
            continue
        
        parts = row['name'].split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                C_id = int(parts[4]) if len(parts) > 4 else None
                
                # Prefer molecules with new components
                is_diverse = (A_id not in used_components['A'] or 
                             B_id not in used_components['B'] or
                             (C_id is not None and C_id not in used_components['C']))
                
                if is_diverse or len(selected) < n_elites * 0.5:  # Always take some top ones
                    selected.append(idx)
                    used_components['A'].add(A_id)
                    used_components['B'].add(B_id)
                    if C_id is not None:
                        used_components['C'].add(C_id)
            except (ValueError, IndexError):
                # If parsing fails, just add it
                if len(selected) < n_elites:
                    selected.append(idx)
    
    # Fill remaining slots with top scorers
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx not in selected:
            selected.append(idx)
    
    return candidates.loc[selected[:n_elites]] if selected else candidates.head(n_elites)

def main(config: dict):
    n_samples = config["num_molecules"] * 5
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score", "Target", "Anti"])
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    iteration = 0
    mutation_prob = 0.1
    elite_frac = 0.25
    prev_avg_score = None
    score_improvement_rate = 0.0
    seen_inchikeys = set()
    start = time.time()

    n_samples_first_iteration = n_samples if config["allowed_reaction"] == "rxn:5" else n_samples*4
    while time.time() - start < 1800:
        iteration += 1
        start_time = time.time()
        neighborhood_limit = 2 if (time.time() - start) > 1680 else 0
        component_weights = build_component_weights(top_pool, rxn_id) if not top_pool.empty else None
        elite_df = select_diverse_elites(top_pool, min(100, len(top_pool))) if not top_pool.empty else pd.DataFrame()
        elite_names = elite_df["name"].tolist() if not elite_df.empty else None
        
        if prev_avg_score is not None and not top_pool.empty:
            current_avg = top_pool['score'].mean()
            score_improvement_rate = (current_avg - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
            if score_improvement_rate > 0.01:  # Good improvement
                elite_frac = min(0.7, elite_frac * 1.1)
                mutation_prob = max(0.05, mutation_prob * 0.95)
            elif score_improvement_rate < -0.01:  # Declining
                elite_frac = max(0.2, elite_frac * 0.9)
                mutation_prob = min(0.4, mutation_prob * 1.1)
            
        data = generate_valid_random_molecules_batch(rxn_id, n_samples=n_samples_first_iteration if iteration == 1 else n_samples, db_path=DB_PATH, subnet_config=config, batch_size=300, elite_names=elite_names, 
                                                     elite_frac=elite_frac, mutation_prob=mutation_prob, avoid_inchikeys=seen_inchikeys, component_weights=component_weights, neighborhood_limit=neighborhood_limit)
                
        if data.empty:
            bt.logging.warning(f"[Miner] Iteration {iteration}: No valid molecules produced; continuing")
            continue

        try:
            filterd_data = data[~data['InChIKey'].isin(seen_inchikeys)]
            if len(filterd_data) < len(data):
                bt.logging.warning(f"[Miner] Iteration {iteration}: {len(data) - len(filterd_data)} molecules were previously seen; continuing with unseen only")

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
        data['Target'] = target_score_from_data(data['smiles'])
        data['Anti'] = antitarget_scores()
        data['score'] = data['Target'] - (config['antitarget_weight'] * data['Anti'])
        seen_inchikeys.update([k for k in data["InChIKey"].tolist() if k])
        total_data = data[["name", "smiles", "InChIKey", "score", "Target", "Anti"]]
        top_pool = pd.concat([top_pool, total_data])
        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
        top_pool = top_pool.sort_values(by="score", ascending=False)
        top_pool = top_pool.head(config["num_molecules"])
        
        current_avg_score = top_pool['score'].mean() if not top_pool.empty else None

        if current_avg_score is not None:
            if prev_avg_score is not None:
                score_improvement_rate = (current_avg_score - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
            prev_avg_score = current_avg_score
        
        bt.logging.info(f"Iteration {iteration} || Time: {round(time.time() - start_time,2)} | Avg: {top_pool['score'].mean():.4f} | Max: {top_pool['score'].max():.4f} | Min: {top_pool['score'].min():.4f} | Improve: {score_improvement_rate*100:.2f}% | Elite frac: {elite_frac:.3f} | Mute: {mutation_prob:.3f} | Neighbor: {neighborhood_limit}")
        
        top_entries = {"molecules": top_pool["name"].tolist()}
        with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
            json.dump(top_entries, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    
    config = get_config()
    start_time_1 = time.time()
    initialize_models(config)
    bt.logging.info(f"{time.time() - start_time_1} seconds for model initialization")
    main(config)
