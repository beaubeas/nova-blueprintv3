import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import json
import time
import torch
import bittensor as bt
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import nova_ph2
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils import ligand_init, protein_init
from nova_ph2.PSICHIC.psichic_utils.data_utils import DataLoader,virtual_screening
from nova_ph2.PSICHIC.psichic_utils.dataset import ProteinMoleculeDataset
from molecules import generate_valid_random_molecules_batch

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")


target_models = []
antitarget_models = []

def get_config(input_file: str = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}

def initialize_models(config: dict):
    """Initialize a SINGLE model instance for all proteins."""
    global target_models, antitarget_models
    
    # Create ONE model instance
    wrapper = PsichicWrapper()
    wrapper.load_model()  # Load model weights once
    
    # Store protein sequences
    wrapper.target_sequences = config["target_sequences"]
    wrapper.antitarget_sequences = config["antitarget_sequences"]
    wrapper.all_sequences = wrapper.target_sequences + wrapper.antitarget_sequences
    
    # Sanitize and initialize ALL proteins ONCE (not per-protein!)
    allowed_chars = set(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'])
    sanitized_sequences = []
    for seq in wrapper.all_sequences:
        sanitized_seq = ''.join([aa if aa in allowed_chars else 'X' for aa in seq])
        sanitized_sequences.append(sanitized_seq)
    
    # Store sanitized sequences
    wrapper.sanitized_sequences = sanitized_sequences
    
    # Initialize protein dict ONCE for all proteins
    bt.logging.info(f"[Miner] Initializing {len(sanitized_sequences)} proteins once...")
    wrapper.combined_protein_dict = protein_init(sanitized_sequences)
    bt.logging.info(f"[Miner] Protein initialization complete")
    
    # Store in globals
    target_models = [wrapper]
    antitarget_models = []


def parallel_score_molecules(smiles_series: pd.Series, config: dict) -> tuple:
    
    global target_models, antitarget_models
    
    wrapper = target_models[0]  # Single model instance
    smiles_list = smiles_series.tolist()
    n_molecules = len(smiles_list)
    
    if n_molecules == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    try:
        
        # Step 1: Initialize smiles ONCE
        bt.logging.info(f"[Miner] Initializing {n_molecules} molecules")
        smiles_dict = ligand_init(smiles_list)
        
        # Step 2: Use pre-computed protein data
        n_targets = len(wrapper.target_sequences)
        n_antitargets = len(wrapper.antitarget_sequences)
        bt.logging.info(f"[Miner] Using pre-initialized {len(wrapper.sanitized_sequences)} proteins ({n_targets} targets, {n_antitargets} antitargets)")
        
        # Step 3: Set wrapper's protein_seq and smiles_list
        wrapper.protein_seq = wrapper.sanitized_sequences
        wrapper.smiles_list = smiles_list
        
        torch.cuda.empty_cache()
        
        # Step 4: Create screen_loader
        wrapper.create_screen_loader(wrapper.combined_protein_dict, smiles_dict)
        wrapper.screen_df = pd.DataFrame({'Protein': [k for k in wrapper.protein_seq for _ in wrapper.smiles_list],
                                        'Ligand': [l for _ in wrapper.protein_seq for l in wrapper.smiles_list]
                                       })
        
        dataset = ProteinMoleculeDataset(wrapper.screen_df, 
                                         smiles_dict, 
                                         wrapper.combined_protein_dict, 
                                         device=wrapper.device
                                         )
        
        wrapper.screen_loader = DataLoader(dataset,
                                        batch_size=wrapper.runtime_config.BATCH_SIZE,
                                        shuffle=False,
                                        follow_batch=['mol_x', 'clique_x', 'prot_node_aa']
                                        )

        # Step 5: ONE forward pass for ALL proteins!
        bt.logging.info(f"[Miner] Running virtual screening...")
        wrapper.screen_df = virtual_screening(
            wrapper.screen_df, 
            wrapper.model, 
            wrapper.screen_loader,
            os.getcwd(),
            save_interpret=False,
            ligand_dict=smiles_dict, 
            device=wrapper.device,
            save_cluster=False,
        )
        
        # === DEBUG: Check results_df structure ===
        results_df = wrapper.screen_df
        bt.logging.info(f"[DEBUG] Results DF shape: {results_df.shape}")
        bt.logging.info(f"[DEBUG] Unique proteins in results: {results_df['Protein'].nunique()}")
        bt.logging.info(f"[DEBUG] Protein values: {results_df['Protein'].unique()[:5]}")
        bt.logging.info(f"[DEBUG] Expected proteins: {wrapper.sanitized_sequences[:3]}")
        
        # Step 6: Parse results by protein type
        target_results = []
        antitarget_results = []
        
        # Process each protein's results
        for i, protein_seq in enumerate(wrapper.sanitized_sequences):
            # Filter rows for this specific protein
            protein_mask = results_df['Protein'] == protein_seq
            protein_df = results_df[protein_mask].copy()
            
            bt.logging.info(f"[DEBUG] Protein {i}: Found {len(protein_df)} rows (expected {n_molecules})")
            
            if len(protein_df) == 0:
                bt.logging.warning(f"[Miner] No results for protein {i}! Using zeros.")
                protein_scores = [0.0] * n_molecules
            else:
                # Sort by ligand to ensure correct order
                protein_df['ligand_idx'] = protein_df['Ligand'].apply(
                    lambda x: smiles_list.index(x) if x in smiles_list else -1
                )
                protein_df = protein_df[protein_df['ligand_idx'] >= 0]  # Remove invalid
                protein_df = protein_df.sort_values('ligand_idx')
                protein_scores = protein_df['predicted_binding_affinity'].tolist()
                
                # Ensure we have exactly n_molecules scores
                if len(protein_scores) != n_molecules:
                    bt.logging.warning(f"Protein {i} returned {len(protein_scores)} scores, expected {n_molecules}. Padding/truncating.")
                    protein_scores = protein_scores[:n_molecules] + [0.0] * max(0, n_molecules - len(protein_scores))
            
            # Categorize as target or antitarget
            category = "TARGET" if i < n_targets else "ANTITARGET"
            bt.logging.info(f"[DEBUG] Protein {i} ({category}): avg score = {np.mean(protein_scores):.4f}")
            
            if i < n_targets:
                target_results.append(protein_scores)
            else:
                antitarget_results.append(protein_scores)
        
        # Step 7: Average scores
        # For target: Should have exactly 1 protein
        if len(target_results) == 1:
            target_series = pd.Series(target_results[0])
            bt.logging.info(f"[DEBUG] Using single target (no averaging needed)")
        elif len(target_results) > 1:
            bt.logging.error(f"[ERROR] Expected 1 target, got {len(target_results)}! Averaging anyway...")
            target_array = np.array(target_results, dtype=np.float32)
            target_series = pd.Series(target_array.mean(axis=0))
        else:
            bt.logging.error(f"[ERROR] No target results! Using zeros.")
            target_series = pd.Series([0.0] * n_molecules)
        
        # For antitarget: Should have exactly 2 proteins
        if len(antitarget_results) == 2:
            antitarget_array = np.array(antitarget_results, dtype=np.float32)
            antitarget_series = pd.Series(antitarget_array.mean(axis=0))
            bt.logging.info(f"[DEBUG] Averaging 2 antitargets")
        elif len(antitarget_results) > 2:
            bt.logging.warning(f"[WARNING] Expected 2 antitargets, got {len(antitarget_results)}! Averaging all...")
            antitarget_array = np.array(antitarget_results, dtype=np.float32)
            antitarget_series = pd.Series(antitarget_array.mean(axis=0))
        elif len(antitarget_results) == 1:
            bt.logging.warning(f"[WARNING] Only 1 antitarget result! Using as-is.")
            antitarget_series = pd.Series(antitarget_results[0])
        else:
            bt.logging.error(f"[ERROR] No antitarget results! Using zeros.")
            antitarget_series = pd.Series([0.0] * n_molecules)
        
        bt.logging.info(f"[Miner] Scoring complete. Target avg: {target_series.mean():.4f}, Antitarget avg: {antitarget_series.mean():.4f}")
        
        return target_series, antitarget_series
        
    except Exception as e:
        bt.logging.error(f"Batch scoring failed: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())
        return pd.Series([0.0] * n_molecules), pd.Series([0.0] * n_molecules)


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
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    n_samples = 512
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score", "Target", "Anti"])
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    iteration = 0
    mutation_prob = 0.1
    elite_frac = 0.25
    prev_avg_score = None
    score_improvement_rate = 0.0
    seen_inchikeys = set()
    start = time.time()
    total_time = 0
    total_requested = 0
    total_unique = 0

    n_samples_first_iteration = n_samples if config["allowed_reaction"] == "rxn:5" else n_samples*4
    
    bt.logging.info(f"[Miner] Starting optimization: {len(target_models)} target, {len(antitarget_models)} antitarget models")
    
    # File writing control
    FILE_WRITE_DELAY = 28 * 60  # 25 minutes in seconds
    file_writing_enabled = False
    
    while time.time() - start < 1800:
        iter_start = time.time()
        iteration += 1
        bt.logging.info(f"[Miner] Iteration {iteration}: sampling {n_samples_first_iteration if iteration == 1 else n_samples} molecules")
        
        # Enable file writing after 25 minutes
        elapsed_time = time.time() - start
        if not file_writing_enabled and elapsed_time >= FILE_WRITE_DELAY:
            file_writing_enabled = True
            bt.logging.info(f"[Miner] File writing enabled at {elapsed_time/60:.1f} minutes")
        
        # Build component weights from top pool for score-guided sampling
        component_weights = build_component_weights(top_pool, rxn_id) if not top_pool.empty else None
        
        # Select diverse elites (not just top by score)
        elite_df = select_diverse_elites(top_pool, min(100, len(top_pool))) if not top_pool.empty else pd.DataFrame()
        elite_names = elite_df["name"].tolist() if not elite_df.empty else None
        
        # Adaptive sampling: adjust based on score improvement
        if prev_avg_score is not None and not top_pool.empty:
            current_avg = top_pool['score'].mean()
            score_improvement_rate = (current_avg - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
            
            # If improving well, increase exploitation; if stagnating, increase exploration
            if score_improvement_rate > 0.01:  # Good improvement
                elite_frac = min(0.7, elite_frac * 1.1)
                mutation_prob = max(0.05, mutation_prob * 0.95)
            elif score_improvement_rate < -0.01:  # Declining
                elite_frac = max(0.2, elite_frac * 0.9)
                mutation_prob = min(0.4, mutation_prob * 1.1)
        
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
            component_weights=component_weights
        )
        
        if data.empty:
            continue

        total_requested += len(data)

        try:
            filtered_data = data[~data['InChIKey'].isin(seen_inchikeys)]
            total_unique += len(filtered_data)

            dup_ratio = (len(data) - len(filtered_data)) / max(1, len(data))
            if dup_ratio > 0.6:
                mutation_prob = min(0.5, mutation_prob * 1.5)
                elite_frac = max(0.2, elite_frac * 0.8)
            elif dup_ratio < 0.2 and not top_pool.empty:
                mutation_prob = max(0.05, mutation_prob * 0.9)
                elite_frac = min(0.8, elite_frac * 1.1)

            data = filtered_data

        except Exception as e:
            bt.logging.error(f"Deduplication failed: {e}")

        data = data.reset_index(drop=True)
        
        if data.empty:
            continue
        
       # PARALLEL SCORING
        target_scores, antitarget_scores = parallel_score_molecules(data['smiles'], config)

        # RESET INDICES
        target_scores = target_scores.reset_index(drop=True)
        antitarget_scores = antitarget_scores.reset_index(drop=True)
        data = data.reset_index(drop=True)

        # ASSIGN SCORES
        data['Target'] = target_scores
        data['Anti'] = antitarget_scores
        data['score'] = data['Target'] - (config['antitarget_weight'] * data['Anti'])

        seen_inchikeys.update([k for k in data["InChIKey"].tolist() if k])

        # UPDATE TOP POOL
        total_data = data[["name", "smiles", "InChIKey", "score", "Target", "Anti"]]
        top_pool = pd.concat([top_pool, total_data])
        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
        top_pool = top_pool.sort_values(by="score", ascending=False)
        top_pool = top_pool.head(config["num_molecules"])

        # CALCULATE STATISTICS
        avg_score = float(top_pool['score'].mean()) if not top_pool.empty else 0.0
        max_score = float(top_pool['score'].max()) if not top_pool.empty else 0.0
        min_score = float(top_pool['score'].min()) if not top_pool.empty else 0.0

        # CALCULATE IMPROVEMENT
        improvement_pct = 0.0
        if prev_avg_score not in (None, 0):
            improvement_pct = (avg_score - prev_avg_score) / abs(prev_avg_score)
            bt.logging.info(f"[DEBUG] Improvement: {prev_avg_score:.4f} → {avg_score:.4f} ({improvement_pct*100:+.2f}%)")

        prev_avg_score = avg_score

        iter_end = time.time()
        total_time += iter_end - iter_start
        print(
            f"[Miner] Iter {iteration} | "
            f"Avg: {avg_score:.6f} | Best: {max_score:.6f} | "
            f"Improvement: {improvement_pct*100:+.2f}% | "
            f"Time: {iter_end - iter_start:.2f}s | Total: {total_time:.2f}s | "
            f"Pool: {len(top_pool)} | Samples: {n_samples_first_iteration if iteration == 1 else n_samples} | "
            f"Requested: {total_requested} | Unique: {total_unique}"
        )

        
        # Save results ONLY after 25 minutes
        if file_writing_enabled:
            top_entries = {"molecules": top_pool["name"].tolist()}
            tmp_path = os.path.join(OUTPUT_DIR, "result.json.tmp")
            final_path = os.path.join(OUTPUT_DIR, "result.json")
            with open(tmp_path, "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, final_path)
    

if __name__ == "__main__":
    config = get_config()
    start_time_1 = time.time()
    initialize_models(config)
    bt.logging.info(f"[Miner] Model initialization: {time.time() - start_time_1:.2f}s")
    main(config)
