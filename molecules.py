from rdkit import Chem, DataStructs
import bittensor as bt
from rdkit.Chem import Descriptors, MACCSkeys
from dotenv import load_dotenv
import pandas as pd
import warnings
import sqlite3
import random
import os
from functools import lru_cache
from typing import List, Tuple, Dict
load_dotenv(override=True)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction, get_reaction_info
from nova_ph2.utils.molecules import get_heavy_atom_count
from collections import defaultdict
from itertools import chain


@lru_cache(maxsize=1000_000)
def _get_smiles_from_reaction_cached(name: str):
    """Cache SMILES retrieval to avoid repeated database queries."""
    try:
        return get_smiles_from_reaction(name)
    except Exception:
        return None

@lru_cache(maxsize=1000_000)
def _mol_from_smiles_cached(smiles: str):
    """Cache molecule parsing to avoid repeated SMILES parsing."""
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


@lru_cache(maxsize=1000_000)
def _maccs_fp_from_smiles_cached(smiles: str):
    """Cache MACCS fingerprints for SMILES strings for fast Tanimoto similarity."""
    if not smiles:
        return None
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return None
        return MACCSkeys.GenMACCSKeys(mol)
    except Exception:
        return None

@lru_cache(maxsize=1000_000)
def _inchikey_from_name_cached(name: str) -> str:
    """Cache InChIKey generation from molecule name to avoid repeated computation."""
    try:
        s = _get_smiles_from_reaction_cached(name)
        if not s:
            return ""
        return generate_inchikey(s)
    except Exception:
        return ""

def num_rotatable_bonds(smiles: str) -> int:
    """Get number of rotatable bonds from SMILES string."""
    if not smiles:
        return 0
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return 0
        return Descriptors.NumRotatableBonds(mol)
    except Exception:
        return 0

def generate_inchikey(smiles: str) -> str:
    """Generate InChIKey from SMILES string."""
    if not smiles:
        return ""
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return ""
        return Chem.MolToInchiKey(mol)
    except Exception as e:
        bt.logging.error(f"Error generating InChIKey for SMILES {smiles}: {e}")
        return ""


def compute_tanimoto_similarity_to_pool(
    candidate_smiles: pd.Series,
    pool_smiles: pd.Series,
) -> pd.Series:
    """
    Compute, for each candidate SMILES, the maximum MACCS Tanimoto similarity
    to any molecule in the reference pool.

    Returns a Series indexed like candidate_smiles.
    """
    if candidate_smiles.empty or pool_smiles.empty:
        # Return zeros with matching index
        return pd.Series(0.0, index=candidate_smiles.index, dtype=float)

    # Pre-compute fingerprints for pool molecules
    pool_fps = []
    for smi in pool_smiles.dropna().unique():
        fp = _maccs_fp_from_smiles_cached(smi)
        if fp is not None:
            pool_fps.append(fp)

    if not pool_fps:
        return pd.Series(0.0, index=candidate_smiles.index, dtype=float)

    similarities = {}
    for idx, smi in candidate_smiles.items():
        fp_cand = _maccs_fp_from_smiles_cached(smi)
        if fp_cand is None:
            similarities[idx] = 0.0
            continue
        max_sim = 0.0
        for fp_ref in pool_fps:
            try:
                sim = DataStructs.TanimotoSimilarity(fp_cand, fp_ref)
            except Exception:
                sim = 0.0
            if sim > max_sim:
                max_sim = sim
        similarities[idx] = float(max_sim)

    return pd.Series(similarities, dtype=float)

seen_cache = {}

def sample_random_valid_molecules(
    n_samples: int,
    subnet_config: dict,
    avoid_inchikeys: set[str] | None = None,
    focus_neighborhood_of: pd.DataFrame | None = None,
) -> pd.DataFrame:
    global seen_cache
    """
    Sample random valid molecules for a reaction, without using elites or
    component weights. This is intended to provide a "pure" random pool
    for similarity-based selection.
    
    If focus_neighborhood_of is provided, we'll focus sampling around the
    neighborhood of those molecules by generating names in the vicinity.
    Excludes previously seen molecules based on avoid_inchikeys.

    Also tracks how many times a molecule has been seen and adjusts the neighborhood
    sampling range based on the count of times it has been seen.
    """

    # Extract neighborhoods from each name in the focus DataFrame
    names = []
    bt.logging.info(f"Cache : {seen_cache}")
    for name in focus_neighborhood_of["name"]:
        try:
            parts = name.split(":")
            if len(parts) == 4:
                rxn_prefix, rxn_type, comp1_id, comp2_id = parts
                comp1_id = int(comp1_id)
                comp2_id = int(comp2_id)
                
                # Check if this molecule has been seen before, and adjust range accordingly
                seen_count = seen_cache.get(name, 0) + 1
                seen_cache[name] = seen_count

                comp1_range = chain(range(max(1, comp1_id - seen_count * n_samples), comp1_id - (seen_count-1) * n_samples), range(max(1, comp1_id + (seen_count - 1) * n_samples), comp1_id + seen_count * n_samples + 1))
                bt.logging.info(f"Range: ({comp1_id - seen_count * n_samples},{comp1_id - (seen_count-1) * n_samples})  and ({comp1_id + (seen_count - 1) * n_samples},{comp1_id + seen_count * n_samples + 1})")
                for new_comp1 in comp1_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{new_comp1}:{comp2_id}"
                    if avoid_inchikeys and new_name in avoid_inchikeys:
                        continue  # Skip if this molecule has already been seen
                    names.append(new_name)
                
                # Generate neighborhood around comp2_id (keep comp1_id fixed)
                comp2_range = chain(range(max(1, comp2_id - seen_count * n_samples), comp2_id - (seen_count-1) * n_samples), range(max(1, comp2_id + (seen_count - 1) * n_samples), comp2_id + seen_count * n_samples + 1))
                for new_comp2 in comp2_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{comp1_id}:{new_comp2}"
                    if avoid_inchikeys and new_name in avoid_inchikeys:
                        continue  # Skip if this molecule has already been seen
                    names.append(new_name)
                
            if len(parts) == 5:
                rxn_prefix, rxn_type, comp1_id, comp2_id, comp3_id = parts
                comp1_id = int(comp1_id)
                comp2_id = int(comp2_id)
                comp3_id = int(comp3_id)
                
                # Check if this molecule has been seen before, and adjust range accordingly
                seen_count = seen_cache.get(name, 0) + 1
                seen_cache[name] = seen_count
                # Generate neighborhood around comp1_id (keep comp2_id and comp3_id fixed)
                comp1_range = chain(range(max(1, comp1_id - seen_count * n_samples), comp1_id - (seen_count-1) * n_samples), range(max(1, comp1_id + (seen_count - 1) * n_samples), comp1_id + seen_count * n_samples + 1))
                for new_comp1 in comp1_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{new_comp1}:{comp2_id}:{comp3_id}"
                    if avoid_inchikeys and new_name in avoid_inchikeys:
                        continue  # Skip if this molecule has already been seen
                    names.append(new_name)
                
                # Generate neighborhood around comp2_id (keep comp1_id and comp3_id fixed)
                comp2_range = chain(range(max(1, comp2_id - seen_count * n_samples), comp2_id - (seen_count-1) * n_samples), range(max(1, comp2_id + (seen_count - 1) * n_samples), comp2_id + seen_count * n_samples + 1))
                for new_comp2 in comp2_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{comp1_id}:{new_comp2}:{comp3_id}"
                    if avoid_inchikeys and new_name in avoid_inchikeys:
                        continue  # Skip if this molecule has already been seen
                    names.append(new_name)
                
                # Generate neighborhood around comp3_id (keep comp1_id and comp2_id fixed)
                comp3_range = chain(range(max(1, comp3_id - seen_count * n_samples), comp3_id - (seen_count-1) * n_samples), range(max(1, comp3_id + (seen_count - 1) * n_samples), comp3_id + seen_count * n_samples + 1))
                for new_comp3 in comp3_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{comp1_id}:{comp2_id}:{new_comp3}"
                    if avoid_inchikeys and new_name in avoid_inchikeys:
                        continue  # Skip if this molecule has already been seen
                    names.append(new_name)

        except (ValueError, IndexError) as e:
            bt.logging.warning(f"Could not parse name '{name}': {e}")
            continue
    
    if not names:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    df = pd.DataFrame({"name": names})
    
    df = df[df["name"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    df = validate_molecules(df, subnet_config)
    if df.empty:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    df = df.drop_duplicates(subset=["InChIKey"], keep="first")

    if avoid_inchikeys:
        df = df[~df["InChIKey"].isin(avoid_inchikeys)]

    return df[["name", "smiles", "InChIKey"]].head(n_samples).copy()



def validate_molecules(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Validate molecules by checking heavy atom count and rotatable bonds.
    Returns DataFrame with validated molecules and their descriptors.
    Defer InChIKey generation until after validation to avoid waste.
    """
    if data.empty:
        return data
    
    data = data.copy()
    data['smiles'] = data["name"].apply(_get_smiles_from_reaction_cached)
    
    data = data[data['smiles'].notna()]
    if data.empty:
        return data
    
    data['heavy_atoms'] = data["smiles"].apply(get_heavy_atom_count)
    data['bonds'] = data["smiles"].apply(num_rotatable_bonds)
    
    mask = (
        (data['heavy_atoms'] >= config['min_heavy_atoms']) &
        (data['bonds'] >= config['min_rotatable_bonds']) &
        (data['bonds'] <= config['max_rotatable_bonds'])
    )
    data = data[mask]
    
    if not data.empty:
        data['InChIKey'] = data["smiles"].apply(generate_inchikey)
    
    return data


@lru_cache(maxsize=None)
def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
    try:
        abs_db_path = os.path.abspath(db_path)
        with sqlite3.connect(f"file:{abs_db_path}?mode=ro&immutable=1", uri=True) as conn:
            conn.execute("PRAGMA query_only = ON")
            cursor = conn.cursor()
            cursor.execute(
                "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?", 
                (role_mask, role_mask)
            )
            results = cursor.fetchall()
        return results
    except Exception as e:
        bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
        return []

def generate_valid_random_molecules_batch(
    rxn_id: int,
    n_samples: int,
    db_path: str,
    subnet_config: dict,
    batch_size: int = 200,
    seed: int = None,
    elite_names: list[str] | None = None,
    elite_frac: float = 0.5,
    mutation_prob: float = 0.1,
    avoid_inchikeys: set[str] | None = None,
    component_weights: dict | None = None,
) -> pd.DataFrame:
    
    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        bt.logging.error(f"Could not get reaction info for rxn_id {rxn_id}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    
    smarts, roleA, roleB, roleC = reaction_info
    is_three_component = roleC is not None and roleC != 0
    
    molecules_A = get_molecules_by_role(roleA, db_path)
    molecules_B = get_molecules_by_role(roleB, db_path)
    molecules_C = get_molecules_by_role(roleC, db_path) if is_three_component else []

    if not molecules_A or not molecules_B or (is_three_component and not molecules_C):
        bt.logging.error(f"No molecules found for roles A={roleA}, B={roleB}, C={roleC}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    elite_As, elite_Bs, elite_Cs = set(), set(), set()
    if elite_names:
        for name in elite_names:
            A, B, C = _parse_components(name)
            if A is not None: 
                elite_As.add(A)
            if B is not None: 
                elite_Bs.add(B)
            if C is not None and is_three_component: 
                elite_Cs.add(C)

    pool_A_ids = _ids_from_pool(molecules_A)
    pool_B_ids = _ids_from_pool(molecules_B)
    pool_C_ids = _ids_from_pool(molecules_C) if is_three_component else []

    valid_dfs = []
    seen_keys = set()
    total_valid = 0

    while total_valid < n_samples:
        needed = n_samples - total_valid
        batch_size_actual = min(max(batch_size, 300), needed * 2)
        
        emitted_names = set()
        if elite_names:
            n_elite = max(0, min(batch_size_actual, int(batch_size_actual * elite_frac)))
            n_rand = batch_size_actual - n_elite

            elite_batch = generate_offspring_from_elites(
                rxn_id=rxn_id,
                n=n_elite,
                pool_A_ids=pool_A_ids,
                pool_B_ids=pool_B_ids,
                pool_C_ids=pool_C_ids,
                is_three_component=is_three_component,
                mutation_prob=mutation_prob,
                seed=seed,
                avoid_names=emitted_names,
                avoid_inchikeys=avoid_inchikeys,
                max_tries=10,
                elite_As=elite_As,
                elite_Bs=elite_Bs,
                elite_Cs=elite_Cs,
            )
            emitted_names.update(elite_batch)

            rand_batch = generate_molecules_from_pools(
                rxn_id, n_rand, molecules_A, molecules_B, molecules_C, is_three_component, seed, component_weights
            )
            rand_batch = [n for n in rand_batch if n and (n not in emitted_names)]
            batch_molecules = elite_batch + rand_batch
        else:
            batch_molecules = generate_molecules_from_pools(
                rxn_id, batch_size_actual, molecules_A, molecules_B, molecules_C, is_three_component, seed, component_weights
            )
        
        if not batch_molecules:
            continue
            
        batch_df = pd.DataFrame({"name": batch_molecules})
        batch_df = batch_df[batch_df["name"].notna()]  # Remove None values
        if batch_df.empty:
            continue
            
        batch_df = validate_molecules(batch_df, subnet_config)
        
        if batch_df.empty:
            continue

        batch_df = batch_df.drop_duplicates(subset=["InChIKey"], keep="first")
        
        mask = ~batch_df["InChIKey"].isin(seen_keys)
        if avoid_inchikeys:
            mask = mask & ~batch_df["InChIKey"].isin(avoid_inchikeys)
        batch_df = batch_df[mask]
        
        if batch_df.empty:
            continue
        
        seen_keys.update(batch_df["InChIKey"].values)
        valid_dfs.append(batch_df[["name", "smiles", "InChIKey"]].copy())
        total_valid += len(batch_df)
        
        if total_valid >= n_samples:
            break
        
    if not valid_dfs:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    
    # Concatenate all DataFrames at once
    result_df = pd.concat(valid_dfs, ignore_index=True)
    return result_df.head(n_samples).copy()


def generate_molecules_from_pools(rxn_id: int, n: int, molecules_A: List[Tuple], molecules_B: List[Tuple], 
                                molecules_C: List[Tuple], is_three_component: bool, seed: int = None,
                                component_weights: dict = None) -> List[str]:
    
    rng = random.Random(seed) if seed is not None else random
    
    A_ids = [a[0] for a in molecules_A]
    B_ids = [b[0] for b in molecules_B]
    C_ids = [c[0] for c in molecules_C] if is_three_component else None
    
    # Use weighted sampling if component weights are provided
    if component_weights:
        # Build weights for each component pool
        weights_A = [component_weights.get('A', {}).get(aid, 1.0) for aid in A_ids]
        weights_B = [component_weights.get('B', {}).get(bid, 1.0) for bid in B_ids]
        weights_C = [component_weights.get('C', {}).get(cid, 1.0) for cid in C_ids] if is_three_component else None
        
        # Normalize weights
        if weights_A:
            sum_w = sum(weights_A)
            weights_A = [w / sum_w if sum_w > 0 else 1.0/len(weights_A) for w in weights_A]
        if weights_B:
            sum_w = sum(weights_B)
            weights_B = [w / sum_w if sum_w > 0 else 1.0/len(weights_B) for w in weights_B]
        if weights_C:
            sum_w = sum(weights_C)
            weights_C = [w / sum_w if sum_w > 0 else 1.0/len(weights_C) for w in weights_C]
        
        picks_A = rng.choices(A_ids, weights=weights_A, k=n) if weights_A else rng.choices(A_ids, k=n)
        picks_B = rng.choices(B_ids, weights=weights_B, k=n) if weights_B else rng.choices(B_ids, k=n)
        if is_three_component:
            picks_C = rng.choices(C_ids, weights=weights_C, k=n) if weights_C else rng.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    else:
        # Uniform random sampling
        picks_A = rng.choices(A_ids, k=n)
        picks_B = rng.choices(B_ids, k=n)
        if is_three_component:
            picks_C = rng.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    
    # Remove duplicates while preserving order
    names = list(dict.fromkeys(names))
    return names

def _parse_components(name: str) -> tuple[int, int, int | None]:
    # name format: "rxn:{rxn_id}:{A}:{B}" or "rxn:{rxn_id}:{A}:{B}:{C}"
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None
    A = int(parts[2]); B = int(parts[3])
    C = int(parts[4]) if len(parts) > 4 else None
    return A, B, C

def _ids_from_pool(pool):
    return [x[0] for x in pool]

def generate_offspring_from_elites(rxn_id: int, n: int,
                                   is_three_component: bool,
                                   pool_A_ids:list,
                                   pool_B_ids:list,
                                   pool_C_ids:list,
                                   mutation_prob: float = 0.1, seed: int | None = None,
                                   avoid_names: set[str] = None,
                                   avoid_inchikeys: set[str] = None,
                                   max_tries: int = 10,
                                   elite_As: set[int] = None,
                                   elite_Bs: set[int] = None,
                                   elite_Cs: set[int] = None) -> list[str]:
    
    rng = random.Random(seed) if seed is not None else random
    
    elite_As_list = list(elite_As) if elite_As else []
    elite_Bs_list = list(elite_Bs) if elite_Bs else []
    elite_Cs_list = list(elite_Cs) if elite_Cs else []

    out = []
    local_names = set()
    check_inchikeys = avoid_inchikeys is not None and len(avoid_inchikeys) > 0
    
    for _ in range(n):
        cand = None
        name = None
        for _try in range(max_tries):
            use_mutA = (not elite_As) or (rng.random() < mutation_prob)
            use_mutB = (not elite_Bs) or (rng.random() < mutation_prob)
            use_mutC = (not elite_Cs) or (rng.random() < mutation_prob)

            A = rng.choice(pool_A_ids) if use_mutA else rng.choice(elite_As_list)
            B = rng.choice(pool_B_ids) if use_mutB else rng.choice(elite_Bs_list)
            if is_three_component:
                C = rng.choice(pool_C_ids) if use_mutC else rng.choice(elite_Cs_list)
                name = f"rxn:{rxn_id}:{A}:{B}:{C}"
            else:
                name = f"rxn:{rxn_id}:{A}:{B}"

            # Fast checks first (set membership is O(1))
            if avoid_names and name in avoid_names:
                continue
            if name in local_names:
                continue

            if check_inchikeys:
                try:
                    key = _inchikey_from_name_cached(name)
                    if key and key in avoid_inchikeys:
                        continue
                except Exception:
                    pass

            cand = name
            break

        if cand is None:
            if name is None:
                A = rng.choice(pool_A_ids)
                B = rng.choice(pool_B_ids)
                if is_three_component:
                    C = rng.choice(pool_C_ids) if pool_C_ids else 0
                    name = f"rxn:{rxn_id}:{A}:{B}:{C}"
                else:
                    name = f"rxn:{rxn_id}:{A}:{B}"
            cand = name
        out.append(cand)
        local_names.add(cand)
        if avoid_names is not None:
            avoid_names.add(cand)
    return out

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
                if len(selected) < n_elites:
                    selected.append(idx)
    
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx not in selected:
            selected.append(idx)
    
    return candidates.loc[selected[:n_elites]] if selected else candidates.head(n_elites)


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
