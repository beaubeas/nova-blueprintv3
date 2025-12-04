"""
Smart Sampler v2 - ZERO DUPLICATES

Key changes:
- Accepts global seen_names set - NEVER generates duplicates
- Keeps generating until target UNIQUE count is reached
- Validates inline - only returns valid, unique, scored-ready molecules
- No more filtering needed in miner!
"""

import sqlite3
import random
import os
import json
from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict
from functools import lru_cache
import bittensor as bt
from rdkit import Chem
from rdkit.Chem import Descriptors

import sys
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from nova_ph2.combinatorial_db.reactions import get_reaction_info, get_smiles_from_reaction
from nova_ph2.utils import get_heavy_atom_count


class EliteLearner:
    """Tracks elite molecules and adapts sampling strategy"""

    def __init__(self):
        self.elite_molecules = []  # (name, score, inchikey)
        self.elite_frac = 0.30
        self.mutation_prob = 0.10
        self.iteration = 0

    def update_elites(self, molecules_with_scores):
        """Update elite pool with new scored molecules"""
        for name, score, inchikey in molecules_with_scores:
            self.elite_molecules.append((name, score, inchikey))

        # Sort by score and keep top 100
        self.elite_molecules.sort(key=lambda x: x[1], reverse=True)
        self.elite_molecules = self.elite_molecules[:100]

        if self.elite_molecules:
            bt.logging.info(f"🏆 Elite pool: {len(self.elite_molecules)} molecules, best={self.elite_molecules[0][1]:.4f}")


# Global learner
_learner = EliteLearner()


@lru_cache(maxsize=500_000)
def get_smiles_from_reaction_cached(name: str) -> Optional[str]:
    """Cached SMILES lookup"""
    try:
        return get_smiles_from_reaction(name)
    except Exception:
        return None


@lru_cache(maxsize=500_000)
def get_inchikey_cached(smiles: str) -> Optional[str]:
    """Cached InChIKey calculation"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToInchiKey(mol)
    except Exception:
        pass
    return None


@lru_cache(maxsize=500_000)
def validate_molecule_cached(smiles: str, min_heavy: int, min_rot: int, max_rot: int) -> bool:
    """Cached molecule validation"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False

        # Check heavy atoms
        heavy = get_heavy_atom_count(smiles)
        if heavy < min_heavy:
            return False

        # Check rotatable bonds
        rot = Descriptors.NumRotatableBonds(mol)
        if rot < min_rot or rot > max_rot:
            return False

        return True
    except Exception:
        return False


@lru_cache(maxsize=1024)
def get_molecules_by_role(role_mask: int, db_path: str) -> Tuple[Tuple[int, str, int], ...]:
    """Get molecules by role (returns tuple for caching)"""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?",
            (role_mask, role_mask)
        )
        results = tuple(cursor.fetchall())
        conn.close()
        return results
    except Exception as e:
        bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
        return ()


def generate_unique_molecules(
    n_target: int,
    rxn_id: int,
    db_path: str,
    config: dict,
    seen_names: Set[str],
    seen_inchikeys: Set[str],
    component_weights: Dict = None,
    learner: EliteLearner = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Generate exactly n_target UNIQUE, VALID molecules.

    Returns: (names, smiles, inchikeys) - all guaranteed unique and valid
    """

    # Get reaction info
    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        bt.logging.error(f"Could not get reaction info for rxn_id {rxn_id}")
        return [], [], []

    smarts, roleA, roleB, roleC = reaction_info
    is_3_component = roleC is not None and roleC != 0

    # Get molecule pools
    molecules_A = list(get_molecules_by_role(roleA, db_path))
    molecules_B = list(get_molecules_by_role(roleB, db_path))
    molecules_C = list(get_molecules_by_role(roleC, db_path)) if is_3_component else []

    if not molecules_A or not molecules_B or (is_3_component and not molecules_C):
        bt.logging.error(f"No molecules found for reaction roles")
        return [], [], []

    # Extract IDs
    A_ids = [mol[0] for mol in molecules_A]
    B_ids = [mol[0] for mol in molecules_B]
    C_ids = [mol[0] for mol in molecules_C] if is_3_component else []

    # Build component weights
    weights_A = None
    weights_B = None
    weights_C = None

    if component_weights:
        weights_A = [component_weights.get('A', {}).get(aid, 0.1) for aid in A_ids]
        weights_B = [component_weights.get('B', {}).get(bid, 0.1) for bid in B_ids]
        sum_A = sum(weights_A) or 1.0
        sum_B = sum(weights_B) or 1.0
        weights_A = [w / sum_A for w in weights_A]
        weights_B = [w / sum_B for w in weights_B]

        if is_3_component:
            weights_C = [component_weights.get('C', {}).get(cid, 0.1) for cid in C_ids]
            sum_C = sum(weights_C) or 1.0
            weights_C = [w / sum_C for w in weights_C]

    # Extract elite components if available
    elite_As = set()
    elite_Bs = set()
    elite_Cs = set()

    if learner and learner.elite_molecules:
        for name, score, _ in learner.elite_molecules[:50]:
            try:
                parts = name.split(":")
                if len(parts) >= 4:
                    elite_As.add(int(parts[2]))
                    elite_Bs.add(int(parts[3]))
                    if len(parts) > 4:
                        elite_Cs.add(int(parts[4]))
            except (ValueError, IndexError):
                continue

    # Validation config
    min_heavy = config.get('min_heavy_atoms', 10)
    min_rot = config.get('min_rotatable_bonds', 0)
    max_rot = config.get('max_rotatable_bonds', 50)

    # Generate until we have n_target unique valid molecules
    result_names = []
    result_smiles = []
    result_inchikeys = []

    local_seen_names = set()  # Track within this batch too
    attempts = 0
    max_attempts = n_target * 20  # Allow plenty of attempts

    # Calculate elite vs random ratio
    elite_frac = learner.elite_frac if learner else 0.3
    mutation_prob = learner.mutation_prob if learner else 0.1

    while len(result_names) < n_target and attempts < max_attempts:
        attempts += 1

        # Decide: elite-based or random
        use_elite = (elite_As and random.random() < elite_frac)

        if use_elite:
            # Elite-based generation with mutation
            use_elite_A = elite_As and random.random() > mutation_prob
            use_elite_B = elite_Bs and random.random() > mutation_prob
            use_elite_C = elite_Cs and random.random() > mutation_prob

            A = random.choice(list(elite_As)) if use_elite_A else random.choice(A_ids)
            B = random.choice(list(elite_Bs)) if use_elite_B else random.choice(B_ids)

            if is_3_component:
                C = random.choice(list(elite_Cs)) if use_elite_C and elite_Cs else random.choice(C_ids)
                name = f"rxn:{rxn_id}:{A}:{B}:{C}"
            else:
                name = f"rxn:{rxn_id}:{A}:{B}"
        else:
            # Weighted random generation
            if weights_A:
                A = random.choices(A_ids, weights=weights_A, k=1)[0]
                B = random.choices(B_ids, weights=weights_B, k=1)[0]
                if is_3_component:
                    C = random.choices(C_ids, weights=weights_C, k=1)[0]
                    name = f"rxn:{rxn_id}:{A}:{B}:{C}"
                else:
                    name = f"rxn:{rxn_id}:{A}:{B}"
            else:
                A = random.choice(A_ids)
                B = random.choice(B_ids)
                if is_3_component:
                    C = random.choice(C_ids)
                    name = f"rxn:{rxn_id}:{A}:{B}:{C}"
                else:
                    name = f"rxn:{rxn_id}:{A}:{B}"

        # Quick check: already seen this name?
        if name in seen_names or name in local_seen_names:
            continue

        # Get SMILES
        smiles = get_smiles_from_reaction_cached(name)
        if not smiles:
            continue

        # Get InChIKey
        inchikey = get_inchikey_cached(smiles)
        if not inchikey:
            continue

        # Check: already seen this InChIKey?
        if inchikey in seen_inchikeys:
            # Mark name as seen too (different name, same molecule)
            local_seen_names.add(name)
            continue

        # Validate molecule
        if not validate_molecule_cached(smiles, min_heavy, min_rot, max_rot):
            local_seen_names.add(name)  # Don't try this name again
            continue

        # SUCCESS! This molecule is unique and valid
        result_names.append(name)
        result_smiles.append(smiles)
        result_inchikeys.append(inchikey)
        local_seen_names.add(name)

    bt.logging.info(f"   Generated {len(result_names)} unique valid molecules in {attempts} attempts ({attempts/max(len(result_names),1):.1f}x overhead)")

    return result_names, result_smiles, result_inchikeys


def run_sampler(
    n_samples: int,
    subnet_config: dict,
    db_path: str,
    seen_names: Set[str] = None,
    seen_inchikeys: Set[str] = None,
    component_weights: dict = None,
    output_path: str = None,
    save_to_file: bool = False,
    **kwargs  # Ignore other args for compatibility
) -> dict:
    """
    Generate n_samples UNIQUE, VALID molecules.

    No filtering needed after this - all returned molecules are ready to score!

    Args:
        n_samples: Target number of unique valid molecules
        subnet_config: Config with reaction, validation params
        db_path: Path to molecules database
        seen_names: Global set of already-seen molecule names (will be updated)
        seen_inchikeys: Global set of already-seen InChIKeys (will be updated)
        component_weights: Optional weights for biased sampling

    Returns:
        dict with 'molecules', 'smiles', 'inchikeys' - all unique and valid
    """
    global _learner

    if subnet_config is None:
        bt.logging.error("subnet_config is required")
        return {"molecules": [], "smiles": [], "inchikeys": []}

    # Initialize seen sets if not provided
    if seen_names is None:
        seen_names = set()
    if seen_inchikeys is None:
        seen_inchikeys = set()

    # Get reaction ID
    allowed_reaction = subnet_config.get("allowed_reaction", "")
    if not allowed_reaction or not allowed_reaction.startswith("rxn:"):
        bt.logging.error(f"Invalid allowed_reaction: {allowed_reaction}")
        return {"molecules": [], "smiles": [], "inchikeys": []}

    try:
        rxn_id = int(allowed_reaction.split(":")[-1])
    except (ValueError, IndexError):
        bt.logging.error(f"Could not parse reaction ID from: {allowed_reaction}")
        return {"molecules": [], "smiles": [], "inchikeys": []}

    _learner.iteration += 1
    bt.logging.info(f"🧬 Smart Sampler v2 - Iteration {_learner.iteration}")
    bt.logging.info(f"   Target: {n_samples} unique valid molecules")
    bt.logging.info(f"   Already seen: {len(seen_names)} names, {len(seen_inchikeys)} InChIKeys")

    # Generate unique valid molecules
    names, smiles_list, inchikeys = generate_unique_molecules(
        n_target=n_samples,
        rxn_id=rxn_id,
        db_path=db_path,
        config=subnet_config,
        seen_names=seen_names,
        seen_inchikeys=seen_inchikeys,
        component_weights=component_weights,
        learner=_learner
    )

    # Update global seen sets
    seen_names.update(names)
    seen_inchikeys.update(inchikeys)

    bt.logging.info(f"   ✅ Returning {len(names)} unique valid molecules (0 duplicates!)")

    result = {
        "molecules": names,
        "smiles": smiles_list,
        "inchikeys": inchikeys
    }

    if save_to_file and output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def update_learner_with_scores(molecules_with_scores: List[Tuple[str, float]]):
    """Update learner with scored molecules"""
    global _learner

    elite_data = []
    for name, score in molecules_with_scores:
        smiles = get_smiles_from_reaction_cached(name)
        if smiles:
            inchikey = get_inchikey_cached(smiles)
            if inchikey:
                elite_data.append((name, score, inchikey))

    if elite_data:
        _learner.update_elites(elite_data)
        bt.logging.info(f"🧠 Learner updated with {len(elite_data)} scored molecules")


def reset_learner():
    """Reset the global learner (useful for testing)"""
    global _learner
    _learner = EliteLearner()
