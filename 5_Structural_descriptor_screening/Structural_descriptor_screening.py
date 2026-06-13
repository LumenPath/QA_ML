from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import pandas as pd
import numpy as np
import os
import time
import shutil
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans
from rdkit import DataStructs
import math  # Add the math module to support exponential functions

# ============ Input and Output Parameters ============
# Input file path
INPUT_CSV = "XXX.csv"
# Output file prefix
OUTPUT_PREFIX = "qa_results"
# Output folder
OUTPUT_FOLDER = "QA_Stability_Score"
# Top N structures will be saved
TOP_N_STRUCTURES = 1000
# Number of processes for parallel processing
NUM_PROCESSES = 8
# Whether to use structural clustering to ensure diversity
USE_CLUSTERING = True
# Number of structural clusters (only valid when USE_CLUSTERING=True)
CLUSTERING_COUNT = 50
# Whether to enable early filtering (structures with β-hydrogen > 6 are directly excluded)
ENABLE_EARLY_FILTERING = None  # True
# Number of gjf files in each group
GJF_GROUP_SIZE = 50

# ============ Stability Level Classification ============
SUPER_HIGH_THRESHOLD = 90  # Super-high stability threshold
HIGH_THRESHOLD = 80  # High stability threshold
MEDIUM_THRESHOLD = 70  # Medium stability threshold


# <50 is low stability

# ==========================================

def analyze_quaternary_ammonium_stability(smiles):
    """Analyze the alkaline stability of quaternary ammonium cations"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Confirm whether it is a quaternary ammonium structure
        quat_n_idx = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 1 and atom.GetDegree() == 4:
                quat_n_idx = atom.GetIdx()
                break

        if quat_n_idx is None:
            return {"SMILES": smiles, "Is quaternary ammonium": False}

        # Get quaternary ammonium nitrogen atom
        n_atom = mol.GetAtomWithIdx(quat_n_idx)

        # Determine structure type: whether it is cyclic quaternary ammonium (nitrogen atom is in a ring)
        is_cyclic = n_atom.IsInRing()
        structure_type = "cyclic" if is_cyclic else "linear"

        # Early filtering: if enabled and the structure has β-hydrogens > 6, directly return a low score
        beta_hydrogens = count_beta_hydrogens(mol, quat_n_idx)
        if ENABLE_EARLY_FILTERING and beta_hydrogens["Total β-position hydrogens"] > 6:
            return {
                "SMILES": smiles,
                "Is quaternary ammonium": True,
                "Structure type": structure_type,
                "β-hydrogen analysis": beta_hydrogens,
                "Raw score": 0,
                "Normalized score": 0.00,
                "Stability level": "Low stability"
            }

        # 1. β-hydrogen count - key factor affecting E2 elimination reaction
        # beta_hydrogens has already been calculated above

        # 2. α-carbon substitution degree analysis - affects the possibility of SN2 reactions
        alpha_carbon_substitution = analyze_alpha_carbons(mol, quat_n_idx)

        # 3. Cyclic structure analysis - affects conformational stability
        ring_analysis = analyze_ring_structures(mol, quat_n_idx)

        # 4. Steric hindrance analysis - affects nucleophile approach
        steric_hindrance = analyze_steric_hindrance(mol, quat_n_idx)

        # 5. Linear-chain characteristic analysis - only applicable to linear structures
        chain_characteristics = analyze_chain_characteristics(mol, quat_n_idx) if not is_cyclic else None

        # Calculate alkaline stability score - use different scoring systems
        if is_cyclic:
            stability_score = calculate_cyclic_stability_score(
                beta_hydrogens,
                alpha_carbon_substitution,
                ring_analysis,
                steric_hindrance
            )
        else:
            stability_score = calculate_linear_stability_score(
                beta_hydrogens,
                alpha_carbon_substitution,
                steric_hindrance,
                chain_characteristics
            )

        # Integrate all analysis results
        result = {
            "SMILES": smiles,
            "Is quaternary ammonium": True,
            "Structure type": structure_type,
            "β-hydrogen analysis": beta_hydrogens,
            "α-carbon substitution degree": alpha_carbon_substitution,
            "Ring analysis": ring_analysis,
            "Steric hindrance": steric_hindrance,
            "Raw score": stability_score.get("Raw score", 0),
            "Normalized score": round(stability_score.get("Normalized score", 0), 4),  # Precision to 2 decimal places
            "Stability level": stability_score.get("Stability level", "Unknown"),
            "Factor scores": stability_score.get("Factor scores", {})  # Detailed scores for each factor
        }

        # Add linear-chain characteristic analysis (if applicable)
        if not is_cyclic and chain_characteristics:
            result["Chain characteristics"] = chain_characteristics

        return result

    except Exception as e:
        print(f"Error processing SMILES {smiles}: {str(e)}")
        return {"SMILES": smiles, "Is quaternary ammonium": False, "Error": str(e)}


def count_beta_hydrogens(mol, n_idx):
    """Calculate the number of hydrogen atoms at the β-position of quaternary ammonium cations"""
    # Get carbon atoms connected to nitrogen (α-carbons)
    alpha_carbons = []
    n_atom = mol.GetAtomWithIdx(n_idx)
    for bond in n_atom.GetBonds():
        other_idx = bond.GetOtherAtomIdx(n_idx)
        if mol.GetAtomWithIdx(other_idx).GetSymbol() == 'C':
            alpha_carbons.append(other_idx)

    # Find β-carbons connected to these α-carbons and calculate hydrogens on β-carbons
    beta_h_count = 0
    beta_carbons_with_h = 0

    for alpha_c in alpha_carbons:
        alpha_atom = mol.GetAtomWithIdx(alpha_c)
        for bond in alpha_atom.GetBonds():
            beta_idx = bond.GetOtherAtomIdx(alpha_c)
            if beta_idx != n_idx:  # Not the nitrogen atom
                beta_atom = mol.GetAtomWithIdx(beta_idx)
                if beta_atom.GetSymbol() == 'C':
                    h_count = beta_atom.GetTotalNumHs()
                    beta_h_count += h_count
                    if h_count > 0:
                        beta_carbons_with_h += 1

    # Risk rating (retain the original risk rating for use in analysis reports)
    if beta_h_count == 0:
        risk = "Low"
    elif 1 <= beta_h_count <= 2:
        risk = "Medium-low"
    elif 3 <= beta_h_count <= 4:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "Total β-position hydrogens": beta_h_count,
        "Number of β-carbons with hydrogens": beta_carbons_with_h,
        "E2 elimination risk": risk
    }


def analyze_alpha_carbons(mol, n_idx):
    """Analyze the substitution pattern of α-position carbon atoms"""
    n_atom = mol.GetAtomWithIdx(n_idx)
    alpha_carbons = []

    for bond in n_atom.GetBonds():
        other_idx = bond.GetOtherAtomIdx(n_idx)
        other_atom = mol.GetAtomWithIdx(other_idx)
        if other_atom.GetSymbol() == 'C':
            # Calculate the substitution degree of this α-carbon
            substitution = other_atom.GetDegree() - 1  # Subtract the connection to N
            is_aromatic = other_atom.GetIsAromatic()

            alpha_carbons.append({
                "Atom ID": other_idx,
                "Substitution degree": substitution,
                "Aromaticity": is_aromatic
            })

    # Calculate average substitution degree
    avg_substitution = sum(c["Substitution degree"] for c in alpha_carbons) / len(alpha_carbons) if alpha_carbons else 0

    # SN2 risk rating (retain the original risk rating for use in analysis reports)
    if avg_substitution >= 2.0:
        risk = "Low"
    elif avg_substitution >= 1.5:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "Number of α-carbons": len(alpha_carbons),
        "α-carbon information": alpha_carbons,
        "Average substitution degree": avg_substitution,
        "SN2 risk": risk
    }


def analyze_ring_structures(mol, n_idx):
    """Analyze ring systems in quaternary ammonium structures"""
    n_atom = mol.GetAtomWithIdx(n_idx)

    # Check whether the nitrogen atom is in a ring
    n_in_ring = n_atom.IsInRing()

    # Get ring information for the whole molecule
    ring_info = mol.GetRingInfo().AtomRings()
    n_rings = len(ring_info)

    # Find nitrogen-containing rings
    n_containing_rings = []
    for ring in ring_info:
        if n_idx in ring:
            n_containing_rings.append(len(ring))

    # Analyze whether carbons connected to nitrogen are in rings
    alpha_carbons_in_ring = 0
    for bond in n_atom.GetBonds():
        other_idx = bond.GetOtherAtomIdx(n_idx)
        if mol.GetAtomWithIdx(other_idx).GetSymbol() == 'C' and mol.GetAtomWithIdx(other_idx).IsInRing():
            alpha_carbons_in_ring += 1

    # Ring stabilization effect rating (retain the original rating for use in analysis reports)
    if not n_in_ring:
        ring_effect = "None"
    elif n_containing_rings and 5 <= n_containing_rings[0] <= 7:
        ring_effect = "Optimal"
    elif n_containing_rings and n_containing_rings[0] >= 8:
        ring_effect = "Suboptimal"
    elif n_containing_rings and n_containing_rings[0] == 4:
        ring_effect = "Medium"
    elif n_containing_rings and n_containing_rings[0] == 3:
        ring_effect = "Poor"
    else:
        ring_effect = "Poor"

    return {
        "Nitrogen in ring": n_in_ring,
        "Total number of rings": n_rings,
        "Nitrogen-containing ring sizes": n_containing_rings if n_containing_rings else None,
        "Number of α-carbons in rings": alpha_carbons_in_ring,
        "Ring stabilization effect": ring_effect
    }


def analyze_steric_hindrance(mol, n_idx):
    """Analyze steric hindrance around quaternary ammonium"""
    n_atom = mol.GetAtomWithIdx(n_idx)

    # Get atoms connected to nitrogen
    neighbors = []
    for bond in n_atom.GetBonds():
        other_idx = bond.GetOtherAtomIdx(n_idx)
        neighbors.append(other_idx)

    # Calculate the volume around these atoms
    neighbor_volumes = []
    bulky_groups_count = 0

    for neigh_idx in neighbors:
        # Use a simple method to estimate the size of the group connected to the atom
        group_size = estimate_group_size(mol, neigh_idx, n_idx)
        neighbor_volumes.append(group_size)

        if group_size >= 8:  # Threshold for bulky groups
            bulky_groups_count += 1

    # Calculate average volume
    avg_volume = sum(neighbor_volumes) / len(neighbor_volumes) if neighbor_volumes else 0

    # Evaluate steric hindrance effect (retain the original rating for use in analysis reports)
    steric_effect = "High" if bulky_groups_count >= 2 or avg_volume >= 5 else \
        "Medium" if bulky_groups_count >= 1 or avg_volume >= 3 else \
            "Low"

    return {
        "Substituent volumes": neighbor_volumes,
        "Average volume": avg_volume,
        "Number of bulky groups": bulky_groups_count,
        "Steric hindrance effect": steric_effect
    }


def estimate_group_size(mol, start_idx, exclude_idx):
    """Estimate substituent size - using breadth-first search"""
    visited = set([exclude_idx])
    queue = [start_idx]
    size = 0

    while queue:
        current = queue.pop(0)
        visited.add(current)
        size += 1

        for bond in mol.GetAtomWithIdx(current).GetBonds():
            neighbor = bond.GetOtherAtomIdx(current)
            if neighbor not in visited:
                queue.append(neighbor)

    return size


def analyze_chain_characteristics(mol, n_idx):
    """Analyze chain length and branching characteristics of linear quaternary ammonium structures"""
    n_atom = mol.GetAtomWithIdx(n_idx)

    # Get atoms connected to nitrogen
    chain_lengths = []
    branching_factors = []

    for bond in n_atom.GetBonds():
        other_idx = bond.GetOtherAtomIdx(n_idx)
        if mol.GetAtomWithIdx(other_idx).GetSymbol() == 'C':
            # Calculate chain length
            length = calculate_max_chain_length(mol, other_idx, n_idx)
            chain_lengths.append(length)

            # Calculate branching factor
            branching = calculate_branch_factor(mol, other_idx, n_idx)
            branching_factors.append(branching)

    # Analyze chain length distribution
    avg_length = sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0
    length_variance = calculate_variance(chain_lengths)

    # Analyze branching degree
    avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0

    # Chain length effect rating (retain the original rating for use in analysis reports)
    if 3 <= avg_length <= 6 and avg_branching >= 0.5:
        chain_effect = "Relatively high"
    elif 3 <= avg_length <= 6 and avg_branching < 0.5:
        chain_effect = "Medium"
    elif avg_length >= 4 and length_variance >= 2:
        chain_effect = "Relatively high"
    else:
        chain_effect = "Relatively low"

    return {
        "Average chain length": avg_length,
        "Chain length variability": length_variance,
        "Average branching degree": avg_branching,
        "Chain length effect rating": chain_effect
    }


def calculate_max_chain_length(mol, start_idx, exclude_idx):
    """Calculate the longest chain length starting from the starting atom"""
    visited = set([exclude_idx])
    max_length = 0

    def dfs(current, length):
        nonlocal max_length

        visited.add(current)
        is_terminal = True

        for bond in mol.GetAtomWithIdx(current).GetBonds():
            neighbor = bond.GetOtherAtomIdx(current)
            if neighbor not in visited:
                neighbor_atom = mol.GetAtomWithIdx(neighbor)
                if neighbor_atom.GetSymbol() == 'C':
                    is_terminal = False
                    dfs(neighbor, length + 1)

        if is_terminal and length > max_length:
            max_length = length

    dfs(start_idx, 1)  # Starting atom counts as length 1
    return max_length


def calculate_branch_factor(mol, start_idx, exclude_idx):
    """Calculate branching factor - number of branch points divided by total atom count"""
    visited = set([exclude_idx])
    branch_points = 0
    total_atoms = 0

    def dfs(current):
        nonlocal branch_points, total_atoms

        visited.add(current)
        total_atoms += 1

        # Calculate the number of non-hydrogen connections (excluding already visited atoms)
        connections = 0
        for bond in mol.GetAtomWithIdx(current).GetBonds():
            neighbor = bond.GetOtherAtomIdx(current)
            if neighbor not in visited and mol.GetAtomWithIdx(neighbor).GetSymbol() != 'H':
                connections += 1

        # If the number of connections > 1, it is a branch point
        if connections > 1:
            branch_points += 1

        # Continue DFS
        for bond in mol.GetAtomWithIdx(current).GetBonds():
            neighbor = bond.GetOtherAtomIdx(current)
            if neighbor not in visited and mol.GetAtomWithIdx(neighbor).GetSymbol() != 'H':
                dfs(neighbor)

    dfs(start_idx)
    return branch_points / total_atoms if total_atoms > 0 else 0


def calculate_variance(values):
    """Calculate the variance of a list"""
    if not values or len(values) < 2:
        return 0

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance


# ============ Improved Continuous Scoring Functions ============

def calculate_beta_hydrogen_score(beta_h_count):
    """
    Use the improved β-hydrogen scoring function
    - 40 points when β-hydrogen = 0
    - 30 points when β-hydrogen = 1
    - Then use exponential decay with a decay rate of 0.35
    """
    max_score = 40

    # Piecewise function
    if beta_h_count == 0:
        return 40.00
    elif beta_h_count == 1:
        return 30.00
    else:
        # Use exponential decay when β-hydrogen > 1, starting from 30 points with a decay rate of 0.35
        base_score = 30.00
        decay_rate = 0.35
        score = base_score * math.exp(-decay_rate * (beta_h_count - 1))
        return round(score, 8)  # Retain two decimal places


def calculate_steric_hindrance_score(steric_volume, bulky_groups_count):
    """
    Improved steric hindrance scoring function
    - Increase the exponential coefficient to 2.0 (instead of 1.5)
    - Ensure the score can approach the full 30 points under extreme conditions
    """
    max_score = 30
    volume_factor = min(1.0, steric_volume / 5.0)  # Normalize volume to the 0-1 range
    bulky_factor = min(1.0, bulky_groups_count / 2.0)  # Normalize the number of bulky groups to the 0-1 range

    # Consider both factors using an exponential growth function; coefficient changed from 1.5 to 2.0
    score = max_score * (1 - math.exp(-2.0 * (volume_factor + bulky_factor)))
    return round(score, 8)


def calculate_alpha_carbon_score(avg_substitution):
    """
    Improved α-carbon substitution degree scoring function
    - Lower the full-score requirement to substitution degree 2.5 (instead of 3.0)
    - Increase the slope of the initial part of the curve
    """
    max_score = 15
    min_sub = 0.0
    max_sub = 2.5  # Reduced from 3.0 to 2.5

    # Use an exponential growth function; higher substitution degree gives a higher score
    if avg_substitution >= max_sub:
        return max_score

    # Exponential growth curve; coefficient increased from 2.5 to 3.0 to increase the initial slope
    normalized_sub = avg_substitution / max_sub
    score = max_score * (1 - math.exp(-3.0 * normalized_sub))
    return round(score, 8)


def calculate_ring_score(ring_info):
    """
    Improved ring stabilization effect scoring function
    - Increase Gaussian distribution width, changing σ from 1.0 to 1.5
    - This allows structures with ring sizes 4-8 to obtain relatively high scores
    """
    max_score = 15

    # If nitrogen is not in a ring, the score is 0
    if not ring_info["Nitrogen in ring"]:
        return 0.00

    # If there is no nitrogen-containing ring size information, give the lowest score
    if not ring_info["Nitrogen-containing ring sizes"]:
        return 2.00

    # Take the size of the first nitrogen-containing ring
    ring_size = ring_info["Nitrogen-containing ring sizes"][0]

    # Optimal ring size is 6; use a Gaussian distribution, with σ increased from 1.0 to 1.5
    optimal_size = 6.0
    sigma = 1.5  # Increase distribution width so that 4-8-membered rings have higher scores

    # Calculate score using Gaussian function
    size_factor = math.exp(-((ring_size - optimal_size) ** 2) / (2 * sigma ** 2))
    score = max_score * size_factor
    return round(score, 8)


def calculate_chain_score(avg_length, avg_branching, length_variance=0):
    """
    Simplified chain length characteristic scoring function
    - Optimal chain length range is 3-5 (instead of 3-6)
    - Simplify the calculation logic
    """
    max_score = 15

    # Ideal chain length range changed from 3-6 to 3-5
    length_factor = 0
    if 3 <= avg_length <= 5:
        length_factor = 1.0 - abs(avg_length - 4.0) / 2.0  # 4 is the new ideal center point
    else:
        length_factor = max(0, 1.0 - abs(avg_length - 4.0) / 3.0)

    # Simplified branching factor calculation
    branch_factor = min(1.0, 2.0 * avg_branching) if avg_branching <= 0.5 else (1.0 - min(1.0, avg_branching - 0.5))

    # Simplified calculation, mainly focusing on chain length and branching degree
    combined_factor = (0.7 * length_factor + 0.3 * branch_factor)
    score = max_score * combined_factor
    return round(score, 8)


def calculate_linear_stability_score(beta_h, alpha_c, steric, chain):
    """Calculate stability score for linear quaternary ammonium cations - using improved continuous functions"""
    factors_scores = {}  # Used to record scores for each factor
    score = 0

    # 1. β-hydrogen factor
    beta_h_count = beta_h["Total β-position hydrogens"]
    beta_score = calculate_beta_hydrogen_score(beta_h_count)
    score += beta_score
    factors_scores["β-hydrogen factor"] = beta_score

    # 2. Steric hindrance
    steric_volume = steric["Average volume"]
    bulky_groups = steric["Number of bulky groups"]
    steric_score = calculate_steric_hindrance_score(steric_volume, bulky_groups)
    score += steric_score
    factors_scores["Steric hindrance"] = steric_score

    # 3. Chain length characteristic analysis
    chain_score = 0
    if chain:
        chain_length = chain["Average chain length"]
        chain_branching = chain["Average branching degree"]
        chain_variance = chain["Chain length variability"]
        chain_score = calculate_chain_score(chain_length, chain_branching, chain_variance)
        score += chain_score
        factors_scores["Chain length characteristics"] = chain_score

    # 4. α-carbon substitution degree
    alpha_subst = alpha_c["Average substitution degree"]
    alpha_score = calculate_alpha_carbon_score(alpha_subst)
    score += alpha_score
    factors_scores["α-carbon substitution degree"] = alpha_score

    # Normalized score is not needed; directly use a 100-point scale
    normalized_score = score

    # Stability rating
    if normalized_score >= SUPER_HIGH_THRESHOLD:
        stability_class = "Super-high stability"
    elif normalized_score >= HIGH_THRESHOLD:
        stability_class = "High stability"
    elif normalized_score >= MEDIUM_THRESHOLD:
        stability_class = "Medium stability"
    else:
        stability_class = "Low stability"

    return {
        "Raw score": score,
        "Normalized score": normalized_score,
        "Stability level": stability_class,
        "Factor scores": factors_scores
    }


def calculate_cyclic_stability_score(beta_h, alpha_c, ring, steric):
    """Calculate stability score for cyclic quaternary ammonium cations - using improved continuous functions"""
    factors_scores = {}  # Used to record scores for each factor
    score = 0

    # 1. β-hydrogen factor
    beta_h_count = beta_h["Total β-position hydrogens"]
    beta_score = calculate_beta_hydrogen_score(beta_h_count)
    score += beta_score
    factors_scores["β-hydrogen factor"] = beta_score

    # 2. Steric hindrance
    steric_volume = steric["Average volume"]
    bulky_groups = steric["Number of bulky groups"]
    steric_score = calculate_steric_hindrance_score(steric_volume, bulky_groups)
    score += steric_score
    factors_scores["Steric hindrance"] = steric_score

    # 3. α-carbon substitution degree
    alpha_subst = alpha_c["Average substitution degree"]
    alpha_score = calculate_alpha_carbon_score(alpha_subst)
    score += alpha_score
    factors_scores["α-carbon substitution degree"] = alpha_score

    # 4. Ring stabilization effect
    ring_score = calculate_ring_score(ring)
    score += ring_score
    factors_scores["Ring stabilization effect"] = ring_score

    # Normalized score is not needed; directly use a 100-point scale
    normalized_score = score

    # Stability rating
    if normalized_score >= SUPER_HIGH_THRESHOLD:
        stability_class = "Super-high stability"
    elif normalized_score >= HIGH_THRESHOLD:
        stability_class = "High stability"
    elif normalized_score >= MEDIUM_THRESHOLD:
        stability_class = "Medium stability"
    else:
        stability_class = "Low stability"

    return {
        "Raw score": score,
        "Normalized score": normalized_score,
        "Stability level": stability_class,
        "Factor scores": factors_scores
    }


def process_batch(data_batch):
    """Process a batch of data in parallel"""
    results = []
    for idx, name, smiles in data_batch:
        result = analyze_quaternary_ammonium_stability(smiles)
        if result and result.get("Is quaternary ammonium", False):
            result["ID"] = idx  # Add original ID
            result["Name"] = name  # Add original Name
            results.append(result)
    return results


def read_csv_data(csv_file):
    """Read data from a CSV file"""
    try:
        df = pd.read_csv(csv_file)
        # Use iloc to access by position to avoid FutureWarning
        data = [(row.iloc[0], row.iloc[1], row.iloc[2]) for _, row in df.iterrows()]
        print(f"Read {len(data)} rows of data from {csv_file}")
        return data
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {str(e)}")
        return []


def cluster_structures(results, n_clusters=50):
    """Use K-means clustering to group structures and ensure diversity"""
    if len(results) <= n_clusters:
        return results  # If the number of structures is less than or equal to the number of clusters, directly return all structures

    print(f"Clustering {len(results)} structures to ensure diversity...")

    # Generate molecular fingerprints - using the updated MorganGenerator method
    mols = [Chem.MolFromSmiles(r["SMILES"]) for r in results]
    fps = []

    for mol in mols:
        if mol:
            # Generate Morgan fingerprints using the new MorganGenerator method
            # If the rdkit version supports it, use MorganGenerator; otherwise fall back to GetMorganFingerprintAsBitVect
            try:
                from rdkit.Chem.AllChem import MorganGenerator
                fp = MorganGenerator.GetMorganFingprint(mol, 2, fpSize=1024)
            except (ImportError, AttributeError):
                # Fall back to the original method
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            fps.append(fp)

    # Convert to numpy array
    np_fps = []
    for fp in fps:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)
    X = np.array(np_fps)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    clusters = kmeans.labels_

    # Select the highest-scoring structure from each cluster
    clustered_results = []
    for i in range(n_clusters):
        cluster_indices = np.where(clusters == i)[0]
        if len(cluster_indices) > 0:
            # Get the highest-scoring structure in this cluster
            best_idx = max(cluster_indices, key=lambda idx: results[idx]["Normalized score"])
            clustered_results.append(results[best_idx])

    print(f"Clustering completed, retained {len(clustered_results)} representative structures")
    return clustered_results


def save_results_to_csv(results, output_file, top_n=None, include_factors_scores=True):
    """Save results to a CSV file"""
    # Sort by score first
    sorted_results = sorted(results, key=lambda x: x.get("Normalized score", 0), reverse=True)

    # If top_n is specified, keep only the top top_n results
    if top_n is not None:
        sorted_results = sorted_results[:top_n]

    # Prepare CSV data
    data = []
    for i, result in enumerate(sorted_results):
        row = {
            "ID": result.get("ID", ""),
            "Name": result.get("Name", ""),
            "SMILES": result["SMILES"],
            "Structure type": result.get("Structure type", ""),
            "Raw score": result["Raw score"],
            "Normalized score": result["Normalized score"],
            "Stability level": result["Stability level"],
            "Rank": i + 1
        }

        # Add β-hydrogen-related data
        if "β-hydrogen analysis" in result:
            row["Total β-position hydrogens"] = result["β-hydrogen analysis"].get("Total β-position hydrogens", "")
            row["E2 elimination risk"] = result["β-hydrogen analysis"].get("E2 elimination risk", "")

        # Add α-carbon-related data
        if "α-carbon substitution degree" in result:
            row["Average substitution degree"] = result["α-carbon substitution degree"].get("Average substitution degree", "")
            row["SN2 risk"] = result["α-carbon substitution degree"].get("SN2 risk", "")

        # Add ring analysis data (if applicable)
        if "Ring analysis" in result:
            row["Ring stabilization effect"] = result["Ring analysis"].get("Ring stabilization effect", "")
            row["Nitrogen-containing ring sizes"] = str(result["Ring analysis"].get("Nitrogen-containing ring sizes", ""))

        # Add factor scores (if needed)
        if include_factors_scores and "Factor scores" in result:
            for factor, score in result["Factor scores"].items():
                row[f"{factor} score"] = score

        data.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(data)} results to {output_file}")


def generate_summary_report(results, output_file):
    """Generate a detailed analysis report"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Quaternary Ammonium Ion Stability Analysis Report\n\n")

        # Overall statistics
        linear_results = [r for r in results if r["Structure type"] == "linear"]
        cyclic_results = [r for r in results if r["Structure type"] == "cyclic"]

        f.write("## 1. Overall Statistics\n\n")
        f.write(f"- Total number of quaternary ammonium structures: {len(results)}\n")
        f.write(f"- Number of linear structures: {len(linear_results)}\n")
        f.write(f"- Number of cyclic structures: {len(cyclic_results)}\n\n")

        # Stability level distribution
        f.write("## 2. Stability Level Distribution\n\n")
        stability_levels = ["Super-high stability", "High stability", "Medium stability", "Low stability"]
        f.write("| Stability level | Count | Percentage | Linear count | Cyclic count | Average score |\n")
        f.write("|-----------------|-------|------------|--------------|--------------|---------------|\n")

        for level in stability_levels:
            level_results = [r for r in results if r["Stability level"] == level]
            level_linear = [r for r in level_results if r["Structure type"] == "linear"]
            level_cyclic = [r for r in level_results if r["Structure type"] == "cyclic"]

            count = len(level_results)
            percent = count / len(results) * 100 if results else 0
            avg_score = sum(r["Normalized score"] for r in level_results) / count if count else 0

            f.write(
                f"| {level} | {count} | {percent:.2f}% | {len(level_linear)} | {len(level_cyclic)} | {avg_score:.2f} |\n")

        # β-hydrogen distribution
        f.write("\n## 3. β-Hydrogen Distribution\n\n")
        f.write("| Total β-position hydrogens | Count | Percentage | Average normalized score | Highest score | Lowest score |\n")
        f.write("|----------------------------|-------|------------|--------------------------|---------------|--------------|\n")

        beta_h_ranges = [(0, 0), (1, 1), (2, 2), (3, 4), (5, float('inf'))]
        for min_h, max_h in beta_h_ranges:
            if max_h == float('inf'):
                range_results = [r for r in results if r["β-hydrogen analysis"]["Total β-position hydrogens"] >= min_h]
                range_label = f"≥{min_h}"
            else:
                range_results = [r for r in results if min_h <= r["β-hydrogen analysis"]["Total β-position hydrogens"] <= max_h]
                range_label = f"{min_h}-{max_h}" if min_h != max_h else f"{min_h}"

            count = len(range_results)
            percent = count / len(results) * 100 if results else 0
            avg_score = sum(r["Normalized score"] for r in range_results) / count if count else 0
            max_score = max([r["Normalized score"] for r in range_results]) if range_results else 0
            min_score = min([r["Normalized score"] for r in range_results]) if range_results else 0

            f.write(
                f"| {range_label} | {count} | {percent:.2f}% | {avg_score:.2f} | {max_score:.2f} | {min_score:.2f} |\n")

        # α-carbon substitution degree distribution
        f.write("\n## 4. α-Carbon Substitution Degree Distribution\n\n")
        f.write("| Average substitution degree | Count | Percentage | Average normalized score |\n")
        f.write("|-----------------------------|-------|------------|--------------------------|\n")

        sub_ranges = [(0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, float('inf'))]
        for min_s, max_s in sub_ranges:
            if max_s == float('inf'):
                range_results = [r for r in results if r["α-carbon substitution degree"]["Average substitution degree"] >= min_s]
                range_label = f"≥{min_s}"
            else:
                range_results = [r for r in results if min_s <= r["α-carbon substitution degree"]["Average substitution degree"] < max_s]
                range_label = f"{min_s}-{max_s}"

            count = len(range_results)
            percent = count / len(results) * 100 if results else 0
            avg_score = sum(r["Normalized score"] for r in range_results) / count if count else 0

            f.write(f"| {range_label} | {count} | {percent:.2f}% | {avg_score:.2f} |\n")

        # Cyclic structure analysis (if there are cyclic structures)
        if cyclic_results:
            # Ring size distribution
            f.write("\n## 5. Ring Size Distribution (Cyclic Structures Only)\n\n")
            f.write("| Nitrogen-containing ring size | Count | Average normalized score |\n")
            f.write("|-------------------------------|-------|--------------------------|\n")

            # Collect all ring sizes
            all_ring_sizes = set()
            for r in cyclic_results:
                if r["Ring analysis"]["Nitrogen-containing ring sizes"]:
                    for size in r["Ring analysis"]["Nitrogen-containing ring sizes"]:
                        all_ring_sizes.add(size)

            # Sort by ring size
            for size in sorted(all_ring_sizes):
                size_results = [r for r in cyclic_results if
                                r["Ring analysis"]["Nitrogen-containing ring sizes"] and size in r["Ring analysis"]["Nitrogen-containing ring sizes"]]
                count = len(size_results)
                avg_score = sum(r["Normalized score"] for r in size_results) / count if count else 0

                f.write(f"| {size} | {count} | {avg_score:.2f} |\n")

        # Steric hindrance analysis
        f.write("\n## 6. Steric Hindrance Analysis\n\n")
        steric_levels = ["High", "Medium", "Low"]
        f.write("| Steric hindrance effect | Count | Percentage | Average score |\n")
        f.write("|-------------------------|-------|------------|---------------|\n")

        for level in steric_levels:
            level_results = [r for r in results if r["Steric hindrance"]["Steric hindrance effect"] == level]
            count = len(level_results)
            percent = count / len(results) * 100 if results else 0
            avg_score = sum(r["Normalized score"] for r in level_results) / count if count else 0

            f.write(f"| {level} | {count} | {percent:.2f}% | {avg_score:.2f} |\n")

        # Linear-chain characteristic analysis (if there are linear structures)
        if linear_results:
            f.write("\n## 7. Linear-Chain Characteristic Analysis\n\n")
            chain_levels = ["Relatively high", "Medium", "Relatively low"]
            f.write("| Chain length effect rating | Count | Percentage | Average score |\n")
            f.write("|----------------------------|-------|------------|---------------|\n")

            for level in chain_levels:
                level_results = [r for r in linear_results if "Chain characteristics" in r and r["Chain characteristics"]["Chain length effect rating"] == level]
                count = len(level_results)
                percent = count / len(linear_results) * 100 if linear_results else 0
                avg_score = sum(r["Normalized score"] for r in level_results) / count if count else 0

                f.write(f"| {level} | {count} | {percent:.2f}% | {avg_score:.2f} |\n")

        # Average factor score analysis
        f.write("\n## 8. Average Factor Score Analysis\n\n")
        f.write("| Factor | Average score | Highest score | Lowest score |\n")
        f.write("|--------|---------------|---------------|--------------|\n")

        # Collect all factors
        all_factors = set()
        for r in results:
            if "Factor scores" in r:
                for factor in r["Factor scores"].keys():
                    all_factors.add(factor)

        # Calculate statistics for each factor
        for factor in sorted(all_factors):
            factor_scores = [r["Factor scores"].get(factor, 0) for r in results if "Factor scores" in r]
            if factor_scores:
                avg_score = sum(factor_scores) / len(factor_scores)
                max_score = max(factor_scores)
                min_score = min(factor_scores)
                f.write(f"| {factor} | {avg_score:.2f} | {max_score:.2f} | {min_score:.2f} |\n")

        # Top 10 structures
        f.write("\n## 9. Top 10 Structures\n\n")
        f.write("| Rank | ID | Name | SMILES | Normalized score | Stability level |\n")
        f.write("|------|----|----|--------|------------------|-----------------|\n")

        top_results = sorted(results, key=lambda x: x["Normalized score"], reverse=True)[:10]
        for i, r in enumerate(top_results):
            smiles = r["SMILES"]
            if len(smiles) > 30:
                smiles = smiles[:27] + "..."
            f.write(
                f"| {i + 1} | {r.get('ID', 'N/A')} | {r.get('Name', 'N/A')} | {smiles} | {r['Normalized score']:.2f} | {r['Stability level']} |\n")

        # Summary
        f.write("\n## 10. Analysis Summary\n\n")

        # Find the highest-scoring structure
        if results:
            best_result = max(results, key=lambda r: r["Normalized score"])
            f.write(f"- Highest-scoring structure: ID={best_result.get('ID', 'N/A')}, Name={best_result.get('Name', 'N/A')}\n")
            f.write(f"- SMILES: {best_result['SMILES']}\n")
            f.write(f"- Score: {best_result['Normalized score']:.2f}\n")
            f.write(f"- Structure type: {best_result['Structure type']}\n")

            if "β-hydrogen analysis" in best_result:
                f.write(f"- Total β-position hydrogens: {best_result['β-hydrogen analysis']['Total β-position hydrogens']}\n")

            if "α-carbon substitution degree" in best_result:
                f.write(f"- Average α-carbon substitution degree: {best_result['α-carbon substitution degree']['Average substitution degree']:.2f}\n")

            if "Steric hindrance" in best_result:
                f.write(f"- Steric hindrance effect: {best_result['Steric hindrance']['Steric hindrance effect']}\n")

            f.write("\n### Key Findings:\n")

            # Relationship between β-hydrogen and score
            avg_scores_by_beta = {}
            for r in results:
                beta_h = r["β-hydrogen analysis"]["Total β-position hydrogens"]
                if beta_h not in avg_scores_by_beta:
                    avg_scores_by_beta[beta_h] = []
                avg_scores_by_beta[beta_h].append(r["Normalized score"])

            f.write("- Relationship between β-hydrogen and score:\n")
            for beta_h in sorted(avg_scores_by_beta.keys()):
                scores = avg_scores_by_beta[beta_h]
                avg = sum(scores) / len(scores)
                f.write(f"  * β-hydrogen={beta_h}: average score={avg:.2f} (sample count={len(scores)})\n")

            # Relationship between substitution degree and score
            f.write("\n- Relationship between α-carbon substitution degree and score:\n")
            avg_scores_by_sub = {}
            for r in results:
                sub = round(r["α-carbon substitution degree"]["Average substitution degree"])
                if sub not in avg_scores_by_sub:
                    avg_scores_by_sub[sub] = []
                avg_scores_by_sub[sub].append(r["Normalized score"])

            for sub in sorted(avg_scores_by_sub.keys()):
                scores = avg_scores_by_sub[sub]
                avg = sum(scores) / len(scores)
                f.write(f"  * Average substitution degree≈{sub}: average score={avg:.2f} (sample count={len(scores)})\n")

        f.write("\nGeneration time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    print(f"Analysis report has been saved to {output_file}")


def organize_gjf_files(results, top_n=1000, cluster_results=None, group_size=50):
    """
    Organize gjf files according to scoring results
    Args:
        results: All scoring results
        top_n: Number of top-ranked structures
        cluster_results: Clustering results
        group_size: Number of files in each group
    """
    source_folder = "gaussian_inputs_gjf"
    if not os.path.exists(source_folder):
        print(f"Error: folder {source_folder} not found, unable to organize gjf files")
        return False

    # Check all gjf files
    all_gjf_files = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.gjf'):
                all_gjf_files.append(os.path.join(root, file))

    if not all_gjf_files:
        print(f"Error: no gjf files found in folder {source_folder}")
        return False

    print(f"Found {len(all_gjf_files)} gjf files in folder {source_folder}")

    # Create output folders
    ss_folder = "QA_Stability_Score/gjf_1000_50"  # Folder for the top 1000 ranked structures
    jl_folder = "QA_Stability_Score/gjf_JL_50"  # Folder for clustered structures

    if os.path.exists(ss_folder):
        shutil.rmtree(ss_folder)
    os.makedirs(ss_folder)

    if os.path.exists(jl_folder):
        shutil.rmtree(jl_folder)
    os.makedirs(jl_folder)

    # Process the top N structures
    sorted_results = sorted(results, key=lambda x: x.get("Normalized score", 0), reverse=True)[:top_n]
    name_to_result = {r.get("Name", ""): r for r in sorted_results if "Name" in r}

    # Count the number of matched files
    matched_files = []

    for gjf_path in all_gjf_files:
        filename = os.path.basename(gjf_path)
        name = os.path.splitext(filename)[0]

        if name in name_to_result:
            matched_files.append((gjf_path, name, name_to_result[name]["Normalized score"]))

    if not matched_files:
        print(f"Warning: no gjf files matching the top {top_n} structures were found")
    else:
        print(f"Found {len(matched_files)} gjf files matching the top {top_n} structures")

        # Sort by score
        matched_files.sort(key=lambda x: x[2], reverse=True)

        # Copy files by group
        total_groups = (len(matched_files) + group_size - 1) // group_size

        for i in range(total_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, len(matched_files))

            # Create subfolder
            if i == total_groups - 1 and end_idx < top_n:
                subfolder_name = f"SS1000_{end_idx}"
            else:
                subfolder_name = f"SS1000_{(i + 1) * group_size}"

            subfolder_path = os.path.join(ss_folder, subfolder_name)
            os.makedirs(subfolder_path)

            # Copy files
            for j in range(start_idx, end_idx):
                gjf_path, name, score = matched_files[j]
                dest_path = os.path.join(subfolder_path, os.path.basename(gjf_path))
                shutil.copy2(gjf_path, dest_path)

            print(f"Created folder {subfolder_name}, containing {end_idx - start_idx} gjf files")

    # Process clustered structures
    if cluster_results:
        cluster_name_to_result = {r.get("Name", ""): r for r in cluster_results if "Name" in r}

        # Count the number of matched files
        matched_cluster_files = []

        for gjf_path in all_gjf_files:
            filename = os.path.basename(gjf_path)
            name = os.path.splitext(filename)[0]

            if name in cluster_name_to_result:
                matched_cluster_files.append((gjf_path, name, cluster_name_to_result[name]["Normalized score"]))

        if not matched_cluster_files:
            print(f"Warning: no gjf files matching clustered structures were found")
        else:
            print(f"Found {len(matched_cluster_files)} gjf files matching clustered structures")

            # Sort by score
            matched_cluster_files.sort(key=lambda x: x[2], reverse=True)

            # Copy files by group
            total_groups = (len(matched_cluster_files) + group_size - 1) // group_size

            for i in range(total_groups):
                start_idx = i * group_size
                end_idx = min((i + 1) * group_size, len(matched_cluster_files))

                # Create subfolder
                if i == total_groups - 1:
                    subfolder_name = f"JL_{end_idx}"
                else:
                    subfolder_name = f"JL_{(i + 1) * group_size}"

                subfolder_path = os.path.join(jl_folder, subfolder_name)
                os.makedirs(subfolder_path)

                # Copy files
                for j in range(start_idx, end_idx):
                    gjf_path, name, score = matched_cluster_files[j]
                    dest_path = os.path.join(subfolder_path, os.path.basename(gjf_path))
                    shutil.copy2(gjf_path, dest_path)

                print(f"Created folder {subfolder_name}, containing {end_idx - start_idx} gjf files")

    return True


def main():
    """Main program, using global parameters"""
    start_time = time.time()

    # Create output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    # Read CSV data
    data = read_csv_data(INPUT_CSV)
    if not data:
        print("No valid data was read, program exiting")
        return

    # Process in batches
    batch_size = max(1, len(data) // NUM_PROCESSES)
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    print(f"Starting parallel analysis using {NUM_PROCESSES} processes...")

    # Parallel processing
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        results_batches = list(tqdm(pool.imap(process_batch, batches), total=len(batches)))

    # Merge results
    all_results = []
    for batch in results_batches:
        all_results.extend(batch)

    print(f"Analysis completed: found {len(all_results)} valid quaternary ammonium structures")

    # Count linear and cyclic structures
    linear_results = [r for r in all_results if r["Structure type"] == "linear"]
    cyclic_results = [r for r in all_results if r["Structure type"] == "cyclic"]

    print(f"Linear structures: {len(linear_results)}")
    print(f"Cyclic structures: {len(cyclic_results)}")

    # If clustering is enabled, cluster results to ensure diversity
    clustered_results = []
    if USE_CLUSTERING and len(all_results) > CLUSTERING_COUNT:
        # Cluster linear and cyclic structures separately
        if len(linear_results) > 0 and len(cyclic_results) > 0:
            linear_count = max(1, int(CLUSTERING_COUNT * len(linear_results) / len(all_results)))
            cyclic_count = CLUSTERING_COUNT - linear_count

            clustered_linear = cluster_structures(linear_results, linear_count) if linear_results else []
            clustered_cyclic = cluster_structures(cyclic_results, cyclic_count) if cyclic_results else []

            clustered_results = clustered_linear + clustered_cyclic
        else:
            # If there is only one structure type, directly cluster all structures
            clustered_results = cluster_structures(all_results, CLUSTERING_COUNT)

        # Save clustering results
        clustered_output = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_PREFIX}_clustered.csv")
        save_results_to_csv(clustered_results, clustered_output)

        print(f"Clustering results: {len(clustered_results)} representative structures")

    # Save results, with linear and cyclic structures saved separately
    linear_output = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_PREFIX}_linear_top{TOP_N_STRUCTURES}.csv")
    cyclic_output = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_PREFIX}_cyclic_top{TOP_N_STRUCTURES}.csv")

    save_results_to_csv(linear_results, linear_output, TOP_N_STRUCTURES)
    save_results_to_csv(cyclic_results, cyclic_output, TOP_N_STRUCTURES)

    # Save all results
    all_output = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_PREFIX}_all_results.csv")
    save_results_to_csv(all_results, all_output)

    # Save the top TOP_N_STRUCTURES structures from the unified ranking of linear and cyclic structures
    combined_output = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_PREFIX}_combined_top{TOP_N_STRUCTURES}.csv")
    save_results_to_csv(all_results, combined_output, TOP_N_STRUCTURES)

    # Generate analysis report
    report_file = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_PREFIX}_analysis_report.md")
    generate_summary_report(all_results, report_file)

    # Statistics
    print("\nAnalysis statistics:")
    print(f"Total number of quaternary ammonium structures: {len(all_results)}")

    # Count each stability level
    super_high = sum(1 for r in all_results if r['Stability level'] == 'Super-high stability')
    high = sum(1 for r in all_results if r['Stability level'] == 'High stability')
    medium = sum(1 for r in all_results if r['Stability level'] == 'Medium stability')
    low = sum(1 for r in all_results if r['Stability level'] == 'Low stability')

    print(f"Super-high stability structures: {super_high}")
    print(f"High stability structures: {high}")
    print(f"Medium stability structures: {medium}")
    print(f"Low stability structures: {low}")
    print(f"Total processing time: {(time.time() - start_time) / 60:.2f} minutes")
    print(f"\nResult files have been saved to the {OUTPUT_FOLDER} folder")

    # Ask whether to organize gjf files
    user_input = input("\nDo you want to extract gjf files from the gaussian_inputs_gjf folder? (y/n): ")
    if user_input.lower() == 'y':
        print("Starting to organize gjf files...")

        # Extract gjf files for the top TOP_N_STRUCTURES structures and clustered structures
        success = organize_gjf_files(
            all_results,
            top_n=TOP_N_STRUCTURES,
            cluster_results=clustered_results,
            group_size=GJF_GROUP_SIZE
        )

        if success:
            print("gjf file organization completed!")
        else:
            print("gjf file organization failed!")
    else:
        print("Skipping gjf file organization step.")

    print("\nProgram execution completed!")


if __name__ == "__main__":
    main()
