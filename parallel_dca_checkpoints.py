import numpy as np
from Bio import AlignIO
from collections import defaultdict
from multiprocessing import Pool
import os

import preprocess_msa, map_residues_to_integers, compute_sequence_weights, compute_frequencies_inner, compute_connected_correlation_matrix, infer_couplings, compute_direct_information from parallel_dca_utils

# Preprocess the MSA

input_file = "PA_PB1_DCA.fasta"
putput_npy= "di_f.npy"

# input_file ="trimmed_test.fasta"
# output_npy = "di_t.npy"

# alignment = AlignIO.read("trimmed_test.fasta", "fasta")
alignment = AlignIO.read(input_file, "fasta")

msa_array = np.array([list(record.seq) for record in alignment])

processed_msa = preprocess_msa(msa_array)

# Map residues to integers
print("Mapping residues to integers...")
with Pool() as pool:
    encoded_msa, num_states = pool.apply(map_residues_to_integers, (processed_msa,))


# Compute sequence weights
print("Computing sequence weights...")
# if weights file exists
if os.path.exists("weights.npy"):
    print("Weights file exists. Loading...")
    sequence_weights = np.load("weights.npy")
else:
    print("Weights file does not exist. Computing...")
    sequence_weights = compute_sequence_weights(processed_msa)
    np.save("weights.npy", sequence_weights)

# Prepare parameters for frequency computation
print("Computing frequencies...")
num_sequences, num_positions = processed_msa.shape

# This is an empiric list 


if os.path.exists("fi.npy") and os.path.exists("fij.npy"):
    fi = np.load("fi.npy")
    fij = np.load("fij.npy")
else:
    # Create a set with the parameters for the problematic tasks
    problematic_params_set = {(38, 2), (67, 13), (1021, 21)}
    
    problematic_params = []
    other_params = []

    if problematic_params_set:
        for i, a in problematic_params_set:
            problematic_params.append((i, a, encoded_msa, num_sequences, num_positions, num_states, sequence_weights))

        # Create a list with the parameters for all other tasks
        other_params = [(i, a, encoded_msa, num_sequences, num_positions, num_states, sequence_weights) 
                    for i in range(num_positions) for a in range(num_states) if not (i, a) in problematic_params_set]

        params = problematic_params + other_params
    else:
        params = [(i, a, encoded_msa, num_sequences, num_positions, num_states, sequence_weights) for i in range(num_positions) for a in range(num_states)]

    # Compute frequencies
    with Pool(processes=85) as p:
        results = p.map(compute_frequencies_inner, params)
    fi = np.zeros((num_positions, num_states))
    fij = np.zeros((num_positions, num_positions, num_states, num_states))
    for result in results:
        fi_result, fij_result = result
        fi += fi_result
        fij += fij_result
    np.save("fi.npy", fi)
    np.save("fij.npy", fij)
        
# Compute connected-correlation matrix
#see if C file exists
if os.path.exists("C.npy"):
    C = np.load("C.npy")
else:
    C = compute_connected_correlation_matrix(fi, fij)
    np.save("C.npy", C)

# Infer couplings
#see if wij file exists
if os.path.exists("wij.npy"):
    wij = np.load("wij.npy")
else:
    wij = infer_couplings(C)
    np.save("wij.npy", wij)

# Compute direct information
di = compute_direct_information(wij, fi)

# print(di)

# save numpy file (maybe npz)

np.save(output_npy, di)
# np.save("di_t.npy", di)