import numpy as np
from Bio import AlignIO
from collections import defaultdict
from multiprocessing import Pool


def preprocess_msa_chunk(params):
    chunk, num_sequences, gap_cutoff = params
    filtered_chunk = [col for col in chunk.T if np.count_nonzero(col == '-') / num_sequences < gap_cutoff]
    return np.array(filtered_chunk).T



def preprocess_msa(msa_array, gap_cutoff=0.5, n_jobs=30):
    print("Preprocessing MSA...")
    num_sequences = msa_array.shape[0]
    num_columns = msa_array.shape[1]
    # Ensure n_jobs does not exceed the number of columns
    n_jobs = min(n_jobs, num_columns)
    chunk_size = (num_columns + n_jobs - 1) // n_jobs  # Calculate chunk size, ensuring it's evenly distributed
    # Split msa_array into chunks
    chunks = [msa_array[:, i*chunk_size:min((i+1)*chunk_size, num_columns)] for i in range(n_jobs)]
    params = [(chunk, num_sequences, gap_cutoff) for chunk in chunks]
    with Pool(n_jobs) as pool:
        results = pool.map(preprocess_msa_chunk, params)
    # Concatenate the filtered chunks
    filtered_msa_array = np.concatenate(results, axis=1)
    # Convert gaps to a special character/state, e.g., -1
    filtered_msa_array[filtered_msa_array == '-'] = -1
    print("Finished preprocessing MSA.")
    return filtered_msa_array


def map_residues_to_integers(msa_array):
    print("Mapping residues to integers...")
    residues = set(np.unique(msa_array))
    residue_map = defaultdict(lambda: len(residue_map))
    for residue in residues:
        residue_map[residue]
    encoded_msa = np.vectorize(residue_map.get)(msa_array)
    num_states = len(residue_map)
    return encoded_msa, num_states


def compute_frequencies_inner(params, pseudocount=0.5, q=21):
    i, a, msa, num_sequences, num_positions, num_states, sequence_weights = params
    fi = np.zeros((num_positions, num_states))
    fij = np.zeros((num_positions, num_positions, num_states, num_states))

    # Compute effective number of sequences
    Meff = np.sum(sequence_weights)
    
    print(f"Frequency {i} / {num_positions} and state {a} / {num_states}")
    
    # Single-site frequencies with pseudocounts
    fi[i, a] = (pseudocount / q + np.sum(sequence_weights * (msa[:, i] == a))) / (Meff + pseudocount)
    
    for j in range(num_positions):
        for b in range(num_states):
            # Pairwise frequencies with pseudocounts
            fij[i, j, a, b] = (pseudocount / q**2 + np.sum(sequence_weights * (msa[:, i] == a) * (msa[:, j] == b))) / (Meff + pseudocount)
    
    return fi, fij


def compute_sequence_weights_inner(args):
    i, msa, threshold, num_sequences = args
    print (f"Sequence weight {i} / {num_sequences}")
    weights = np.zeros(num_sequences)
    for j in range(i + 1, num_sequences):
        if np.mean(msa[i] == msa[j]) > threshold:
            weights[i] += 1
            weights[j] += 1
    return weights

def compute_sequence_weights(msa, threshold=0.8):
    num_sequences = msa.shape[0]
    sequence_weights = np.ones(num_sequences)
    print("computing sequence weights")
    
    args = [(i, msa, threshold, num_sequences) for i in range(num_sequences)]
    
    with Pool() as pool:
        results = pool.map(compute_sequence_weights_inner, args)
    
    for weights in results:
        sequence_weights += weights
    
    return 1.0 / sequence_weights

# def compute_sequence_weights(msa, threshold=0.8):
#     num_sequences = msa.shape[0]
#     sequence_weights = np.ones(num_sequences)
#     print("computing sequence weights...")
#     for i in range(num_sequences):
#         for j in range(i + 1, num_sequences):
#             if np.mean(msa[i] == msa[j]) > threshold:
#                 sequence_weights[i] += 1
#                 sequence_weights[j] += 1
    
#     return 1.0 / sequence_weights

def compute_connected_correlation_matrix_inner(i, j, fi, fij, num_states):
    C_ij = np.zeros((num_states, num_states))
    print(f"Corr Matrix {i} / {num_positions} and {j} / {num_positions}")
    if i != j:
        for a in range(num_states):
            for b in range(num_states):
                C_ij[a, b] = fij[i, j, a, b] - fi[i, a] * fi[j, b]
    return i, j, C_ij

def compute_connected_correlation_matrix(fi, fij):
    num_positions, num_states = fi.shape
    C = np.zeros_like(fij)
    
    args = [(i, j, fi, fij, num_states) for i in range(num_positions) for j in range(num_positions)]
    
    with Pool() as pool:
        results = pool.starmap(compute_connected_correlation_matrix_inner, args)
    
    for i, j, C_ij in results:
        C[i, j] = C_ij
    
    return C


def infer_couplings_inner(i, j, C, num_states, lambda_reg):
    if i == j:
        return i, j, np.zeros((num_states, num_states))
    C_ij = C[i, j] + lambda_reg * np.eye(num_states)
    wij_ij = -np.linalg.inv(C_ij)
    return i, j, wij_ij

def infer_couplings(C, lambda_reg=0.01):
    num_positions, _, num_states, _ = C.shape
    wij = np.zeros_like(C)
    
    args = [(i, j, C, num_states, lambda_reg) for i in range(num_positions) for j in range(num_positions) if i != j]
    
    with Pool() as pool:
        results = pool.starmap(infer_couplings_inner, args)
    
    for i, j, wij_ij in results:
        wij[i, j] = wij_ij
    
    return wij

def compute_P_dir(eij, fi, fj):
    num_states = len(fi)
    P_dir = np.zeros((num_states, num_states))
    Z_ij = 0
    
    # Initialize auxiliary fields to zero
    h_tilde_i = np.zeros(num_states)
    h_tilde_j = np.zeros(num_states)
    
    for _ in range(10):  # Iterate to ensure convergence (10 iterations)
        for A in range(num_states):
            for B in range(num_states):
                P_dir[A, B] = np.exp(eij[A, B] + h_tilde_i[A] + h_tilde_j[B])
        
        Z_ij = np.sum(P_dir)
        P_dir /= Z_ij
        
        # Update auxiliary fields
        for A in range(num_states):
            h_tilde_i[A] = np.log(fi[A]) - np.log(np.sum(P_dir[A, :]))
            h_tilde_j[A] = np.log(fj[A]) - np.log(np.sum(P_dir[:, A]))
    
    return P_dir


def compute_direct_information_inner(params):
    i, j, wij, fi, num_states = params
    di_ij = 0
    print(f"DI {i} and {j}...")
    if i != j:
        P_dir = compute_P_dir(wij[i, j], fi[i], fi[j])
        for A in range(num_states):
            for B in range(num_states):
                if P_dir[A, B] > 0:
                    di_ij += P_dir[A, B] * np.log(P_dir[A, B] / (fi[i, A] * fi[j, B]))
    
    return i, j, di_ij


def compute_direct_information(wij, fi):
    print("Computing direct information...")
    num_positions, _, num_states, _ = wij.shape
    di = np.zeros((num_positions, num_positions))
    params = [(i, j, wij, fi, num_states) for i in range(num_positions) for j in range(num_positions)]
    
    with Pool() as p:
        results = p.map(compute_direct_information_inner, params)
        for result in results:
            i, j, di_ij = result
            di[i, j] = di_ij
    
    print("Finished computing direct information.")
    return di


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
sequence_weights = compute_sequence_weights(processed_msa)

# Prepare parameters for frequency computation
print("Computing frequencies...")
num_sequences, num_positions = processed_msa.shape

# params = [(i, j, wij, fi, fi, num_states) for i in range(num_positions) for j in range(num_positions)]
params = [(i, a, encoded_msa, num_sequences, num_positions, num_states, sequence_weights) for i in range(num_positions) for a in range(num_states)]

# Compute frequencies
with Pool() as p:
    results = p.map(compute_frequencies_inner, params)
    
fi = np.zeros((num_positions, num_states))
fij = np.zeros((num_positions, num_positions, num_states, num_states))
    
for result in results:
    fi_result, fij_result = result
    fi += fi_result
    fij += fij_result
    
# Compute connected-correlation matrix
C = compute_connected_correlation_matrix(fi, fij)

# Infer couplings
wij = infer_couplings(C)

# Compute direct information
di = compute_direct_information(wij, fi)

# print(di)

# save numpy file (maybe npz)

np.save(output_npy, di)
# np.save("di_t.npy", di)