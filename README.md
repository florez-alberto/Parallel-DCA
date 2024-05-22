# Parallel-DCA

Parallel-DCA is a python script designed to perform Direct Coupling Analysis (DCA) on multiple sequence alignments (MSAs) efficiently using parallel processing. This tool leverages the `multiprocessing` module to speed up computations, making it suitable for large MSAs. This is a direct implementation of the original paper[^1].


## Theoretical background

### Preprocessing MSA

The preprocessing step handles gaps and filters columns based on a gap cutoff, as follows:

$$
\text{filteredMSA} = 
  \text{column} \in \text{MSA} \mid \frac{\text{count}(\text{gaps in column})}{\text{numSequences}} < \text{gapCutoff}
$$


Where:
- $\text{MSA}$ is the multiple sequence alignment.
- $\text{numSequences}$ is the number of sequences in the MSA.
- $\text{gapCutoff}$ is the threshold for filtering columns based on the proportion of gaps.


### DCA calculation

Once the MSA data has been preprocessed, it follows the steps:

1. **Estimates the frequency counts** $f_i(A)$ and $f_{ij}(A, B)$ from the MSA, using the pseudocount $\lambda = M_{eff}$  in Eqs. 1 and 2.

    $$f_i(A) = \frac{1}{M_{eff}+\lambda} \left( \frac{\lambda}{q}+ \sum_{a=1}^{M} \frac{1}{m^a} \delta(A, A_i^a) \tag{1} \right)$$

    $$f_{ij}(A, B) = \frac{1}{M_{eff}+\lambda}  \left(\frac{\lambda}{q^2} +\sum_{a=1}^{M} \frac{1}{m^a}  \delta(A, A_i^a) \delta(B, A_j^a) \tag{2} \right)$$

2. **Determines the empirical estimate of the connected-correlation matrix** $ C_{ij}(A, B) $ using Eq. 3.

    $$
    C_{ij}(A, B) = f_{ij}(A, B) - f_i(A) f_j(B) \tag{3}
    $$

3. **Determines the couplings** $ e_{ij}(A, B) $ according to the second part of Eq. 4.
    $$
    e_{ij}(A, B) =  -(C^{-1})_{ij}(A, B) \tag{4}
    $$

4. **For each column pair** $ i < j $, estimates the direct information $ DI_{ij} $ by solving Eqs. 5 and 6 for $ P_{ij}^{(dir)}(A, B) $, and plugs the results into Eq. 7.

    $$
    P_{ij}^{(dir)}(A, B) = \frac{\exp(e_{ij}(A, B) + h_i(A) + h_j(B))}{Z_{ij}} \tag{5}
    $$

    $$
    f_{i}(A) = \sum_{B}P_{ij}^{(dir)}(A, B), \quad 
    f_{j}(B) = \sum_{A}P_{ij}^{(dir)}(A, B) \tag{6} 
    $$

    $$
    DI_{ij} = \sum_{A, B} P_{ij}^{(dir)}(A, B) \ln \left( \frac{P_{ij}^{(dir)}(A, B)}{f_i(A) f_j(B)} \right) \tag{7}
    $$

## Features

- **Preprocess MSA**: Handles gaps and filters columns based on a gap cutoff.
- **Map Residues to Integers**: Converts amino acid residues to integer representations.
- **Sequence Weights**: Adjusts for sequence redundancy.
- **Frequencies**: Calculates single-site and pairwise frequencies.
- **Connected Correlation Matrix**
- **Infers Couplings**: using regularized inverse of the correlation matrix.
- **Direct Information**: between residue pairs.

## Dependencies

To install the dependencies, use the following command:

```bash
pip install numpy biopython matplotlib
```

## Usage


1. **Run the analysis**:
    Place your MSA file in the repository directory and modify the input file name in the script if necessary.

    ```python
    python parallel_dca.py
    ```
2. **Checkpoints implementation**: In case of an unreliable environment or long compute times, a script called `parallel_dca_checkpoints.py` was implemented. It contains regular checkpoints and saves the numpy arrays on every step. It can also handle problematic or computationally intensive frequency calculations, which are calculated first.

    ```python
    python parallel_dca_checkpoints.py
    ```

## References
[^1]: Morcos, F., et al. (2011). Direct-coupling analysis of residue coevolution captures native contacts across many protein families. *Proceedings of the National Academy of Sciences*, 108(49), E1293-E1301.

