import numpy as np
from numba import njit
import pandas as pd
import torch
from .Device_aux_functions import move_to_device, get_device
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from .class_def import k_max_class, criteria_class, Matrix

# validate inputs #
def _validate_input(input, class_type):
        if not isinstance(input, class_type):
              input = class_type(input)
        return input

### Estimating Factors ###
def _output(F, L, N, T, dim_factor, restrict):
    return F, L, {'N': N, 'T': T, 'dim_factor': dim_factor, 'restrict utilized': restrict}

def PCA(X: Matrix, 
        dim_factor: int, 
        restrict: str = 'optimize',
        Torch_cuda: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    '''
    Computes the principal components of a dataset by maximizing tr(F'X'XF) or tr(L'X'XL),  
    where F is the common factor, L is the loading factor and X is the data matrix.  
    This method follows the approach proposed by Bai and Ng (2002):  
    https://doi.org/10.1111/1468-0262.00273.  

    Parameters:
        X (2D numpy.ndarray): The data matrix of shape (T, N),  
            where T is the number of time observations and N is the number of individuals.  
        dim_factor (int): The number of principal components (k).  
        restrict (str, optional): The restriction method used for calculating factors.  
            Options are:  
            - 'common': Restricts the common factor such that F^T F / T = I.  
            - 'loading': Restricts the loading factor such that L^T L / N = I.  
            - 'optimize' (default): Automatically selects the most computationally efficient option:  
                - Uses 'common' if N ≥ T.  
                - Uses 'loading' if T > N. 
        Torch_cuda (bool, optional): Whether to use PyTorch for computations. If cuda torch was available, the computations will be done on the GPU else they use pytorch in cpu. Default is **False**.

    Returns:
        F (numpy.ndarray): The estimated common factor of shape (T, dim_factor).  
        L (numpy.ndarray): The estimated loading factor of shape (N, dim_factor). 
        info (dict): A dictionary containing the inputs utilized. 
    '''

    # validate inputs #
    X = _validate_input(X, Matrix)
    dim_factor = _validate_input(dim_factor, k_max_class)
    
    F, L, N, T, dim_factor, restrict, device, _ = _PCA(X, dim_factor, restrict, Torch_cuda)

    if Torch_cuda:
        F = F.cpu().numpy()
        L = L.cpu().numpy()

    return _output(F, L, N, T, dim_factor, restrict)

def _PCA(X: Matrix, 
        dim_factor: int, 
        restrict: str = 'optimize',
        Torch_cuda: bool = False):
    '''
    Internal function for computing the principal components of a dataset.
    For direct usage, please refer to the `PCA` function.

    Parameters:
        see `PCA` function for details.

    Returns:
        F (numpy.ndarray): The estimated common factor of shape (T, dim_factor).
        L (numpy.ndarray): The estimated loading factor of shape (N, dim_factor).
        N (int): The number of individuals (variables).
        T (int): The number of time observations.
        dim_factor (int): The number of principal components (k).
        restrict (str): The restriction method used for calculating factors.
        
    '''
    
    # Calculate the dimensions of the data matrix
    T, N = X.shape

    # Ensure XX is a PyTorch tensor
    if Torch_cuda:
        
        device = get_device()
        X = move_to_device(X, device)
    else:
        device = None

    # optimize the restriction method if not specified
    if restrict == 'optimize':
        restrict = 'common' if N >= T else 'loading'

    if restrict == 'common':
        # Calculate the common factor

        XX = X @ X.T

        F = _restrict_to_F(XX, dim_factor, Torch_cuda, device)
        # Calculate the loading factor
        L_T = F.T @ X / T
        L = L_T.T
        
    else:
        # Calculate the loading factor
        XX = X.T @ X

        N = X.shape[1]
        L = _restrict_to_L(XX, dim_factor, Torch_cuda, device)
        
        # Calculate the common factor
        F = X @ L / N
    
    return F, L, N, T, dim_factor, restrict, device, X

def _restrict_to_F(XX, dim_factor, use_torch=False, device=None):
    """
    Return the eigenvectors of the largest eigenvalues restricted to F'F/T = I.

    Parameters:
        XX (numpy.ndarray or torch.Tensor): XX^T, a T x T matrix.
        dim_factor (int): Dimension of factors (k).
        use_torch (bool): Whether to use PyTorch for computations.

    Returns:
        numpy.ndarray or torch.Tensor: Matrix F.
    """
    T = XX.shape[0]  # Get the size of the square matrix

    if use_torch:

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = torch.linalg.eigh(XX)

        F = torch.sqrt(torch.tensor(T, dtype=torch.float32, device=device)) * eigvecs[:, -dim_factor:]
        F = F.flip([1])

    else:
        # Compute eigenvalues and eigenvectors using NumPy
        eigvals, eigvecs = np.linalg.eigh(XX)

        # Select the eigenvectors of the largest `dim_factor` eigenvalues
        F = np.sqrt(T) * eigvecs[:, -dim_factor:][:, ::-1]
        
    return F

def _restrict_to_L(XX, dim_factor, use_torch=False, device=None):
    '''
    Return the eicenvectors of the largest eigenvalues restricted L'L/N = I. 

    Parameters:
        XX (numpy.ndarray or torch.Tensor): X^T X, a N x N matrix.
        dim_factor (int): Dimension of factors (k).
        use_torch (bool): Whether to use PyTorch for computations.

    Returns:
        numpy.ndarray or torch.Tensor: Matrix F.
    '''

    N = XX.shape[0]  # Get the size of the square matrix

    if use_torch:
        
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = torch.linalg.eigh(XX)

        # Select the eigenvectors of the largest `dim_factor` eigenvalues
        L = torch.sqrt(torch.tensor(N, dtype=torch.float32, device=device)) * eigvecs[:, -dim_factor:]
        L = L.flip([1])

    else:
        # Compute eigenvalues and eigenvectors using NumPy
        eigvals, eigvecs = np.linalg.eigh(XX)

        L = np.sqrt(N) * eigvecs[:, -dim_factor:][:, ::-1]

    return L

### Estimating dimension of factor structure ###
# Define PC and IC criteria functions
def _criteria_function_PC(k, V, sigma2_hat, penalty):
    return V[k - 1] + k * sigma2_hat * penalty

def _criteria_function_IC(k, V, sigma2_hat, penalty):
    return np.log(V[k - 1]) + k * penalty

@njit(fastmath=True, cache=True)
def _SSE_numpa(k, X, F_hat, L_hat):
    F_k = F_hat[:, :k]
    L_k = L_hat[:, :k]
    FL = np.dot(F_k, L_k.T)  # Numba-accelerated matrix multiplication

    # Compute squared error efficiently
    T, N = X.shape
    error_sum = 0.0
    for i in range(T):  # Standard loop (No parallelism)
        for j in range(N):
            diff = X[i, j] - FL[i, j]
            error_sum += diff * diff

    return error_sum / (T * N)  # Mean squared error

def _SSE(k, X, F_hat, L_hat):
    F_k = F_hat[:, :k]
    L_k = L_hat[:, :k]
    FL = F_hat[:, :k] @ L_hat[:, :k].T
    return np.mean((X - FL) ** 2)

def _SSE_Cuda(k, X, F_hat, L_hat):
    F_k = F_hat[:, :k]
    L_k = L_hat[:, :k]
    FL = torch.matmul(F_k, L_k.T)
    return torch.mean((X - FL) ** 2)

def _FDE(X: Matrix,
        k_max: k_max_class = 8,
        Criteria: criteria_class = ['PC1', 'PC2', 'PC3', 'IC1', 'IC2', 'IC3'],
        restrict: str = 'optimize',
        Numpa_opt: bool = False,
        Torch_cuda: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    
    '''
    Internal function for computing the **number of factors (k)** in large-dimensional factor models.  
    This function is intended for use inside other functions.  
    For direct usage, please refer to the `FDE` function.  

    Parameters:
        See `FDE` function for details.  

    Returns:
        k (int): The estimated factor dimensionality.  
        F (numpy.ndarray): The estimated common factor of shape (T, k).  
        L (numpy.ndarray): The estimated loading factor of shape (N, k).
        k (int): The estimated factor dimensionality.
    '''

    # Calculate the factor using the PCA method with k_max
    F_hat, L_hat, _, _, _, _, device, X = _PCA(X, k_max, restrict=restrict, Torch_cuda=Torch_cuda)

    # Calculate the error
    e_kmax = X - F_hat @ L_hat.T
    sigma2_hat = np.mean(e_kmax**2) if not Torch_cuda else torch.mean(e_kmax**2).cpu().numpy()
    T, N = X.shape

    # Calculate V(k, F^k)
    if Torch_cuda:
        V = torch.tensor([_SSE_Cuda(k, X, F_hat, L_hat) for k in range(1, k_max + 1)], device=device)
        V = V.cpu().numpy()
    elif Numpa_opt:
        V = [_SSE_numpa(k, X, F_hat, L_hat) for k in range(1, k_max + 1)]
    else:
        V = [_SSE(k, X, F_hat, L_hat) for k in range(1, k_max + 1)]
    
    # Calculate the penalties for each criteria
    C_NT_square = min(T, N)
    A = (N + T) / (N * T)  # Corrected division

    Penalty_term1 = A * np.log(1/A)
    Penalty_term2 = A * np.log(C_NT_square)
    Penalty_term3 = np.log(C_NT_square) / C_NT_square

    penalties = [Penalty_term1, Penalty_term2, Penalty_term3]

    # Calculate the criteria matrix
    criteria_matrix = np.zeros((len(Criteria), k_max))
    for i, c in enumerate(Criteria):
        fun, penalty_idx = c[:2], int(c[2]) - 1
        penalty = penalties[penalty_idx]
        criteria_matrix[i] = [globals()['_criteria_function_' + fun](k, V, sigma2_hat, penalty) for k in range(1, k_max + 1)]
    
    # Determine the factor dimensionality
    k = np.argmax(-criteria_matrix[0, :]) + 1

    return criteria_matrix, F_hat[:, :k], L_hat[:, :k], k

class FDE_output:
    def __init__(self, criteria_matrix, Criteria, k_max, F_hat, L_hat, k, X, residuals):

        k_list = [f"K={k}" for k in range(1, k_max + 1)]
        criteria_pd = pd.DataFrame(criteria_matrix, columns = k_list, index = Criteria)
        self.Data = X
        self.criteria_matrix = criteria_pd
        self.Criterias = Criteria
        self.k_max = k_max
        self.F_hat = F_hat
        self.L_hat = L_hat
        self.k = k
        self.residuals = residuals
    
    def summary(self):

        k_opt = [np.argmin(self.criteria_matrix.iloc[i, :]) + 1 for i in range(len(self.criteria_matrix.index))]
        k_opt = pd.DataFrame([k_opt], columns=self.criteria_matrix.index, index=['K estimated'])

        print("\n")
        print("** Factor Dimensionality Estimation Summary **")
        print("\n")
        print("Criteria K estimated:")
        print(k_opt)
        print("\n")
        print(f"Selected Criteria: {self.Criterias[0]}")
        print(f"Selected Factor Dimensionality: {self.k}")
        print("Residuals: Mean = ", np.mean(self.residuals).round(4), ", Std = ", np.std(self.residuals).round(4), ". R^2: ", 1 - np.var(self.residuals) / np.var(self.Data), sep = '')
        print("\n")
        print("Criteria Matrix with max factor dimensionality (K_max) equal to ", self.k_max, ":")
        print(self.criteria_matrix)
        print("\n")
        print("Algorithm based on Bai and Ng (2002): https://doi.org/10.1111/1468-0262.00273")
        return None

def factor_dimensionality(X: Matrix,
        k_max: k_max_class = 8,
        Criteria: criteria_class = ['PC1', 'PC2', 'PC3', 'IC1', 'IC2', 'IC3'],
        restrict: str = 'optimize',
        Numpa_opt: bool = False,
        Torch_cuda: bool = False) -> FDE_output:
    '''
    Estimates the **number of factors (k)** in large-dimensional factor models, along with the **common factors (F)** and **factor loadings (L)** for a given matrix T x N (**X**). The estimation is based on one of the criteria proposed by Bai and Ng (2002): https://doi.org/10.1111/1468-0262.00273.

    Parameters:
        X (numpy.ndarray): The data matrix of shape (T, N),  
            where **T** is the number of time observations,  
            and **N** is the number of individuals (variables).  
        k_max (int, optional): The maximum number of principal components to consider.  
            Must be less than **min(T, N)**. Default is **8**. 
        Criteria (list, optional): The criteria used to estimate the factor dimensionality. They printed all criterias in the list but use only the first criteria in the list to determine the factor dimensionality.
            Options are:  
                - 'IC1': Information Criterion such that
                        IC1 = ln(V(k, F)) + k * S * ln(1/S), where S = (N+T)/NT.
                - 'IC2': Information Criterion with the form
                        IC2 = ln(V(k, F)) + k * S * ln(C^2), where S = (N+T)/NT and C = min{N^(1/2), T^(1/2)}.
                - 'IC3': Information Criterion such that
                        IC3 = ln(V(k, F)) + k * ln(C^2)/C^2, whereC = min{N^(1/2), T^(1/2)}.
                - 'PC1': Information Criterion such that
                        PC1 = V(k, F) + k * sigma^2 * S * ln(1/S), where S = (N+T)/NT.
                - 'PC2': Information Criterion such that
                        PC2 = V(k, F) + k * sigma^2 * S * ln(C^2), where S = (N+T)/NT and C = min{N^(1/2), T^(1/2)}.
                - 'PC3': Information Criterion such that
                        PC2 = V(k, F) + k * sigma^2 * ln(C^2)/C^2, where C = min{N^(1/2), T^(1/2)}.  
            Default is **['PC1', 'PC2', 'PC3', 'IC1', 'IC2', 'IC3']**. 
        restrict (str, optional): The restriction method for calculating factors.  
            Options are:  
            - 'common': Restricts the common factor such that FᵀF / T = I.  
            - 'loading': Restricts the loading factor such that LᵀL / N = I.  
            - 'optimize' (default): Automatically selects the most computationally efficient option:  
                - Uses 'common' if N ≥ T.  
                - Uses 'loading' if T > N.  
        Numpa_opt (bool, optional):  Whether to use Numba-accelerated matrix operations. If `factor_dimensionality` is called multiple times for matrices of the same dimension, enabling Numba can improve performance. However, the first call will be slower due to the compilation overhead. For one-time use per matrix dimension, it is recommended to set `Numpa_opt = False`. Default is **False**.
    
    Returns:
        FDE_output: A class containing the following attributes:  
            - **'Data'**: The original data matrix.  
            - **'criteria_matrix'**: A DataFrame containing the criteria values for each factor dimensionality.  
            - **'Criterias'**: The criterias calculated.  
            - **'k_max'**: The maximum number of factor dimensionality considered.  
            - **'F_hat'**: The estimated common factor of shape (T, k).  
            - **'L_hat'**: The estimated loading factor of shape (N, k).  
            - **'k'**: The estimated factor dimensionality.  
            - **'residuals'**: The residuals of the estimated factors.

    Example:

    ```python
    # Import the FDE function
    import factorAnalysis as fa
    
    # Generate a random dataset (OPTIONAL)
    import numpy as np
    T, N = 200, 200
    k = 1
    F = np.random.normal(1, 1, (T, k))
    L = np.random.normal(-2, 1, (N, k))
    X = F @ L.T + np.random.normal(0, 1, (T, N))

    # Estimate the factor dimensionality
    FDE_res = fa.factor_dimensionality(X)

    # summary of the results
    FDE_res.summary()

    # Access the estimated common factor
    F = FDE_res.F_hat

    # Access the estimated loading factor
    L = FDE_res.L_hat

    ```

    '''
    
    # Validate inputs #
    k_max = _validate_input(k_max, k_max_class)
    Criteria = _validate_input(Criteria, criteria_class)
    X = _validate_input(X, Matrix)

    criteria_matrix, F_hat, L_hat, k = _FDE(X, k_max, Criteria, restrict, Numpa_opt, Torch_cuda)

    if Torch_cuda:
        F_hat = F_hat.cpu().numpy()
        L_hat = L_hat.cpu().numpy()

    return FDE_output(criteria_matrix, Criteria, k_max, F_hat, L_hat, k, X, X - F_hat @ L_hat.T)