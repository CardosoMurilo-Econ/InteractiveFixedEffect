import numpy as np
from scipy.sparse.linalg import eigsh

from .class_def import k_max_class, criteria_class, Matrix

##########################
### Estimating Factors ###
###         PCA        ###
##########################

def _PCA(X: Matrix, 
        dim_factor: int, 
        restrict: str = 'optimize'):
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
        X (Matrix): The data matrix.
        
    '''
    
    # Calculate the dimensions of the data matrix
    T, N = X.shape

    # optimize the restriction method if not specified
    if restrict == 'optimize':
        restrict = 'common' if N >= T else 'loading'

    if restrict == 'common':
        F, L = _restrict_to_F(X, dim_factor)
         
    else:
        F, L = _restrict_to_L(X, dim_factor)
        
    return F, L, N, T, dim_factor, restrict, X

def _restrict_to_F(X, dim_factor):
    """
    Return the eigenvectors of the largest eigenvalues restricted to F'F/T = I.

    Parameters:
        X (numpy.ndarray or torch.Tensor): X, a T x N matrix.
        dim_factor (int): Dimension of factors (k).
        use_torch (bool): Whether to use PyTorch for computations.

    Returns:
        numpy.ndarray or torch.Tensor: Matrix F.
    """

    XX = X @ X.T
    T = XX.shape[0]  # Get the size of the square matrix

    # Compute eigenvalues and eigenvectors using NumPy
    eigvals, eigvecs = eigsh(XX, k=dim_factor, which='LM')
    
    # Select the eigenvectors of the largest `dim_factor` eigenvalues
    F = np.sqrt(T) * eigvecs[:, ::-1]

    L_T = F.T @ X / T
    L = L_T.T
        
    return F, L

def _restrict_to_L(X, dim_factor):
    '''
    Return the eicenvectors of the largest eigenvalues restricted L'L/N = I. 

    Parameters:
        XX (numpy.ndarray or torch.Tensor): X^T X, a N x N matrix.
        dim_factor (int): Dimension of factors (k).
        use_torch (bool): Whether to use PyTorch for computations.

    Returns:
        numpy.ndarray or torch.Tensor: Matrix F.
    '''
    XX = X.T @ X
    N = XX.shape[0]  # Get the size of the square matrix

    # Compute eigenvalues and eigenvectors using NumPy
    eigvals, eigvecs = eigsh(XX, k=dim_factor, which='LM')

    L = np.sqrt(N) * eigvecs[:, ::-1]

    F = X @ L / N

    return F, L

##########################
### Estimating Factors ###
###   Dimensionality   ###
##########################

def _criteria_function_PC(k, V, sigma2_hat, penalty):
    return V[k - 1] + k * sigma2_hat * penalty

def _criteria_function_IC(k, V, sigma2_hat, penalty):
    return np.log(V[k - 1]) + k * penalty

# def _SSE_old(k, X, F_hat, L_hat):
#     FL = F_hat[:, :k] @ L_hat[:, :k].T
#     return np.nanmean((X - FL) ** 2)

def _SSE(k, X, F_hat, L_hat):
    FL = F_hat[:, :k] @ L_hat[:, :k].T
    E = X - FL
    mean = np.einsum('ij, ij -> ', E, E) / (E.shape[0] * E.shape[1])
    return mean

def _sigma2_hat(X, F_hat, L_hat):
    e_kmax = X - F_hat @ L_hat.T
    return np.nanmean(e_kmax**2) 

CRITERIA_DISPATCH = {
    'IC': _criteria_function_IC,  
    'PC': _criteria_function_PC,  
}

def _choose_k(
            F_hat,
            L_hat,
            X: Matrix,
            sigma2_hat: float,
            k_max,
            Criteria):
    
    #T, N = X.shape
    # adjust to missing values:
    T, N = X.shape

    # Calculate V(k, F^k)

    #V = [_SSE_old(k, X, F_hat, L_hat) for k in range(1, k_max + 1)]

    V = [_SSE(k, X, F_hat, L_hat) for k in range(1, k_max + 1)]

    # Calculate the penalties for each criteria
    C_NT_square = min(T, N)
    A = (N + T) / (N * T) 

    Penalty_term1 = A * np.log(1/A)
    Penalty_term2 = A * np.log(C_NT_square)
    Penalty_term3 = np.log(C_NT_square) / C_NT_square

    penalties = [Penalty_term1, Penalty_term2, Penalty_term3]

    # Calculate the criteria matrix
    criteria_matrix = np.zeros((len(Criteria), k_max))
    for i, c in enumerate(Criteria):
        fun, penalty_idx = c[:2], int(c[2]) - 1
        penalty = penalties[penalty_idx]
        criteria_function = CRITERIA_DISPATCH[fun]

        for k in range(1, k_max + 1):
            criteria_matrix[i, k-1] = criteria_function(k, V, sigma2_hat, penalty)


    # Determine the factor dimensionality
    k_all = np.argmin(criteria_matrix, axis=1) + 1
    k = k_all[0]

    return k, k_all, criteria_matrix

def _FDE(X: Matrix,
        k_max: k_max_class = 10,
        Criteria: criteria_class = ['PC1', 'PC2', 'PC3', 'IC1', 'IC2', 'IC3'],
        restrict: str = 'optimize') -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    
    '''
    Internal function for computing the **number of factors (k)** in large-dimensional factor models.  
    This function is intended for use inside other functions.  
    For direct usage, please refer to the `factor_dimensionality` function.  

    Parameters:
        See `factor_dimensionality` function for details.  

    Returns:
        Criteria trix (numpy.matrix): A matrix containing the criteria values for each factor dimensionality.  
        F (numpy.ndarray): The estimated common factor of shape (T, k).  
        L (numpy.ndarray): The estimated loading factor of shape (N, k).
        k (int): The estimated factor dimensionality.
        k_all (numpy.ndarray): An array containing the optimal k for each criteria.
    '''

    # Calculate the factor using the PCA method with k_max
    F_hat, L_hat, _, _, _, _, X = _PCA(X, k_max, restrict=restrict)

    # Calculate the residuals
    sigma2_hat = _sigma2_hat(X, F_hat, L_hat)

    # Choose the optimal k based on the criteria
    
    k, k_all, criteria_matrix = _choose_k(F_hat, L_hat, X, sigma2_hat, k_max, Criteria)

    return criteria_matrix, F_hat[:, :k], L_hat[:, :k], k, k_all

