import numpy as np
import pandas as pd
import torch
from .Device_aux_functions import move_to_device, get_device, move_to_cpu
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from .class_def import k_class, k_max_class, criteria_class, Matrix
import warnings

# validate inputs #
def _validate_input(input, class_type, **kwargs):
    if not isinstance(input, class_type):
            input = class_type(input, **kwargs)
    return input

def _convergence_criteria(beta, delta, previous_delta, small_change_counter, Tol, Tol_2deriv, convergence_threshold = 1):
        
        # Check if delta is less than tolerance
        beta_norm = np.linalg.norm(beta, axis = 0) + 1e-6 if not isinstance(beta, torch.Tensor) else torch.linalg.vector_norm(beta, axis = 0) + 1e-6
        delta_standard = delta / beta_norm
        
        # Check if delta is less than tolerance 
        if all(delta_standard <= Tol):
                small_change_counter += 1
        else:
                small_change_counter = 0  # Reset if delta increases

        # Check if delta is no longer decreasing significantly
        delta_dev = abs(previous_delta - delta)/beta_norm
        if all(delta_dev < Tol_2deriv):
                small_change_counter += 1  # Treat as a small change

        # Final check: Stop only if multiple conditions are met
        if small_change_counter >= convergence_threshold:
                return True, small_change_counter, delta_standard, delta_dev
        
        return False, small_change_counter, delta_standard, delta_dev

##########################
### Estimating Factors ###
###         PCA        ###
##########################

def _output(F, L, N, T, dim_factor, restrict, N_int = None):
    return F, L, {'N': N, 'T': T, 'dim_factor': dim_factor, 'restrict utilized': restrict, 'N_int': N_int}

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
    T, N = X.shape
    dim_factor = _validate_input(dim_factor, k_max_class, N=N, T=T)
    
    missing_values = np.isnan(X).sum()
    if missing_values > 0:
        Warning(f"Warning: Missing values in the data matrix. The algorithm to estimate the factors with missing values was used.")
        F, L, N, T, dim_factor, restrict, device, N_int = PCA_MISSING(X, dim_factor, restrict, Torch_cuda=Torch_cuda)

    F, L, N, T, dim_factor, restrict, device, _ = _PCA(X, dim_factor, restrict, Torch_cuda)
    N_int = None

    if Torch_cuda:
        F = F.cpu().numpy()
        L = L.cpu().numpy()

    return _output(F, L, N, T, dim_factor, restrict, N_int)

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

##### missing Algorithm #####

def _initial_points(init, dim_factor, T, N, Torch_cuda):
    '''
    Internal function for initializing the factors and tracking variables.
    For direct usage, please refer to the `PCA` function.
    '''
    F_old, L_old = np.ones((T, dim_factor))*init, np.ones((N, dim_factor)*init)
    FL_old = F_old @ L_old.T
    delta_standard_F = np.zeros((dim_factor,))
    delta_standard_L = np.zeros((dim_factor,))
    small_change_counter_L = 0
    small_change_counter_F = 0
    k_old = dim_factor

    if Torch_cuda:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        F_old = torch.tensor(F_old, dtype=torch.float32, device=device)
        L_old = torch.tensor(L_old, dtype=torch.float32, device=device)
        FL_old = torch.tensor(FL_old, dtype=torch.float32, device=device)
        delta_standard_F = torch.tensor(delta_standard_F, dtype=torch.float32, device=device)
        delta_standard_L = torch.tensor(delta_standard_L, dtype=torch.float32, device=device)

    return F_old, L_old, FL_old, delta_standard_F, delta_standard_L, small_change_counter_L, small_change_counter_F, k_old

def PCA_MISSING(X: Matrix, 
                dim_factor: int, 
                restrict: str = 'optimize',
                max_iterations: int = 500,
                init: float = 1,
                Tol = 1e-5,
                Torch_cuda: bool = False):
    
    if Torch_cuda:
        device = get_device()
        X = move_to_device(X, device)

        def replace_missing(X, FL, FL_old=None):
            X_star = X.clone()
            missings = torch.isnan(X) if Torch_cuda else np.isnan(X)
            X_star[missings] = FL[missings]
            if FL_old is not None:
                delta = torch.mean(abs(FL[:,-1] - FL_old[:,-1]))/(torch.linalg.vector_norm(FL[:,-1]) + 1e-9)
                return X_star, delta
            return X_star, None

        def update(F, L, k_new):
            k_old, F_old, L_old = k_new, F, L
            FL_old = F @ L.T
            return k_old, F_old, L_old, FL_old
        
    else:
        def replace_missing(X, FL, FL_old=None):
            X_star = X.copy()
            missings = torch.isnan(X) if Torch_cuda else np.isnan(X)
            X_star[missings] = FL[missings]
            if FL_old is not None:
                delta = np.mean(abs(FL[:,-1] - FL_old[:,-1]))/(np.linalg.norm(FL[:,-1]) + 1e-9)
                return X_star, delta
            return X_star, None
        
        def update(F, L, k_new):
            k_old, F_old, L_old = k_new, F, L
            FL_old = F @ L.T
            return k_old, F_old, L_old, FL_old

    T, N = X.shape
    # Initialize factors and tracking variables
    F_old, L_old, FL_old, delta_standard_F, delta_standard_L, small_change_counter_F, small_change_counter_L, k_old = _initial_points(init, dim_factor,  T, N, Torch_cuda)
    X_star, _ = replace_missing(X, FL_old)
    
    # Algorithm
    for iteration in range(1, max_iterations + 1):
        
        # Compute factor decomposition
        F, L, N, T, dim_factor, restrict, device, _ = _PCA(X_star, dim_factor, Torch_cuda=Torch_cuda)
        
        FL = F @ L.T

        # Check convergence for F
        X_star_new, delta = replace_missing(X, FL, FL_old)


        # Stop if both converge
        if delta <= Tol:
            break

        # Update previous values
        FL_old, F_old, L_old, X_star = FL, F, L, X_star_new

    return F, L, N, T, dim_factor, restrict, device, iteration

##########################
### Estimating Factors ###
###         FDE        ###
##########################

def _criteria_function_PC(k, V, sigma2_hat, penalty):
    return V[k - 1] + k * sigma2_hat * penalty

def _criteria_function_IC(k, V, sigma2_hat, penalty):
    return np.log(V[k - 1]) + k * penalty

def _SSE(k, X, F_hat, L_hat):
    F_k = F_hat[:, :k]
    L_k = L_hat[:, :k]
    FL = F_hat[:, :k] @ L_hat[:, :k].T
    return np.nanmean((X - FL) ** 2)

def _SSE_Cuda(k, X, F_hat, L_hat):
    F_k = F_hat[:, :k]
    L_k = L_hat[:, :k]
    FL = torch.matmul(F_k, L_k.T)
    return torch.nanmean((X - FL) ** 2)

def _sigma2_hat(X, F_hat, L_hat):
    e_kmax = X - F_hat @ L_hat.T
    return np.nanmean(e_kmax**2) if not isinstance(X, torch.Tensor) else torch.nanmean(e_kmax**2).cpu().numpy()

def _choose_k(
            F_hat,
            L_hat,
            X: Matrix,
            sigma2_hat: float,
            k_max,
            Criteria,
            Torch_cuda,
            device):
    
    #T, N = X.shape
    # adjust to missing values:
    T, N = X.shape

    # Calculate V(k, F^k)
    if Torch_cuda:
        V = torch.tensor([_SSE_Cuda(k, X, F_hat, L_hat) for k in range(1, k_max + 1)], device=device)
        V = V.cpu().numpy()
    else:
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
        criteria_matrix[i] = [globals()['_criteria_function_' + fun](k, V, sigma2_hat, penalty) for k in range(1, k_max + 1)]

    # Determine the factor dimensionality
    k = np.argmax(-criteria_matrix[0, :]) + 1

    return k, criteria_matrix

def _FDE(X: Matrix,
        k_max: k_max_class = 10,
        Criteria: criteria_class = ['PC1', 'PC2', 'PC3', 'IC1', 'IC2', 'IC3'],
        restrict: str = 'optimize',
        Torch_cuda: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    
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
    '''

    # Calculate the factor using the PCA method with k_max
    F_hat, L_hat, _, _, _, _, device, X = _PCA(X, k_max, restrict=restrict, Torch_cuda=Torch_cuda)

    # Calculate the residuals
    sigma2_hat = _sigma2_hat(X, F_hat, L_hat)

    # Choose the optimal k based on the criteria
    k, criteria_matrix = _choose_k(F_hat, L_hat, X, sigma2_hat, k_max, Criteria, Torch_cuda, device)

    return criteria_matrix, F_hat[:, :k], L_hat[:, :k], k

##### missing Algorithm #####

def _FDE_MISSING(X: Matrix, 
                k_max: k_max_class = 10,
                Criteria: criteria_class = ['IC1', 'IC2', 'IC3', 'PC1', 'PC2', 'PC3'],
                restrict: str = 'optimize',
                Torch_cuda: bool = False,
                # Parameters for the missing algorithm
                max_iterations: int = 1000,
                init: float = 1,
                Tol = 1e-6,
                Tol_2deriv = 1e-5,
                ):
    
    if Torch_cuda:
        device = get_device()
        X = move_to_device(X, device)

        def replace_missing(X, FL, FL_old=None):
            X_star = X.clone()
            missings = torch.isnan(X) if Torch_cuda else np.isnan(X)
            X_star[missings] = FL[missings]
            if FL_old is not None:
                delta = torch.mean(abs(FL[:,-1] - FL_old[:,-1]))/(torch.linalg.vector_norm(FL[:,-1]) + 1e-9)
                return X_star, delta
            return X_star, None

        def update(F, L, k_new):
            k_old, F_old, L_old = k_new, F, L
            FL_old = F @ L.T
            return k_old, F_old, L_old, FL_old
        
    else:
        def replace_missing(X, FL, FL_old=None):
            X_star = X.copy()
            missings = torch.isnan(X) if Torch_cuda else np.isnan(X)
            X_star[missings] = FL[missings]
            if FL_old is not None:
                delta = np.mean(abs(FL[:,-1] - FL_old[:,-1]))/(np.linalg.norm(FL[:,-1]) + 1e-9)
                return X_star, delta
            return X_star, None
        
        def update(F, L, k_new):
            k_old, F_old, L_old = k_new, F, L
            FL_old = F @ L.T
            return k_old, F_old, L_old, FL_old
    
    T, N = X.shape
    # Initialize factors and tracking variables
    F_old, L_old, FL_old, delta_standard_F, delta_standard_L, small_change_counter_F, small_change_counter_L, k_old = _initial_points(init, k_max,  T, N, Torch_cuda)
    X_star, _ = replace_missing(X, FL_old)

    for iteration in range(1, max_iterations + 1):
        
        # Compute factor decomposition
        F_hat, L_hat, _, _, _, _, device, _ = _PCA(X_star, k_max, restrict=restrict, Torch_cuda=Torch_cuda)
        sigma2_hat = _sigma2_hat(X, F_hat, L_hat)
        k_new, criteria_matrix = _choose_k(F_hat, L_hat, X_star, sigma2_hat, k_max, Criteria, Torch_cuda, device)
        F, L = F_hat[:, :k_new], L_hat[:, :k_new]
        
        #criteria_matrix, F, L, k_new = _FDE(X_star, k_max, Criteria=Criteria, restrict=restrict, Torch_cuda=Torch_cuda)

        # Compute new factor loadings
        FL = F @ L.T
        # Check convergence for F
        X_star_new, delta = replace_missing(X, FL, FL_old)

        # If rank changes, reinitialize
        if k_old != k_new:
            k_old, F_old, L_old, FL_old = update(F, L, k_new)
            X_star = X_star_new
            continue

        # Stop if both converge
        if delta <= Tol:
            break

        # Update previous values
        FL_old, F_old, L_old, X_star = FL, F, L, X_star_new

    return criteria_matrix, F, L, k_new, iteration

class FDE_output:
    def __init__(self, criteria_matrix, Criteria, k_max, F_hat, L_hat, k, X, residuals, Number_Iterations = None):

        k_list = [f"K={k}" for k in range(1, k_max + 1)]
        criteria_pd = pd.DataFrame(criteria_matrix, columns = k_list, index = Criteria) if criteria_matrix is not None else None
        self.Data = move_to_cpu(X) 
        self.criteria_matrix = criteria_pd 
        self.Criterias = Criteria 
        self.k_max = k_max
        self.F_hat = move_to_cpu(F_hat)
        self.L_hat = move_to_cpu(L_hat)
        self.k = k
        self.residuals = move_to_cpu(residuals)
        self.Number_Iterations = Number_Iterations
    
    def summary(self):

        if self.criteria_matrix is not None:
            k_opt = [np.argmin(self.criteria_matrix.iloc[i, :]) + 1 for i in range(len(self.criteria_matrix.index))]
            k_opt = pd.DataFrame([k_opt], columns=self.criteria_matrix.index, index=['K estimated'])
        else:
            k_opt = self.k

        print("\n")
        print("** Factor Dimensionality Estimation Summary **")
        print("\n")
        print("Criteria K estimated:")
        print(k_opt)
        print("\n")
        print(f"Selected Criteria: {self.Criterias[0]}")
        print(f"Selected Factor Dimensionality: {self.k}")
        print("Residuals: Mean = ", np.nanmean(self.residuals).round(4), ", Std = ", np.nanstd(self.residuals).round(4), ". R^2: ", 1 - np.nanvar(self.residuals) / np.nanvar(self.Data), sep = '')
        print("\n")
        if self.criteria_matrix is None:
            print(f"Warning: k = {self.k} was defined by the user. No criteria matrix was calculated.")
        else:
            print("Criteria Matrix with max factor dimensionality (K_max) equal to ", self.k_max, ":")
            print(self.criteria_matrix)
        print("\n")
        print("Algorithm based on Bai and Ng (2002): https://doi.org/10.1111/1468-0262.00273")
        return None

def factor_dimensionality(
        X: Matrix,
        k: int = None, 
        k_max: k_max_class = 10,
        Criteria: criteria_class = ['IC1', 'IC2', 'IC3', 'PC1', 'PC2', 'PC3'],
        restrict: str = 'optimize',
        Torch_cuda: bool = False) -> FDE_output:
    '''
    Estimates the **number of factors (k)** in large-dimensional factor models, along with the **common factors (F)** and **factor loadings (L)** for a given matrix T x N (**X**). The estimation is based on one of the criteria proposed by Bai and Ng (2002): https://doi.org/10.1111/1468-0262.00273.

    Parameters:
        X (numpy.ndarray): The data matrix of shape (T, N),  
            where **T** is the number of time observations,  
            and **N** is the number of individuals (variables).
        k (int, optional): The number of factors to estimate. If not define (**None**), the function will estimate the number of factors. Default is **None**.
        k_max (int, optional): The maximum number of principal components to consider.  
            Must be less than **min(T, N)**. Default is **10**. 
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
            Default is **['IC1', 'IC2', 'IC3', 'PC1', 'PC2', 'PC3']**.
        restrict (str, optional): The restriction method for calculating factors.  
            Options are:  
            - 'common': Restricts the common factor such that FᵀF / T = I.  
            - 'loading': Restricts the loading factor such that LᵀL / N = I.  
            - 'optimize' (default): Automatically selects the most computationally efficient option:  
                - Uses 'common' if N ≥ T.  
                - Uses 'loading' if T > N.  
        Torch_cuda (bool, optional): Whether to use PyTorch for computations. If cuda torch was available, the computations will be done on the GPU else they use pytorch in cpu. Default is **False**.
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
    X = _validate_input(X, Matrix)
    T, N = X.shape
    k = _validate_input(k, k_class, N=N, T=T)
    k_max = _validate_input(k_max, k_max_class, N=N, T=T)
    Criteria = _validate_input(Criteria, criteria_class)
    
    missing_values = np.isnan(X).sum()
    has_missing = missing_values > 0

    if has_missing:
        warnings.warn("Warning: Missing values in the data matrix. The algorithm to estimate the factors with missing values was used.")

    if k is not None and has_missing:
        F_hat, L_hat, _, _, _, _, _, Number_Iterations = PCA_MISSING(X, k, restrict, Torch_cuda=Torch_cuda)
        criteria_matrix = None
        if Torch_cuda:
            F_hat = F_hat.cpu().numpy()
            L_hat = L_hat.cpu().numpy()
        return FDE_output(criteria_matrix, Criteria, k_max, F_hat, L_hat, k, X, X - F_hat @ L_hat.T, Number_Iterations)

    if k is not None:
        F_hat, L_hat, _, _, _, _, _, _ = _PCA(X, k, restrict, Torch_cuda)
        criteria_matrix = None
        if Torch_cuda:
            F_hat = F_hat.cpu().numpy()
            L_hat = L_hat.cpu().numpy()
        return FDE_output(criteria_matrix, Criteria, k_max, F_hat, L_hat, k, X, X - F_hat @ L_hat.T)
    if has_missing:
        criteria_matrix, F_hat, L_hat, k, Number_Iterations = _FDE_MISSING(X, k_max, Criteria, restrict, Torch_cuda)
    else:
        criteria_matrix, F_hat, L_hat, k = _FDE(X, k_max, Criteria, restrict, Torch_cuda)
        Number_Iterations = None

    if Torch_cuda:
        F_hat = F_hat.cpu().numpy()
        L_hat = L_hat.cpu().numpy()

    return FDE_output(criteria_matrix, Criteria, k_max, F_hat, L_hat, k, X, X - F_hat @ L_hat.T, Number_Iterations)