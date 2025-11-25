
import numpy as np
import pandas as pd
from scipy import stats
from .class_def import Matrix, k_class, k_max_class, criteria_class, var_type, fixed_effect, criteria_conv_class, restrict_PCA_class, tolerance_class, SOR_hyperparam_class, boll_class
import warnings

# Helper function to validate and convert input types

def _validate_input(input, class_type, **kwargs):
    if not isinstance(input, class_type):
            input = class_type(input, **kwargs)
    return input

# Output Formatter

def _output(F, L, N, T, dim_factor, restrict, N_int = None):
    return F, L, {'N': N, 'T': T, 'dim_factor': dim_factor, 'restrict utilized': restrict, 'Number_iteration': N_int}

# -------------------------------------------------- #
# -------------- PCA Estimation -------------------- #
# -------------------------------------------------- #

def PCA(X: Matrix, 
        dim_factor: int, 
        restrict: restrict_PCA_class = 'optimize',
        torch_cuda: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
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
    restrict = _validate_input(restrict, restrict_PCA_class)


    if torch_cuda:
        from .factor_dimensionality_torch import _PCA

        F, L, N, T, dim_factor, restrict, device, _ = _PCA(X, dim_factor, restrict)
        F = F.cpu().numpy()
        L = L.cpu().numpy()
        N = int(N.cpu().numpy())
        T = int(T.cpu().numpy())
        restrict = str(restrict)

        return _output(F, L, N, T, dim_factor, restrict)

    from .factor_dimensionality_numpy import _PCA

    F, L, N, T, dim_factor, restrict, _ = _PCA(X, dim_factor)

    return _output(F, L, N, T, dim_factor, restrict)
        
# missing_values = np.isnan(X).sum()
    # if missing_values > 0:
    #     Warning(f"Warning: Missing values in the data matrix. The algorithm to estimate the factors with missing values was used.")
    #     F, L, N, T, dim_factor, restrict, device, N_int = PCA_MISSING(X, dim_factor, restrict, Torch_cuda=Torch_cuda) 

# -------------------------------------------------- #
# -------------- FDE Estimation -------------------- #
# -------------------------------------------------- #

class FDE_output:
    def __init__(self, criteria_matrix, Criteria, k_max, F_hat, L_hat, k, k_all, X, residuals, Number_Iterations = None):

        k_list = [f"K={k}" for k in range(1, k_max + 1)]
        criteria_pd = pd.DataFrame(criteria_matrix, columns = k_list, index = Criteria) if criteria_matrix is not None else None
        self.Data = X 
        self.criteria_matrix = criteria_pd 
        self.Criterias = Criteria 
        self.k_max = k_max
        self.F_hat = F_hat
        self.L_hat = L_hat
        self.k = k
        self.k_all = k_all
        self.residuals = residuals
        self.Number_Iterations = Number_Iterations
    
    def summary(self):

        if self.criteria_matrix is not None:
            k_opt = pd.DataFrame([self.k_all], columns=self.criteria_matrix.index, index=['K estimated'])
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
        criteria: criteria_class = ['IC1', 'IC2', 'IC3', 'PC1', 'PC2', 'PC3'],
        restrict: str = 'optimize',
        torch_cuda: bool = False) -> FDE_output:
    """
    Estimate the number of factors (k) and the factor matrices F and L using Bai & Ng (2002).

    Parameters
    ----------
    X : Matrix or numpy.ndarray
        Data matrix of shape (T, N) (T time observations, N cross-sectional units).
    k : int or None, optional
        If provided, use this fixed number of factors. If None (default), k is estimated.
    k_max : int or k_max_class, optional
        Maximum number of factors to consider when estimating k. Default: 10.
    Criteria : list or criteria_class, optional
        List of criteria to use for estimating k. Defaults to
        ['IC1', 'IC2', 'IC3', 'PC1', 'PC2', 'PC3'].
    restrict : {'common', 'loading', 'optimize'}, optional
        Restriction used when computing factors:
          - 'common': enforce F.T @ F / T = I
          - 'loading': enforce L.T @ L / N = I
          - 'optimize': pick the computationally cheaper option based on N and T.
    Torch_cuda : bool, optional
        If True and PyTorch CUDA is available, perform computations on GPU. Default: False.

    Returns
    -------
    FDE_output
        Object containing:
          - Data: original data matrix X
          - criteria_matrix: DataFrame of evaluated criteria (or None if k provided)
          - Criterias: criteria list used
          - k_max: maximum k considered
          - F_hat: estimated common factors (T x k)
          - L_hat: estimated loadings (N x k)
          - k: selected number of factors
          - residuals: X - F_hat @ L_hat.T
          - Number_Iterations: iterations used for algorithms handling missing data (if any)

    Notes
    -----
    Criteria definitions follow Bai and Ng (2002). IC criteria are log-based (penalized log V(k)),
    PC criteria are in-sample squared-error criteria adjusted by penalty terms.

    Example
    -------
    >>> FDE_res = factor_dimensionality(X)
    >>> FDE_res.summary()
    """
    
    # Validate inputs #
    X = _validate_input(X, Matrix)
    T, N = X.shape
    k = _validate_input(k, k_class, N=N, T=T)
    k_max = _validate_input(k_max, k_max_class, N=N, T=T)
    criteria = _validate_input(criteria, criteria_class)
    
    missing_values = np.isnan(X).sum()
    has_missing = missing_values > 0

    if has_missing:
        warnings.warn("Warning: Missing values in the data matrix. The algorithm to estimate the factors with missing values was used.")

    if not torch_cuda:

        from .factor_dimensionality_numpy import _PCA, _FDE
        
        if k is not None:
            warnings.warn("Warning: k is provided by the user. No factor dimensionality estimation will be performed.")
            
            F_hat, L_hat, _, _, _, _, _, _ = _PCA(X, k, restrict)
            criteria_matrix = None 
            k_all = None
            return FDE_output(criteria_matrix, criteria, k_max, F_hat, L_hat, k, k_all, X, X - F_hat @ L_hat.T, Number_Iterations = None)
        
        if k is None and not has_missing:
            criteria_matrix, F_hat, L_hat, k, k_all = _FDE(X, k_max, criteria, restrict)

            return FDE_output(criteria_matrix, criteria, k_max, F_hat, L_hat, k, k_all, X, X - F_hat @ L_hat.T, Number_Iterations = None)

    if torch_cuda:

        from .factor_dimensionality_torch import _PCA, _FDE
       
        if k is not None:
            warnings.warn("Warning: k is provided by the user. No factor dimensionality estimation will be performed.")
            
            F_hat, L_hat, _, _, _, _, _, _ = _PCA(X, k, restrict)
            F_hat = F_hat.cpu().numpy()
            L_hat = L_hat.cpu().numpy()
            criteria_matrix = None
            return FDE_output(criteria_matrix, criteria, k_max, F_hat, L_hat, k, k_all, X, X - F_hat @ L_hat.T, Number_Iterations = None)
        
        if k is None and not has_missing:
            criteria_matrix, F_hat, L_hat, k, k_all = _FDE(X, k_max, criteria, restrict)

            F_hat = F_hat.cpu().numpy()
            L_hat = L_hat.cpu().numpy()
            criteria_matrix = criteria_matrix.cpu().numpy()
            k = int(k.cpu().numpy())
            k_all = k_all.cpu().numpy()
            
            return FDE_output(criteria_matrix, criteria, k_max, F_hat, L_hat, k, k_all, X, X - F_hat @ L_hat.T, Number_Iterations = None)

    # if k is not None and has_missing:
    #     F_hat, L_hat, _, _, _, _, _, Number_Iterations = PCA_MISSING(X, k, restrict, Torch_cuda=Torch_cuda)
    #     criteria_matrix = None
    #     if Torch_cuda:
    #         F_hat = F_hat.cpu().numpy()
    #         L_hat = L_hat.cpu().numpy()
    #     return FDE_output(criteria_matrix, Criteria, k_max, F_hat, L_hat, k, X, X - F_hat @ L_hat.T, Number_Iterations)

# -------------------------------------------------- #
# -------------- IFE Estimation -------------------- #
# -------------------------------------------------- #

class InteractiveFixedEffectModelOutput:
    
    class DataContainer:
        def __init__(self, Y: np.ndarray, X: np.ndarray):
            self.Y = Y
            self.X = X

    def __init__(self, beta, F_hat, L_hat, k, N_sim, converges, crit_eval, 
                 residuals, cov, criteria, 
                 Y, X, fixed_effects, A, D0, Z,
                 df):
               
        self.N = Y.shape[1]  # Number of entities
        self.T = Y.shape[0]  # Number of time periods

        self.A = A
        self.D0 = D0
        self.Z = Z
        
        self.beta = np.array(beta)
        self.data = self.DataContainer(Y, X)
        self.fixed_effects = fixed_effects
        self.F_hat = np.array(F_hat)
        self.L_hat = np.array(L_hat)
        self.k = k
        self.criteria = criteria
        self.num_iterations = N_sim
        self.converged = converges
        self.critical_convergence_criterios = crit_eval
        self.residuals = np.array(residuals)
        self.cov = np.array(cov)
        self.df = df

        # Compute statistics
        self.coefficients_matrix = self._compute_coefficients(beta)

    def _compute_coefficients(self, beta: np.ndarray) -> pd.DataFrame:
        """Computes the coefficient estimates, standard errors, t-values, and p-values."""
        beta = beta.flatten()
        sd = np.array([np.sqrt(self.cov[ i, i])/np.sqrt(self.N*self.T) for i in range(beta.shape[0])])
        t = np.array([beta[i]/sd[i] for i in range(beta.shape[0])])
        p_value = np.array([2*(1 - stats.t.cdf(np.abs(t[i]), self.N*self.T - 1)) for i in range(beta.shape[0])])


        return pd.DataFrame({
            'Estimate': beta,
            'Std. Error': sd,
            't-value': t,
            'p-Value': p_value
        })

    def summary(self):

        caracther_number = 45

        """Prints a formatted summary of the model results."""
        print("\n** Interactive Fixed Effect Model Summary **\n")

        print("Estimated Coefficients\n")
        print(self.coefficients_matrix.round(4))
        print('-' * caracther_number)

        print("\nFactor Characteristics\n")
        print(f"Selected Criteria: {self.criteria[0]}")
        print(f"Selected Factor Dimensionality: {self.k}")
        print(f"Loading Factors: Mean = {np.mean(self.L_hat, axis=0).round(4)}")
        print(f"Factor Structure: Mean = {np.mean(self.F_hat, axis=0).round(4)}")
        print('-' * caracther_number)

        print("\nResiduals")
        r_squared = 1 - np.var(self.residuals) / np.var(self.data.Y)
        r_std = np.sqrt(np.sum(self.residuals ** 2) / self.df)
        print(f" Mean = {np.mean(self.residuals):.4f}, Std = {r_std:.4f}, R² = {r_squared:.4f}")
        if self.fixed_effects is not None:
                print("\nFixed Effects")
                print(f"Intercept: {self.fixed_effects['intercept'].flatten()[0].round(4) if self.fixed_effects['intercept'] is not None else 'N/A'}"), 
                print(f"* Individual and time fixed effect can be ")
                print("accessed by 'output.fixed_effects' attribute.")
        print('-' * caracther_number)
        print("\nConvergence\n" if self.converged else "\nNo Convergence\n")
        print(f"Iterations until convergence: {self.num_iterations}")
        
        num_criteria = len(self.critical_convergence_criterios)
        names = ['Relative Norm: ', 'Objective Function Diff: ', 'Gradient Norm: '][:num_criteria]
        result_string = " \n".join([f"{name}{eval:.12f}" for name, eval in zip(names, self.critical_convergence_criterios)])
        print(result_string + "\n")

def IFE(Y: Matrix,
        X: list[Matrix],
        k: int = None,
        fixed_effects: fixed_effect = 'twoways',
        variance_type: var_type = 'iid',
        criteria: criteria_class = ['IC1'],
        k_max: k_max_class = 8,
        max_iter: int = 10_000,
        convergence_criteria: criteria_conv_class = ['Relative_norm', 'Obj_fun', 'Grad_norm'],
        tolerance: np.ndarray = np.array([1e-6, 1e-12, 1e-5]),
        SOR_hyperparam: float = 1.0,
        max_SOR_hyperparam: float = None,
        inc_factor: float = 1.1, 
        dec_factor: float = 0.5,
        verbose: bool = False,
        torch_cuda: bool = False) -> InteractiveFixedEffectModelOutput:
        
        '''
        Estimates the **Interactive Fixed Effects** model for a large panel dataset by the following equation:

        .. math:: Y_{it} = X_{it} \\beta + F_{it} L^{T} + \epsilon_{it}
        
        The algorithm is based on the paper **"PANEL DATA MODELS WITH INTERACTIVE FIXED EFFECTS"** by Bai (2009): https://doi.org/10.3982/ECTA6135.
        
        Parameters:
                Y (numpy.ndarray or torch.Tensor): Dependent variable of the model. Shape (T, N).
                X (list[numpy.ndarray or torch.Tensor]): List of covariates. Each element of the list is a covariate matrix. Shape (T, N, p).
                k (int, optional): Number of factors to be estimated. Default is **None**. If **None**, the number of factors is estimated by the factor dimensionality algorithm.
                fixed_effects (str, optional): Type of fixed effects. Options are: 
                        - 'Twoways' (default): Include both individual and time fixed effects. We estimate the model by transforming the data as described in Bai (2009). More efficient and recommended.
                        - 'None': Not include fixed effects. In this case, the model is estimated by the original data.

                Variance_type (str, optional): Type of variance-covariance matrix: 
                        - 'iid': Assume that the error in model is independent and identically distributed. So, sqrt(NT) (beta_hat - beta) ~ N(0, σ^2 D^(-1)).
                        - 'heteroskedastic': Assume the error in model is independent but heteroskedastic in both dimensions. In this case, the estimator is biased, so we aplly a bias correction as described in Bai (2009):
                                - β_hat = β - (1/N) B - (1/T) C
                                - \sqrt(NT) (β_hat - β) ~ N(0, D_0^(-1) D_3 D_0^(-1))
                        see Bai (2009) for more details.
                        Default is 'iid'.
                Criteria (list, optional): The criteria used to estimate the factor dimensionality. All criterias in the list will be printed but use only the first criteria in the list to determine the factor dimensionality.
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
                                Default is ['IC1'].
                k_max (int, optional): Maximum number of factors to be estimated. Must be less than **min(T, N)**. Default is **8**. If the true dimensionality is less than K_max the algorithm will be consistent, however, more k_max will increase the computational cost. 
                        Default is 8.
                Tol (float, optional): Tolerance for convergence. Default is 1e-6.
                Tol_2order (float, optional): Tolerance for second order convergence. This tolerance is used to check if the first order convergence criteria is no longer decreasing significantly across interaction. How much bigger is this value, faster they stop the interaction in case of non-convergence, however, a high value can stop the code before converging. Default is 1e-9.
                Max_iter (int, optional): Maximum number of iterations. Default is 2_000. 
                verbose (bool, optional): If **True**, print the convergence information at each iteration. Default is **False**.
                Torch_cuda (bool): Use torch GPU for calculations. If a GPU is available, increase considerably the performance for matrix with large dimensions (N, T > 500). However, cuda 11.8+ torch library is necessary. For installation of cuda torch see: https://pytorch.org/get-started/locally/. It is recomended utilized a virtual enviroment in this case and is mandatory install cuda torch before the installation of the this function's library. If the cuda is not installed on the current virtual enviroment, exclude torch library and reinstall it cuda version. Default is **False**.
        
        Returns:
                IFE_OUTPUT: A class (IFE_OUTPUT) containing the following attributes: 
                        - 'Data' (class): A class containing the original data used in the model. Data.Y is the dependent variable and Data.X is the list of covariates matrix.
                        - 'Coefficients_matrix' (panda.DataFrame): A pandas DataFrame with the estimated coefficients, standard deviation, t-statistic and p-value.
                        - 'F_hat' (numpy.ndarray): Estimated factors structure. Shape (T, k).
                        - 'L_hat' (numpy.ndarray): Estimated loadings structure. Shape (N, k).
                        - 'k' (int): Number of factors.
                        - 'residuals' (numpy.ndarray): Residuals of the model. Shape (T, N).
                        - 'Criteria' (str): Criteria used to estimate the factor dimensionality.
                        - 'Number_of_interaction' (int): Number of iterations until convergence. If Number_of_interaction is equal to Max_iter, the algorithm did not converge.
                        - 'Convergence_value' (int): Convergence value calculated as the norm of the difference between the estimated coefficients in two consecutive iterations.
                        - 'Convergence_value_stand' (int): Convergence value standardized by the norm of the estimated coefficients.
                        - 'Second_order_Convergence_value' (int): Convergence value calculated as the difference between the convergence value in two consecutive iterations. (standardized by the norm of the estimated coefficients).
                        - 'cov' (numpy.ndarray): Estimated covariance matrix of the coefficients. Shape (p, p).
        
        Example:

        ```python

        # Create a random dataset
        T, N, k = 200, 100, 2
        F = np.random.normal(1, 1, (T, k))
        L = np.random.normal(-2, 1, (N, k))
        E = np.random.normal(0, 1, (T, N))
        Lx1 = np.random.normal(2, 1, (N, k))
        Lx2 = np.random.normal(-1, 1, (N, k))
        X1 = 1 + F @ Lx1.T + np.random.normal(0, 1, (T, N))
        X2 = 2 + F @ Lx2.T + np.random.normal(0, 1, (T, N))

        alpha = -2
        beta1 = -1
        beta2 = 2

        Y = alpha + beta1 * X1 + beta2 * X2 + F @ L.T + E

        # Estimate the model
        import factorAnalysis as fa
        output = fa.IFE(Y, [X1, X2], Criteria=['IC1'], k_max=8)

        # Print the summary
        output.summary()

        # Access the estimated coefficients
        output.Coefficients_matrix

        # Access the estimated factors structure
        common_factor = output.F_hat
        loading_factor = output.L_hat

        FL = F @ L.T

        ```
        '''

        # Validate inputs #
        Y = _validate_input(Y, Matrix)
        X = [_validate_input(x, Matrix) for x in X]
        T,N = Y.shape
        k = _validate_input(k, k_class, N=N, T=T)
        k_max = _validate_input(k_max, k_max_class, N=N, T=T)
        fixed_effects = _validate_input(fixed_effects, fixed_effect)
        variance_type = _validate_input(variance_type, var_type)
        criteria = _validate_input(criteria, criteria_class)
        convergence_criteria = _validate_input(convergence_criteria, criteria_conv_class)
        max_iter = _validate_input(max_iter, int)
        tolerance = _validate_input(tolerance, tolerance_class, convergence_criteria=convergence_criteria)
        SOR_hyperparam, max_SOR_hyperparam, inc_factor, dec_factor = _validate_input(SOR_hyperparam, SOR_hyperparam_class, max_SOR_hyperparam=max_SOR_hyperparam, inc_factor=inc_factor, dec_factor=dec_factor)
        verbose = _validate_input(verbose, boll_class, name = 'echo')
        torch_cuda = _validate_input(torch_cuda, boll_class, name = 'torch_cuda')
        #save_path = _validate_input(save_path, boll_class, name = 'save_path')
        

        if not torch_cuda:
            from .PDMwIFE_numpy import _est_alg, _get_two_way_transform_data, _demean, _get_fixed_effects, _VAR_COV_Estimation
        else:
            from .PDMwIFE_torch import _est_alg, _get_two_way_transform_data, _demean, _get_fixed_effects, _VAR_COV_Estimation, move_to_cpu
             
        if fixed_effects == 'twoways':
            dotY, Y_tdot, Y_idot, Y_dobledot, dotX, X_tdot, X_idot, X_dobledot = _get_two_way_transform_data(Y, X)
            Y_adj = dotY
            X_adj = dotX
        elif fixed_effects == 'demeaned':
            Y_demeaned, X_demeaned = _demean(Y, X)
            Y_adj = Y_demeaned
            X_adj = X_demeaned
        else:
            Y_adj = Y
            X_adj = X

        # Estimate the coefficients, factors and loadings
        beta, F_hat, L_hat, k, N_sim, converges, crit_eval = _est_alg(
            Y_adj, X_adj, k, criteria, k_max, 'common', #restrict = common loading (F'F/T = I) 
            max_iter, convergence_criteria, tolerance, 
            SOR_hyperparam, max_SOR_hyperparam, inc_factor, dec_factor,
            verbose, save_path = False
        )

        # Compute variance-covariance matrix
        beta, cov, residuals, A, D0, inv_D0, X_2, Z, df = _VAR_COV_Estimation(Y_adj, X_adj, beta, F_hat, L_hat, variance_type, fixed_effects)

        if torch_cuda:

            beta = move_to_cpu(beta)
            F_hat = move_to_cpu(F_hat)
            L_hat = move_to_cpu(L_hat)
            residuals = move_to_cpu(residuals)
            cov = move_to_cpu(cov)
            Y = move_to_cpu(Y)
            Z = move_to_cpu(Z)
            

        k_hat = F_hat.shape[1]

        # Compute fixed effects if necessary
        if fixed_effects == 'twoways':
            mu, alpha_i, gamma_t = _get_fixed_effects(fixed_effects=fixed_effects, beta=beta, Y_tdot=Y_tdot, Y_idot=Y_idot, Y_dobledot=Y_dobledot, X_tdot=X_tdot, X_idot=X_idot, X_dobledot=X_dobledot)
        elif fixed_effects == 'demeaned':
            mu, alpha_i, gamma_t = _get_fixed_effects(fixed_effects=fixed_effects, beta=beta, Y=Y, X=X, F_hat=F_hat, L_hat=L_hat)
        else:
            mu, alpha_i, gamma_t = None, None, None

        fixed_effects = {'intercept': mu, 'individual': alpha_i, 'time': gamma_t}

        return InteractiveFixedEffectModelOutput(beta, F_hat, L_hat, k_hat, N_sim, converges, crit_eval, residuals, cov, criteria, Y, X_2, fixed_effects, A, D0, Z, df)
