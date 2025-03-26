from .factor_dimensionality import _FDE, _PCA
from .class_def import Matrix, k_class, k_max_class, criteria_class, var_type, fixed_effect
from .Device_aux_functions import move_to_device, get_device, move_to_cpu
import numpy as np
import torch
import pandas as pd
import scipy.stats as stats
torch.backends.cudnn.deterministic = True

#######################    Inputs Validation   #######################

def _validate_input(input, class_type, **kwargs):
        if not isinstance(input, class_type):
              input = class_type(input, **kwargs)
        return input

####################### Two way transformation #######################

def _two_ways_transform(X):
        T, N = X.shape

        # Compute the means
        X_tdot = np.mean(X, axis=1, keepdims=True)  # Mean across columns (over N)
        X_idot = np.mean(X, axis=0, keepdims=True)  # Mean across rows (over T)
        X_dobledot = np.mean(X)

        dotX = X - X_tdot - X_idot + X_dobledot

        return dotX, X_tdot, X_idot, X_dobledot
      
def _get_two_way_transform_data(Y, X):

        T, N = Y.shape

        dotX, X_tdot, X_idot, X_dobledot = [], np.zeros((T, 0)), np.zeros((0, N)), np.zeros((0, 1))
        for X_i in X:
                dotx, x_tdot, x_idot, x_dobledot = _two_ways_transform(X_i)
                dotX = dotX + [dotx]
                X_tdot = np.column_stack([X_tdot, x_tdot])
                X_idot = np.vstack([X_idot, x_idot])
                X_dobledot = np.vstack([X_dobledot, x_dobledot])

        dotY, Y_tdot, Y_idot, Y_dobledot = _two_ways_transform(Y)

        return dotY, Y_tdot, Y_idot, Y_dobledot, dotX, X_tdot, X_idot, X_dobledot

def _get_fixed_effects(beta, Y_tdot, Y_idot, Y_dobledot, X_tdot, X_idot, X_dobledot):
        
        #beta = np.array(beta, ndmin=2).T
        mu = Y_dobledot - X_dobledot.T @ beta
        alpha_i = Y_idot.T - X_idot.T @ beta - mu
        gamma_t = Y_tdot - X_tdot @ beta - mu

        return mu, alpha_i, gamma_t   

#####################  Estimation Algorithm   ########################

def _LS(X, XX_inv, y):
     XTy = X.T @ y
     return XX_inv @ XTy

def _convergence_criteria(beta, delta, previous_delta, small_change_counter, Tol, Tol_2deriv, convergence_threshold = 3):
        # Check if delta is less than tolerance
        beta_norm = np.linalg.norm(beta) + 1e-8 if not isinstance(beta, torch.Tensor) else torch.norm(beta) + 1e-8
        delta_standard = delta / beta_norm
        if delta_standard <= Tol:
                small_change_counter += 1
        else:
                small_change_counter = 0  # Reset if delta increases

        # Check if delta is no longer decreasing significantly
        delta_dev = abs(previous_delta - delta)/beta_norm
        if delta_dev < Tol_2deriv:
                small_change_counter += 1  # Treat as a small change

        # Final check: Stop only if multiple conditions are met
        if small_change_counter >= convergence_threshold:
                return True, None, delta_standard, delta_dev
        
        return False, small_change_counter, delta_standard, delta_dev

def _est_alg(Y: Matrix, 
             X: list[Matrix],
             k: int = None,
             Criteria: criteria_class = ['PC1'],
             k_max: k_max_class = 8,
             restrict: str = 'optimize',
             Tol: float = 1e-6,
             Tol_2deriv: float = 1e-9,
             Max_iter: int = 2_000,
             Torch_cuda: bool = False,
             echo = False) -> tuple:
        
        '''
        Internal function for computing the coefficients, factors structure and its dimensionality in a Large-panel model with interactive fixed effects. The algorithm is based on the paper **"PANEL DATA MODELS WITH INTERACTIVE FIXED EFFECTS"** by Bai (2009): https://doi.org/10.3982/ECTA6135.
        This function is intended for use inside other functions.  
        For direct usage, please refer to the `XX` function.

        Inputs: 
                See `XX` function.
        
        Returns: (tuple) with the following elements:
                - beta (numpy.ndarray): Coefficients of the model.
                - F_hat (numpy.ndarray): Factors structure.
                - L_hat (numpy.ndarray): Loadings structure.
                - k (int): Number of factors.
                - i (int): Number of iterations until convergence.
        '''

        # Define dimensions
        T, N = Y.shape

        # Defining covariate matrix (x) N*T x (p+1)
        x = np.hstack([M.reshape(-1, 1) for M in X])

        # Reshape Y to N*T x 1
        y = Y.reshape(-1, 1)

        if Torch_cuda:
                torch.cuda.empty_cache()

                device = get_device()

                x = move_to_device(x, device)
                y = move_to_device(y, device)
                Y = move_to_device(Y, device)
        
        xx_inv = np.linalg.inv(x.T @ x) if not Torch_cuda else torch.inverse(x.T @ x)

        # Initialize beta
        beta = _LS(x, xx_inv, y)
        previous_delta = 1
        small_change_counter = 0

        # Looping Algorithm
        for i in range(Max_iter):
 
                w = y - x @ beta
                W = w.reshape(T, N)
                if k is not None:
                       F_hat, L_hat, _, _, _, _, _, _ = _PCA(W, k, restrict, Torch_cuda) 
                else:
                       _, F_hat, L_hat, k = _FDE(W, k_max = k_max, Criteria=Criteria, restrict = restrict, Torch_cuda=Torch_cuda)
                
                Y_new = Y - F_hat @ L_hat.T
                y_new = Y_new.reshape(-1, 1)
                beta_new = _LS(x, xx_inv, y_new)

                delta = np.linalg.norm(beta_new - beta) if not Torch_cuda else torch.linalg.vector_norm(beta_new - beta)

                # Check convergence
                converges, small_change_counter, delta_standard, delta_dev = _convergence_criteria(beta, delta, previous_delta = previous_delta, small_change_counter = small_change_counter, Tol=Tol, Tol_2deriv=Tol_2deriv)
                
                beta = beta_new
                previous_delta = delta

                if echo:
                        print_progress(beta, k, i, delta_standard, delta_dev, Tol, Tol_2deriv)

                if converges:
                        break

        return beta, F_hat, L_hat, k, i+1, delta, previous_delta

def print_progress(beta, k, i, delta, delta_deriv, tol, tol_2deriv):
        import sys
        if i == 0:
                sys.stdout.write("\n")
                sys.stdout.write("\n")
                sys.stdout.write("\n")
                sys.stdout.write("\n")


        sys.stdout.write("\033[F" * 3)  # Move up 4 lines to overwrite previous values
        sys.stdout.write(f"Iteration {i+1}:\n")  
        sys.stdout.write(f"Theta = {beta.flatten().round(3)}, Number of Factors = {k}\n")  
        sys.stdout.write(f"Convergence = {delta:.9f}, Second Order Convergence = {delta_deriv:.9f}\n")
        sys.stdout.flush()

class InteractiveFixedEffectModelOutput:
    
    class DataContainer:
        def __init__(self, Y: np.ndarray, X: np.ndarray):
            self.Y = Y
            self.X = X

    def __init__(self, beta, F_hat, L_hat, k, N_sim, delta, previous_delta, 
                 residuals, cov, criteria, 
                 Y, X, fixed_effects, A, D0, Z):
               
        self.N = Y.shape[1]  # Number of entities
        self.T = Y.shape[0]  # Number of time periods

        self.A = A
        self.D0 = D0
        self.Z = Z
        
        self.beta = beta
        self.data = self.DataContainer(Y, X)
        self.fixed_effects = fixed_effects
        self.F_hat = F_hat
        self.L_hat = L_hat
        self.k = k
        self.criteria = criteria
        self.num_iterations = N_sim
        self.convergence_value = delta
        self.convergence_value_stand = delta / (np.linalg.norm(beta) + 1e-8)
        self.second_order_convergence_value = (delta - previous_delta) / (np.linalg.norm(beta) + 1e-8)
        self.residuals = residuals
        self.cov = cov

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
        print(f" Mean = {np.mean(self.residuals).round(4)}, Std = {np.std(self.residuals).round(4)}, R² = {r_squared.round(4)}")
        if self.fixed_effects is not None:
                print("\nFixed Effects")
                print(f"Intercept: {self.fixed_effects['intercept'][0,0].round(4)}"), 
                print(f"* Individual and time fixed effect can be ")
                print("accessed by 'output.fixed_effects' attribute.")
        print('-' * caracther_number)
        print("\nConvergence\n")
        print(f"Iterations until convergence: {self.num_iterations}")
        print(f"Convergence value: {self.convergence_value} (level)")
        print(f"Standardized Convergence value: {self.convergence_value_stand:.6f}")
        print(f"Second-order Convergence value (Standardized): {self.second_order_convergence_value:.6f}")

def IFE(Y: Matrix,
        X: list[Matrix],
        k: int = None,
        fixed_effects: fixed_effect = 'twoways',
        Variance_type: var_type = 'iid',
        Criteria: criteria_class = ['IC1'],
        k_max: k_max_class = 8,
        Tol: float = 1e-6,
        Tol_2order: float = 1e-9,
        Max_iter: int = 2_000,
        echo: bool = False,
        Torch_cuda: bool = False) -> InteractiveFixedEffectModelOutput:
        '''
        Estimates the **Interactive Fixed Effects** model for a large panel dataset by the following equation:

        .. math::
                Y_{it} = X_{it} \\beta + F_{it} L^{T} + \epsilon_{it}
        
        The algorithm is based on the paper **"PANEL DATA MODELS WITH INTERACTIVE FIXED EFFECTS"** by Bai (2009): https://doi.org/10.3982/ECTA6135.
        
        Parameters:
                Y (numpy.ndarray or torch.Tensor): Dependent variable of the model. Shape (T, N).
                X (list[numpy.ndarray or torch.Tensor]): List of covariates. Each element of the list is a covariate matrix. Shape (T, N, p).
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
                echo (bool, optional): Print the convergence criteria in each iteration. Default is **False**.
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
        Variance_type = _validate_input(Variance_type, var_type)
        Criteria = _validate_input(Criteria, criteria_class)

        if fixed_effects == 'twoways':
                dotY, Y_tdot, Y_idot, Y_dobledot, dotX, X_tdot, X_idot, X_dobledot = _get_two_way_transform_data(Y, X)
                Y = dotY
                X = dotX

        # Estimate the coefficients, factors and loadings
        beta, F_hat, L_hat, k, N_sim, delta, previous_delta = _est_alg(
                Y, X, k=k, restrict='common',
                Criteria=Criteria, k_max=k_max,
                Tol=Tol, Tol_2deriv=Tol_2order, Max_iter=Max_iter,
                Torch_cuda=Torch_cuda, echo=echo
        ) 

        # Compute variance-covariance matrix
        beta, cov, residuals, A, D0, inv_D0, X_2, Z = _VAR_COV_Estimation(Y, X, beta, F_hat, L_hat, Variance_type, Torch_cuda)

        if Torch_cuda:

                beta = move_to_cpu(beta)
                F_hat = move_to_cpu(F_hat)
                L_hat = move_to_cpu(L_hat)
                residuals = move_to_cpu(residuals)
                cov = move_to_cpu(cov)
                delta = move_to_cpu(delta)
                previous_delta = move_to_cpu(previous_delta)
                Y = move_to_cpu(Y)
                Z = move_to_cpu(Z)

        # Compute fixed effects if necessary
        if fixed_effects == 'twoways':
                mu, alpha_i, gamma_t = _get_fixed_effects(beta, Y_tdot, Y_idot, Y_dobledot, X_tdot, X_idot, X_dobledot)
                fixed_effects = {'intercept': mu, 'individual': alpha_i, 'time': gamma_t}
        else:
                fixed_effects = None


        return InteractiveFixedEffectModelOutput(beta, F_hat, L_hat, k, N_sim, delta, previous_delta, residuals, cov, Criteria, Y, X_2, fixed_effects, A, D0, Z)

#####################################################################

def _VAR_COV_Estimation(Y, X, beta, F_hat, L_hat, Variance_type, Torch_cuda):
        
        # Compute residuals
        X_2 = np.stack(X, axis=2)
        if Torch_cuda:
                X_2 = move_to_device(X_2, F_hat.device)
                Y = move_to_device(Y, F_hat.device)

                residuals = _residuals_torch(Y, X_2, beta, F_hat, L_hat)
                # Compute D0 and its inverse
                D0, inv_D0, A, M_F, Z = _est_D0_F_torch(X_2, F_hat, L_hat)
        else:

                residuals = _residuals(Y, X_2, beta, F_hat, L_hat)
                # Compute D0 and its inverse
                D0, inv_D0, A, M_F, Z = _est_D0_F(X_2, F_hat, L_hat)
        
        if Variance_type == 'iid':
                cov = _cov_iid_torch(residuals, inv_D0) if Torch_cuda else _cov_iid(residuals, inv_D0)

        elif Variance_type == 'heteroskedastic':
                T, N = Y.shape
                if Torch_cuda:
                        inv_LL = torch.linalg.inv(L_hat.T @ L_hat/N)
                        beta_adj, cov = _cov_het_torch(residuals, Z, inv_D0, beta, A, X_2, M_F, F_hat, L_hat, inv_LL)
                        beta = beta_adj
                else:
                        inv_LL = np.linalg.inv(L_hat.T @ L_hat/N) 
                        beta_adj, cov = _cov_het(residuals, Z, inv_D0, beta, A, X_2, M_F, F_hat, L_hat, inv_LL)
                beta = beta_adj
        
        return beta, cov, residuals, A, D0, inv_D0, X_2, Z

##################### Torch Variance calculations ####################

def _residuals_torch(Y, X, beta, F_hat, L_hat):
        
        T, N = Y.shape
        # Equivalent to np.einsum('itp, pj -> itj', X, beta)
        Xb = torch.einsum('itp,pj -> itj', X, beta)  # (T, N, J)
        Xb = Xb.reshape(T, N)
        E = Y - Xb - F_hat @ L_hat.T

        return E

def _est_D0_F_torch(X, F_hat, L_hat):
        T, N, p = X.shape

        # Compute M_F = I_T - F_hat @ F_hat.T / T
        I_T = torch.eye(T, device=X.device)  # Identity matrix
        M_F = I_T - (F_hat @ F_hat.T) / T  # (T, T)

        # Compute A = L_hat @ (L_hat.T @ L_hat / N)^(-1) @ L_hat.T
        LTL_inv = torch.linalg.inv((L_hat.T @ L_hat) / N)  # (R, R)
        A = L_hat @ LTL_inv @ L_hat.T  # (N, N)

        # Step 1: Apply M_F to X (broadcasting over p dimension)
        M_FX = torch.einsum('tj, jnp -> tnp', M_F, X)  # (T, N, p)

        # Step 2: Compute (X @ A) along N dimension
        M_FXA = torch.einsum('tnp, nk -> tkp', M_FX, A)  # (T, N, p)

        # Step 4: Compute final Z
        Z = M_FX - (1 / N) * M_FXA  # (T, N, p)

        # Step 5: Compute D0
        D0 = (1 / (N * T)) * torch.einsum('tnp , tnq -> pq', Z, Z)  # (p, p)

        # Step 6: Compute inverse of D0
        inv_D0 = torch.linalg.inv(D0)  # (p, p)

        return D0, inv_D0, A, M_F, Z

def _cov_iid_torch(E, inv_D0):
        T, N = E.shape
        e = E.reshape((T * N, 1))
        Sigma2 = torch.mean(e ** 2)
        cov = inv_D0*Sigma2

        return cov

# Heteroskedasticity variance-covariance matrix #
#  with bias correction and without correlation #

def _B_torch(residuals, inv_D0, A, X, F_hat, L_hat, inv_LL):
        T, N, p = X.shape
        
        V = torch.einsum('ij,tjp-> tip', A, X)/N
        sigma2_i = torch.mean(residuals**2, axis=0)
        X_V = X - V
        L_hatS = L_hat * sigma2_i[:, None]
        X_VFLL = torch.einsum('tnp, tk -> npk', X_V, F_hat @ inv_LL)/T
        DB = torch.einsum('npk, nk -> p', X_VFLL, L_hatS)
        B = -inv_D0/N @ DB

        return B.reshape(-1, 1)

def _C_torch(residuals, inv_D0, X, M_F, F_hat, L_hat, inv_LL):
        T, N, p = X.shape
        
        sigma2_t = torch.mean(residuals**2, dim=1)
        Sigma = torch.diag_embed(sigma2_t)
        MFSFLL = M_F @ Sigma @ F_hat @ inv_LL
        XMFSFLL = torch.einsum('tnp, tk -> npk', X, MFSFLL)  
        DC = torch.einsum('npk, nk -> p', XMFSFLL, L_hat)

        C = -inv_D0/(N*T) @ DC

        return C.reshape(-1, 1)

def _D3_torch(residuals, Z):
        T, N = residuals.shape

        sigmait = residuals**2
        sigmait = sigmait.unsqueeze(-1)
        weighted_Z = Z * sigmait
        D3 = torch.einsum('tnd,tne->de', weighted_Z, Z)
        D3 = D3/(N*T)
        return D3

def _cov_het_torch(residuals, Z, inv_D0, beta, A, X, M_F, F_hat, L_hat, inv_LL):
        # Compute bias correction
        T, N = residuals.shape
        B = _B_torch(residuals=residuals, inv_D0=inv_D0, A=A, X=X, F_hat=F_hat, L_hat=L_hat, inv_LL=inv_LL)
        C = _C_torch(residuals=residuals, inv_D0=inv_D0, X=X, M_F=M_F, F_hat=F_hat, L_hat=L_hat, inv_LL=inv_LL)
        beta_adj = beta - (1/N)*B - (1/T)*C

        # Compute covariance matrix
        D3 = _D3_torch(residuals, Z)
        cov = inv_D0 @ D3 @ inv_D0

        return beta_adj, cov
   
##################### Numpy Variance calculations ####################

# i.i.d. variance-covariance matrix #
def _residuals(Y, X, beta, F_hat, L_hat):
    
        T, N = Y.shape
        Xb = np.einsum('itp, pj -> itj', X, beta)
        Xb = np.reshape(Xb, (T, N))

        E = Y - Xb - F_hat @ L_hat.T

        return E

def _est_D0_F(X, F_hat, L_hat):
        T, N, p = X.shape
        M_F = np.diag(np.ones(T)) - F_hat @ F_hat.T/T
        A = L_hat @ np.linalg.inv(L_hat.T @ L_hat/N) @ L_hat.T

        # Step 1: Apply M_F to X (broadcasting over p dimension)
        M_FX = np.einsum('tj, jnp -> tnp', M_F, X)  # Shape (T, N, p)

        # Step 2: Compute (X @ A) along N dimension
        M_FXA = np.einsum('tnp, nk -> tkp', M_FX, A)  # Shape (T, N, p)

        # Step 4: Compute final Z
        Z = M_FX - (1 / N) * M_FXA

        D0 = (1 / (N * T)) * np.einsum('tnp , tnq -> pq', Z, Z)
        inv_D0 = np.linalg.inv(D0)

        return D0, inv_D0, A, M_F, Z

def _cov_iid(E, inv_D0):
        T, N = E.shape
        e = E.reshape((T * N, 1))
        Sigma2 = np.mean(e ** 2)
        cov = inv_D0*Sigma2

        return cov

# Heteroskedasticity variance-covariance matrix #
#  with bias correction and without correlation #

def _B(residuals, inv_D0, A, X, F_hat, L_hat, inv_LL):
        T, N, p = X.shape
        V = np.einsum('ij,tjp-> tip', A, X)/N
        sigma2_i = np.mean(residuals**2, axis=0)
        DB = 0
        for i in range(N):
                DB += (X[:, i, :] - V[:, i, :]).T @ F_hat / T @ inv_LL @ L_hat[i, :] * sigma2_i[i]
        B = -inv_D0/N @ DB

        return B.reshape(-1, 1)

def _C(residuals, inv_D0, X, M_F, F_hat, L_hat, inv_LL):
        T, N, p = X.shape
        sigma2_t = np.mean(residuals**2, axis=1)
        Sigma = np.diag(sigma2_t)

        DC = 0
        for i in range(N):
                DC += X[:, i, :].T @ M_F @ Sigma @ F_hat @ inv_LL @ L_hat[i, :]
        C = -inv_D0/(N*T) @ DC
        return C.reshape(-1, 1)

def _D3(residuals, Z):
        T, N = residuals.shape

        sigmati = residuals**2

        D3 = 0
        for i in range(N):
                for t in range(T):
                        D3 += np.outer(Z[t, i, :], Z[t, i, :]) * sigmati[t, i]
        D3 = D3/(N*T)
        return D3

def _cov_het(residuals, Z, inv_D0, beta, A, X, M_F, F_hat, L_hat, inv_LL):
        # Compute bias correction
        T, N = residuals.shape
        B = _B(residuals=residuals, inv_D0=inv_D0, A=A, X=X, F_hat=F_hat, L_hat=L_hat, inv_LL=inv_LL)
        C = _C(residuals=residuals, inv_D0=inv_D0, X=X, M_F=M_F, F_hat=F_hat, L_hat=L_hat, inv_LL=inv_LL)
        beta_adj = beta - (1/N)*B - (1/T)*C

        # Compute covariance matrix
        D3 = _D3(residuals, Z)
        cov = inv_D0 @ D3 @ inv_D0

        return beta_adj, cov
        