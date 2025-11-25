import torch
import numpy as np
from .class_def import Matrix, k_max_class, criteria_class
#from .factor_dimensionality_torch import _PCA, _FDE
from .factor_dimensionality_numpy import _PCA, _FDE
from .Device_aux_functions import move_to_cpu, move_to_device, get_device

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float64)

# ------------------------ #
# ---- Aux Functions ----- #
# ------------------------ #

def print_progress(beta, k, i, crit_eval, f_eval, previoues_f_eval, SOR_hyperparam):
        import sys
        if i == 0:
                sys.stdout.write("\n")
                sys.stdout.write("\n")
                sys.stdout.write("\n")
                sys.stdout.write("\n")
        
        sys.stdout.write("\033[F" * 4)  # Move up 4 lines to overwrite previous values
        sys.stdout.write(f"Iteration {i+1}:\n") 

        # Print beta values 
        if len(beta.flatten()) <= 5:
            beta_print = [f"Beta[{j}] = {b:.6f}" for j, b in enumerate(beta.flatten())]
        else:
            beta_print = [f"Beta[{j}] = {b:.6f}" for j, b in enumerate(beta.flatten()[:4])]
            beta_print.append("...")  # Indicate that there are more coefficients not shown
            beta_print.append(f" | Avg Beta: {np.mean(beta):.6f}")
        sys.stdout.write(', '.join(beta_print) + f", Number of Factors = {k}, SOR_parameter = {SOR_hyperparam:.2f} \n")

        # Print criteria evaluations 
        num_criteria = len(crit_eval)
        names = ['Relative Norm: ', 'Objective Function Diff: ', 'Gradient Norm: '][:num_criteria]
        delta = (f_eval - previoues_f_eval)
        sys.stdout.write(f"Obj Function Value = {f_eval:.5f}, Delta Obj Function = {delta:.12} \n")  


        result_string = ", ".join([f"{name}{eval:.12f}" for name, eval in zip(names, crit_eval)])
        sys.stdout.write(result_string + "\n")
        sys.stdout.flush()

def _demean(Y, X):
        Y_demeaned = Y - np.mean(Y)
        X_demeaned = [M - np.mean(M, axis=0) for M in X]
        return Y_demeaned, X_demeaned

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

def _get_fixed_effects(fixed_effects, 
                       beta, 
                       Y=None, X=None, F_hat=None, L_hat=None,
                       Y_dobledot = None, X_dobledot = None, Y_idot = None, X_idot = None, Y_tdot = None, X_tdot = None 
                       ):
      
    if fixed_effects == 'none':
        return None, None, None
    
    if fixed_effects == 'demeaned':
        W = Y-F_hat @ L_hat.T
        for i in range(len(X)): 
            W -= beta[i] * X[i]
        mu = np.mean(W)
        return mu, None, None 
    
    if fixed_effects == 'twoways':
        mu = Y_dobledot - X_dobledot.T @ beta
        alpha_i = Y_idot.T - X_idot.T @ beta - mu
        gamma_t = Y_tdot - X_tdot @ beta - mu

        return mu, alpha_i, gamma_t

# ------------------------ #
# ----  Convergence  ----- #
# ----    Criterio   ----- #
# ------------------------ #

def _Obj_fun_diff(f_eval, f_eval_last):
    relative_objective_diff = abs(f_eval - f_eval_last)/abs(f_eval_last)
    return relative_objective_diff.flatten()[0]

def _relative_norm(beta_new, beta):
    relative_norm = torch.linalg.norm(beta_new - beta) / torch.linalg.norm(beta)
    return relative_norm.flatten()[0]

def _grad_norm(Y, F, L, x, beta):

    T,N = Y.shape
    Y_new = Y - F @ L.T
    y_new = Y_new.reshape(-1, 1)

    GRAD = 2* x.T @ (y_new - x @ beta)

    return torch.linalg.norm(GRAD)

def _criteria_calculation(beta_new, beta, 
             f_eval, f_eval_last,
             Y, F_hat, L_hat, x,
             convergence_criteria = ['Relative_norm', 'Obj_fun', 'Grad_norm']):
    
    crit_eval = torch.tensor([], device=get_device())
    if 'Relative_norm' in convergence_criteria:
        crit_i = _relative_norm(beta_new, beta)
        crit_eval = torch.cat([crit_eval, crit_i.view(1)])

    if 'Obj_fun' in convergence_criteria:
        crit_i = _Obj_fun_diff(f_eval, f_eval_last)
        crit_eval = torch.cat([crit_eval, crit_i.view(1)])

    if 'Grad_norm' in convergence_criteria:
        crit_i = _grad_norm(Y, F_hat, L_hat, x, beta_new)
        crit_eval = torch.cat([crit_eval, crit_i.view(1)])
    
    return crit_eval

# ------------------------ #
# ----  IFE Function  ---- #
# ------------------------ #

def _beta_estimate(y_new, x, xx_inv_xt):

    beta = xx_inv_xt @ y_new
    e = y_new - x @ beta
    f_eval = torch.sum(e**2)
    return beta, f_eval

def _factor_estimation(W, k, k_max, criteria, restrict):
    
    if k is not None:
        F_hat, L_hat, _, _, _, _, _ = _PCA(W.cpu().numpy(), k, restrict.cpu()) 
    else:
        _, F_hat, L_hat, ki, _ = _FDE(W.cpu().numpy(), k_max = k_max, Criteria=criteria, restrict = restrict)
    
    F_hat = move_to_device(F_hat, get_device())
    L_hat = move_to_device(L_hat, get_device())
    
    return F_hat, L_hat, ki

class update_method:
    def __init__(self, K, w: float = 1.2, w_max: float = None,
                 inc_factor: float = 1.1, dec_factor: float = 0.5):
        self.w = w
        self.w_max = w if w_max is None else w_max
        self.inc_factor = inc_factor
        self.dec_factor = dec_factor
        # Use a scalar check to avoid issues with vector length
        self.delta_last = torch.zeros(K, dtype=torch.float64, device=get_device())

    def update(self, beta, beta_est):
        delta = beta_est - beta

        # Check the product of the last update and the current one
        # This assumes 'beta' is a 1D numpy array
        current_delta_product = torch.dot(self.delta_last.ravel(), delta.ravel())

        if current_delta_product < 1e-8:
            # Oscillation: decrease w
            self.w = max(self.w * self.dec_factor, 1.0)
        else:
            # Stable: increase w
            self.w = min(self.w * self.inc_factor, self.w_max)

        beta_new = beta + self.w * delta
        
        # Store the last update vector for the next check
        self.delta_last = delta 
        return beta_new
    
def _est_alg(Y: Matrix, 
            X: list[Matrix],
            k: int = None,
            criteria: criteria_class = ['IC1'],
            k_max: k_max_class = 8,
            restrict: str = 'optimize',
            max_iter: int = 10_000,
            convergence_criteria: list[str] = ['Relative_norm', 'Obj_fun', 'Grad_norm'],
            tolerance: np.ndarray = np.array([1e-6, 1e-12, 1e-5]),
            SOR_hyperparam: float = 1.0,
            max_SOR_hyperparam: float = None,
            inc_factor: float = 1.1, 
            dec_factor: float = 0.5,
            echo = False,
            save_path: bool = False
        ):
    
    '''
        Internal function for computing the coefficients, factors structure and its dimensionality in a Large-panel model with interactive fixed effects. The algorithm is based on the paper **"PANEL DATA MODELS WITH INTERACTIVE FIXED EFFECTS"** by Bai (2009): https://doi.org/10.3982/ECTA6135.
        This function is intended for use inside other functions.  
        For direct usage, please refer to the `IFE` function.

        The algorithm use SOR (Successive Over-Relaxation) method to update the coefficients estimates at each iteration. The SOR hyperparameter ($\omega$) can be adjusted through the `SOR_hyperparam` argument:
                - SOR_hyperparam = 1.0: No over-relaxation, equivalent to standard Gauss-Seidel update.
                - SOR_hyperparam > 1.0: Over-relaxation, which may accelerate convergence.
                - 0 < SOR_hyperparam < 1.0: Under-relaxation, which may improve stability in some cases.

                $\beta_{new} = \beta_{old} + \omega (\beta_{est} - \beta_{old})$

        Inputs: 
                See `IFE` function.
        
        Returns: (tuple) with the following elements:
                - beta (numpy.ndarray): Coefficients of the model.
                - F_hat (numpy.ndarray): Factors structure.
                - L_hat (numpy.ndarray): Loadings structure.
                - k (int): Number of factors.
                - n (int): Number of iterations until convergence.
                - converges (bool): Whether the algorithm converged.
                - crit_eval (list): Final evaluation of the convergence criteria.
        '''

    tolerance = move_to_device(tolerance, get_device())

    T, N = Y.shape
    K = len(X)

    up_method = update_method(K = K, 
                              w=SOR_hyperparam, w_max=max_SOR_hyperparam,
                              inc_factor=inc_factor, dec_factor=dec_factor)

    # x = np.empty((T*N, K + 1)) # Use np.zeros if you prefer, but empty is faster
    # x[:, 0] = 1
    
    Y = move_to_device(Y, get_device())
    x = torch.empty((T*N, K), device=get_device())

    for i, M in enumerate(X):
        # Use reshape(-1) to flatten M into a 1D array of T*N elements
        x[:, i] = move_to_device(M.reshape(-1), get_device())

    #x = np.hstack([np.ones((T*N, 1))] + [M.reshape(-1, 1) for M in X])
    xx = x.T @ x
    xx_inv = torch.linalg.inv(xx)
    xx_inv_xT = xx_inv @ x.T

    y = Y.reshape(-1, 1)
    beta_est, previous_f_eval = _beta_estimate(y, x, xx_inv_xT)
    beta = beta_est

    if save_path:
        path = []

    for i in range(max_iter):
        
        w = y - x @ beta
        W = w.reshape(T, N)
        F_hat, L_hat, ki = _factor_estimation(W, k, k_max, criteria, restrict)

        Y_new = Y - F_hat @ L_hat.T
        y_new = Y_new.reshape(-1, 1)
        beta_est, f_eval = _beta_estimate(y_new, x, xx_inv_xT)

        crit_eval = _criteria_calculation(beta_est, beta, 
                         f_eval, previous_f_eval,
                         Y, F_hat, L_hat, x,
                         convergence_criteria)

        converges = all(crit_eval <= tolerance)
        if converges:
            beta = beta_est
            break
        
        beta = up_method.update(beta, beta_est)

        if echo:
            print_progress(beta, ki, i, crit_eval, f_eval/(N*T), previous_f_eval/(N*T), up_method.w)

        previous_f_eval = f_eval

        if save_path:
            path.append(beta)
        
    k = ki
    if save_path:
        np.save('path.npy', path)
    
    return beta, F_hat, L_hat, k, i+1, converges, crit_eval

# ------------------------ #
# ----  IFE Variance  ---- #
# ----   Calculation  ---- #
# ------------------------ #

def _VAR_COV_Estimation(Y, X, beta, F_hat, L_hat, variance_type, fixed_effects):
        
    # Compute residuals
    X = [move_to_device(M, get_device()) for M in X]
    Y = move_to_device(Y, get_device())
    X_2 = torch.stack(X, axis=2)
    residuals = _residuals(Y, X_2, beta, F_hat, L_hat)

    T, N = Y.shape
    k = F_hat.shape[1]
    p = X_2.shape[2]

    df = _degrees_of_freedom(N, T, k, p, fixed_effects)

    # Compute D0 and its inverse
    D0, inv_D0, A, M_F, Z = _est_D0_F(X_2, F_hat, L_hat, df)
    
    if variance_type == 'iid':
        p = X_2.shape[2]
        k = F_hat.shape[1]
        cov = _cov_iid(residuals, inv_D0, df)

    elif variance_type == 'heteroskedastic':
        T, N = Y.shape
        inv_LL = torch.linalg.inv(L_hat.T @ L_hat/N) 
        beta_adj, cov = _cov_het(residuals, Z, inv_D0, beta, A, X_2, M_F, F_hat, L_hat, inv_LL, df)
        beta = beta_adj
    
    return beta, cov, residuals, A, D0, inv_D0, X_2, Z, df

# i.i.d. variance-covariance matrix #
def _residuals(Y, X, beta, F_hat, L_hat):
    
        T, N = Y.shape
        Xb = torch.einsum('itp, pj -> itj', X, beta)
        Xb = torch.reshape(Xb, (T, N))

        E = Y - Xb - F_hat @ L_hat.T

        return E

def _degrees_of_freedom(N, T, k, p, fixed_effects):    
    w = k + 1 if fixed_effects == 'twoways' else k
    df = N*T - p - w*(N + T)
    
    return df

def _est_D0_F(X, F_hat, L_hat, df):
        T, N, p = X.shape
        M_F = torch.diag(torch.ones(T, device=get_device())) - F_hat @ torch.linalg.inv(F_hat.T @ F_hat) @ F_hat.T
        A = L_hat @ torch.linalg.inv(L_hat.T @ L_hat/N) @ L_hat.T

        # Step 1: Apply M_F to X (broadcasting over p dimension)
        M_FX = torch.einsum('tj, jnp -> tnp', M_F, X)  # Shape (T, N, p)

        # Step 2: Compute (X @ A) along N dimension
        M_FXA = torch.einsum('tnp, nk -> tkp', M_FX, A)  # Shape (T, N, p)

        # Step 4: Compute final Z
        Z = M_FX - (1 / N) * M_FXA

        D0 = torch.einsum('tnp , tnq -> pq', Z, Z) / df
        inv_D0 = torch.linalg.inv(D0)

        return D0, inv_D0, A, M_F, Z

def _cov_iid(E, inv_D0, df):
        T, N = E.shape
        e = E.reshape((T * N, 1))
        Sigma2 = torch.sum(e ** 2) / df
        cov = inv_D0*Sigma2

        return cov

# Heteroskedasticity variance-covariance matrix #
#  with bias correction and without correlation #

def _B(residuals, inv_D0, A, X, F_hat, L_hat, inv_LL):
        T, N, p = X.shape
        V = torch.einsum('ij,tjp-> tip', A, X)/N
        sigma2_i = torch.mean(residuals**2, axis=0)

        Diff = X - V

        FLL_inv = F_hat @ inv_LL / T
        FLL_invL = FLL_inv @ L_hat.T
        FLL_invL_sigma = FLL_invL * sigma2_i
        
        DB = torch.einsum('tnp,tn->p', Diff, FLL_invL_sigma)
        B = -inv_D0/N @ DB

        return B.reshape(-1, 1)

def _C(residuals, inv_D0, X, M_F, F_hat, L_hat, inv_LL):
    T, N, p = X.shape
    sigma2_t = torch.mean(residuals**2, axis=1)
    sigma_F = sigma2_t.unsqueeze(1) * F_hat

    Msigma_FLL = M_F @ sigma_F @ inv_LL

    Msigma_FLL_L = Msigma_FLL @ L_hat.T

    DC = torch.einsum('tnp,tn->p', X, Msigma_FLL_L)

    C = -(inv_D0/(N*T)) @ DC
    return C.reshape(-1, 1)

def _D3(residuals, Z, df):
    T, N = residuals.shape

    residuals_expanded = residuals[..., np.newaxis]
    Z_weighted = Z * residuals_expanded

    Z_flat = Z_weighted.reshape(-1, Z.shape[2])
    D3 = Z_flat.T @ Z_flat

    D3 = D3 / df

    return D3

def _cov_het(residuals, Z, inv_D0, beta, A, X, M_F, F_hat, L_hat, inv_LL, df):
        # Compute bias correction
        T, N = residuals.shape
        B = _B(residuals=residuals, inv_D0=inv_D0, A=A, X=X, F_hat=F_hat, L_hat=L_hat, inv_LL=inv_LL)
        C = _C(residuals=residuals, inv_D0=inv_D0, X=X, M_F=M_F, F_hat=F_hat, L_hat=L_hat, inv_LL=inv_LL)
        beta_adj = beta - (1/N)*B - (1/T)*C

        # Compute covariance matrix
        D3 = _D3(residuals, Z, df)

        cov = inv_D0 @ D3 @ inv_D0

        # Adjust for degrees of freedom in finite samples
        p = X.shape[2]
        k = F_hat.shape[1]
        dg = N*T - p - k*(N + T - k)
        cov = cov * (N*T) / dg

        return beta_adj, cov
