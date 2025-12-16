import torch
import numpy as np
from .class_def import Matrix, k_max_class, criteria_class
from .factor_dimensionality_torch import _PCA, _FDE
#from .factor_dimensionality_numpy import _PCA, _FDE
from .Device_aux_functions import move_to_cpu, move_to_device, get_device
from .Update_method import set_convergence_alg

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float64)

# ------------------------ #
# ---- Aux Functions ----- #
# ------------------------ #

def print_progress(beta, k, i, crit_eval, f_eval, previoues_f_eval):
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
        sys.stdout.write(', '.join(beta_print) + f", Number of Factors = {k} \n")

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
        F_hat, L_hat, _, _, _, _, _ = _PCA(W, k, restrict)
        ki = k
    else:
        _, F_hat, L_hat, ki, _ = _FDE(W, k_max = k_max, Criteria=criteria, restrict = restrict)
    
    return F_hat, L_hat, ki

def _get_estimates(beta, 
                   Y, x, xx_inv_xT,
                   k, k_max, criteria, restrict):
    T, N = Y.shape
    y = Y.reshape(-1, 1)

    w = y - x @ beta.reshape(-1, 1)
    W = w.reshape(T, N)
    F_hat, L_hat, ki = _factor_estimation(W, k, k_max, criteria, restrict)

    Y_new = Y - F_hat @ L_hat.T
    y_new = Y_new.reshape(-1, 1)
    beta_est, f_eval = _beta_estimate(y_new, x, xx_inv_xT)

    return beta_est, F_hat, L_hat, f_eval, ki

# Randomic Beta search #

def _uniform_draws(beta, scale=1, number_of_draws=3):
    
    # Ensure beta is column vector
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=torch.float64, device=get_device())
    else:
        beta = beta.float()
    
    beta = beta.reshape(-1, 1)
    p = beta.shape[0]
    device = beta.device

    is_scalar = isinstance(scale, (int, float)) or (isinstance(scale, torch.Tensor) and scale.numel() == 1)

    # Normalize scale to be broadcastable with beta
    if is_scalar:
        s = float(scale)
        h = 5 * s * torch.abs(beta) / 2.0
    else:
        # Convert list/array to tensor if needed, move to same device as beta
        if not isinstance(scale, torch.Tensor):
            s = torch.tensor(scale, dtype=beta.dtype, device=device)
        else:
            s = scale.to(device=device, dtype=beta.dtype)
        
        s = s.reshape(-1, 1)
        h = 5 * s * torch.abs(beta) / 2.0
    
   
    low_beta = beta - h
    high_beta = beta + h
    low_flat = low_beta.flatten()
    high_flat = high_beta.flatten()

    rand_0_to_1 = torch.rand((number_of_draws, p), device=device, dtype=beta.dtype)
    betas_gen = low_flat + (high_flat - low_flat) * rand_0_to_1

    return betas_gen

def _normal_draws(beta, scale=1, number_of_draws=3):
    # Ensure beta is column vector
    
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, device=get_device(), dtype=torch.float64)

    device = beta.device
    beta = beta.flatten()
    p = beta.shape[0]

    is_scalar = isinstance(scale, (int, float)) or (isinstance(scale, torch.Tensor) and scale.numel() == 1)

    if is_scalar:
        # Keep as simple float or tensor for broadcasting
        s = float(scale) if not isinstance(scale, torch.Tensor) else scale.item()
    else:
        # Convert to tensor if list/array
        if not isinstance(scale, torch.Tensor):
            s = torch.tensor(scale, device=device, dtype=beta.dtype)
        else:
            s = scale.to(device=device, dtype=beta.dtype)
    
        s = s.flatten()

    epsilon = torch.randn((number_of_draws, p), device=device, dtype=beta.dtype)
    betas_gen = beta + s * epsilon

    return betas_gen

def _random_centred_beta(beta,
                        Y, x, xx_inv_xT, 
                        k, k_max, criteria, restrict,
                        scale=1, number_of_draws=3, dist = 'uniform'):
    
    T, N = Y.shape

    if dist == 'normal':
        betas_gen = _normal_draws(beta, scale=scale, number_of_draws=number_of_draws)
    
    elif dist == 'uniform':
        betas_gen = _uniform_draws(beta, scale=scale, number_of_draws=number_of_draws)

    betas_gen = torch.vstack([beta.flatten(), betas_gen])
    min_eval = np.inf
    beta_best = beta
    F_hat_best = torch.ones((T, k_max), device=get_device())
    L_hat_best = torch.ones((T, k_max), device=get_device())
    ki_best = torch.ones((1,), device=get_device())

    for beta in betas_gen:

        beta_est, F_hat, L_hat, f_eval, ki = _get_estimates(beta, 
                                                            Y, x, xx_inv_xT,
                                                            k=k, k_max=k_max, criteria=criteria, restrict=restrict)
    
        C_NT_square = min(T, N)
        Penalty_term = np.log(C_NT_square) / C_NT_square

        f_val_adj = torch.log(f_eval/(N*T)) + ki * Penalty_term
        if f_val_adj <= min_eval:
            beta_best = beta_est
            F_hat_best = F_hat
            L_hat_best = L_hat
            min_eval = f_val_adj
            ki_best = ki
    
    return beta_best.flatten().reshape(-1,1), F_hat_best, L_hat_best, min_eval, ki_best

def _est_alg(Y: Matrix, 
            X: list[Matrix],
            k: int = None,
            criteria: criteria_class = ['IC1'],
            k_max: k_max_class = 8,
            restrict: str = 'optimize',
            max_iter: int = 10_000,
            convergence_criteria: list[str] = ['Relative_norm', 'Obj_fun', 'Grad_norm'],
            tolerance: np.ndarray = np.array([1e-8, 1e-10, 1e-5]),
            convergence_method: str = 'SOR',
            convergence_patience: int = 5,
            echo = False,
            save_path: bool = False,
            **options_convergence_method
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

    device = get_device()
    Y = move_to_device(Y, device)
    X = [move_to_device(M, device) for M in X]

    T, N = Y.shape
    p = len(X)

    up_method, number_of_draws, dist_random_draw = set_convergence_alg(p, convergence_patience, convergence_method, **options_convergence_method)
    # x = np.empty((T*N, K + 1)) # Use np.zeros if you prefer, but empty is faster
    # x[:, 0] = 1
    
    x = torch.empty((T*N, p), device=device, dtype=torch.float64)

    for i, M in enumerate(X):
        # Use reshape(-1) to flatten M into a 1D array of T*N elements
        x[:, i] = M.reshape(-1)

    #x = np.hstack([np.ones((T*N, 1))] + [M.reshape(-1, 1) for M in X])
    xx = x.T @ x
    xx_inv = torch.linalg.inv(xx)
    xx_inv_xT = xx_inv @ x.T
    y = Y.reshape(-1, 1)

    beta_initial, previous_f_eval = _beta_estimate(y, x, xx_inv_xT)
    beta, F_hat, L_hat, f_eval, ki = _random_centred_beta(beta_initial,
                                                        Y, x, xx_inv_xT, 
                                                        k, k_max, criteria, restrict,
                                                        scale=1, number_of_draws = number_of_draws, dist=dist_random_draw)
    beta = beta_initial
    change_method = False
    delta_beta = torch.abs(beta_initial - beta).flatten()
    scale = 1

    if save_path:
        path = []

    n_converges = 0
    for i in range(max_iter):
        
        beta_est, F_hat, L_hat, f_eval, ki = _random_centred_beta(beta,
                                                                Y, x, xx_inv_xT, 
                                                                k, k_max, criteria, restrict,
                                                                scale = scale, number_of_draws=number_of_draws, dist = dist_random_draw)

        crit_eval = _criteria_calculation(beta_est, beta, 
                         f_eval, previous_f_eval,
                         Y, F_hat, L_hat, x,
                         convergence_criteria)
        
        if echo:
            print_progress(beta_est, ki, i, crit_eval, f_eval/(N*T), previous_f_eval/(N*T))

        converges = all(crit_eval.cpu().numpy() <= tolerance)
        n_converges += 1 if converges else 0
        if n_converges >= convergence_patience:
            beta = beta_est
            break

        theta = torch.concatenate((beta_est.flatten(), (L_hat.mean(axis=0) * F_hat.mean(axis=0)).flatten()))
        theta = up_method.apply(theta.cpu().numpy(), ki, f_eval.cpu().numpy()/(N*T))

        if theta is None:
            if convergence_method.lower() == 'andersonacceleration':
                print("Anderson Acceleration failed to converge. Changing to SOR method.")
                convergence_method = 'SOR'
                SOR_hyperparam = 5.0
                up_method, number_of_draws, dist_random_draw = set_convergence_alg(p, convergence_method, SOR_hyperparam=SOR_hyperparam)
                change_method = True
                theta = np.concatenate((beta_initial.flatten(), (L_hat.mean(axis=0) * F_hat.mean(axis=0)).flatten()))
        
        beta_new = torch.tensor(theta[:p], device = device).reshape(-1, 1)

        delta_beta = torch.abs(beta_est - beta).flatten()
        scale = 2*torch.sqrt(delta_beta)
        beta = beta_new
        previous_f_eval = f_eval
        
        if save_path:
            path.append(beta)
        
    k = ki
    if save_path:
        np.save('path.npy', path)
    
    return beta, F_hat, L_hat, k, i+1, converges, crit_eval, change_method

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
