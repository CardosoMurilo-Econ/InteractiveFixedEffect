import numpy as np
from .class_def import Matrix, k_max_class, criteria_class
from .factor_dimensionality_numpy import _PCA, _FDE
from .Update_method import set_convergence_alg

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
    relative_objective_diff = abs(f_eval - f_eval_last)/(abs(f_eval_last)+1e-15)
    return relative_objective_diff.flatten()[0]

def _relative_norm(beta_new, beta):
    relative_norm = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta)+1e-15)
    return relative_norm.flatten()[0]

def _grad_norm(Y, F, L, x, beta):

    Y_new = Y - F @ L.T
    y_new = Y_new.reshape(-1, 1)

    GRAD = 2*x.T @ (y_new - x @ beta)

    return np.linalg.norm(GRAD)

def _criteria_calculation(beta_new, beta, 
             f_eval, f_eval_last,
             Y, F_hat, L_hat, x,
             convergence_criteria = ['Relative_norm', 'Obj_fun', 'Grad_norm']):
    
    crit_eval = []
    if 'Relative_norm' in convergence_criteria:
        crit_eval.append(_relative_norm(beta_new, beta))

    if 'Obj_fun' in convergence_criteria:
        crit_eval.append(_Obj_fun_diff(f_eval, f_eval_last))

    if 'Grad_norm' in convergence_criteria:
        crit_eval.append(_grad_norm(Y, F_hat, L_hat, x, beta_new))
    
    return crit_eval

# ------------------------ #
# ----  IFE Function  ---- #
# ------------------------ #

def _beta_estimate(y_new, x, xx_inv_xt):

    beta = xx_inv_xt @ y_new
    e = y_new - x @ beta
    f_eval = np.sum(e**2)
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
    beta = beta.reshape(-1, 1)

    p = beta.shape[0]

    # Normalize scale to be broadcastable with beta
    if np.isscalar(scale) or (isinstance(scale, np.ndarray) and scale.size == 1):
        s = float(scale)
        h = 5 * s * np.abs(beta) / 2.0
    else:
        s = np.asarray(scale).reshape(-1, 1)
        if s.shape[0] != beta.shape[0]:
            raise ValueError("scale must be a scalar or have length equal to p")
        h = 5 * s * np.abs(beta) / 2.0

    low_beta = beta - h
    high_beta = beta + h

    betas_gen = np.random.uniform(low=low_beta.flatten(), 
                    high=high_beta.flatten(), 
                    size=(number_of_draws, p))

    return betas_gen

def _normal_draws(beta, scale=1, number_of_draws=3):
    # Ensure beta is column vector
    
    p = beta.shape[0]
    beta = beta.flatten()

    if not np.isscalar(scale) or not (isinstance(scale, int)):
        scale = scale.flatten()

    betas_gen = np.random.normal(loc=beta, scale = scale, size = (number_of_draws, p))

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

    betas_gen = np.vstack([beta.flatten(), betas_gen])
    min_eval = np.inf

    for beta in betas_gen:

        beta_est, F_hat, L_hat, f_eval, ki = _get_estimates(beta, 
                                                            Y, x, xx_inv_xT,
                                                            k=k, k_max=k_max, criteria=criteria, restrict=restrict)
    
        C_NT_square = min(T, N)
        Penalty_term = np.log(C_NT_square) / C_NT_square

        f_val_adj = np.log(f_eval/(N*T)) + ki * Penalty_term
        if f_val_adj < min_eval:
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
            restrict: str = 'common',
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
                - change_method (bool): Whether the convergence method was changed during execution.
        '''

    T, N = Y.shape
    p = len(X)

    up_method, number_of_draws, dist_random_draw = set_convergence_alg(p, convergence_patience, convergence_method, **options_convergence_method)
    # x = np.empty((T*N, K + 1)) # Use np.zeros if you prefer, but empty is faster
    # x[:, 0] = 1
    
    x = np.empty((T*N, p))

    for i, M in enumerate(X):
        # Use reshape(-1) to flatten M into a 1D array of T*N elements
        x[:, i] = M.reshape(-1)

    #x = np.hstack([np.ones((T*N, 1))] + [M.reshape(-1, 1) for M in X])
    xx = x.T @ x
    xx_inv = np.linalg.inv(xx)
    xx_inv_xT = xx_inv @ x.T
    y = Y.reshape(-1, 1)

    beta_initial, previous_f_eval = _beta_estimate(y, x, xx_inv_xT)
    beta, F_hat, L_hat, f_eval, ki = _random_centred_beta(beta_initial,
                                                        Y, x, xx_inv_xT, 
                                                        k, k_max, criteria, restrict,
                                                        scale=1, number_of_draws = number_of_draws, dist=dist_random_draw)
    beta = beta_initial
    change_method = False
    delta_beta = np.abs(beta_initial - beta).flatten()

    if save_path:
        path = []
        f_eval_history = []

    n_converges = 0
    for i in range(max_iter):
        
        beta_est, F_hat, L_hat, f_eval, ki = _random_centred_beta(beta,
                                                                Y, x, xx_inv_xT, 
                                                                k, k_max, criteria, restrict,
                                                                scale = delta_beta, number_of_draws=number_of_draws, dist = dist_random_draw)

        crit_eval = _criteria_calculation(beta_est, beta, 
                         f_eval, previous_f_eval,
                         Y, F_hat, L_hat, x,
                         convergence_criteria)
        
        if echo:
            print_progress(beta_est, ki, i, crit_eval, f_eval/(N*T), previous_f_eval/(N*T))

        converges = all(np.array(crit_eval) <= tolerance)
        n_converges += 1 if converges else min(0, -n_converges)
        if n_converges >=  convergence_patience:
            beta = beta_est
            break

        theta = np.concatenate((beta_est.flatten(), (L_hat.mean(axis=0) * F_hat.mean(axis=0)).flatten()))
        theta = up_method.apply(theta, ki, f_eval/(N*T))

        if theta is None:
            if convergence_method.lower() == 'andersonacceleration':
                print("Anderson Acceleration failed to converge. Changing to SOR method.")
                convergence_method = 'SOR'
                SOR_hyperparam = 5.0
                up_method, number_of_draws, dist_random_draw = set_convergence_alg(p, convergence_method, SOR_hyperparam=SOR_hyperparam)
                change_method = True
                theta = np.concatenate((beta_initial.flatten(), (L_hat.mean(axis=0) * F_hat.mean(axis=0)).flatten()))
        
        beta_new = theta[:p].reshape(-1, 1)

        delta_beta = np.abs(beta_est - beta).flatten()
        beta = beta_new
        previous_f_eval = f_eval
        
        if save_path:
            path.append(beta)
            f_eval_history.append(f_eval/(N*T))
        
    k = ki
    if save_path:
        np.save('path.npy', path)
        np.save('f_eval.npy', f_eval_history)
    
    return beta, F_hat, L_hat, k, i+1, converges, crit_eval, change_method

# ------------------------ #
# ----  IFE Variance  ---- #
# ----   Calculation  ---- #
# ------------------------ #

def _VAR_COV_Estimation(Y, X, beta, F_hat, L_hat, variance_type, fixed_effects):
        
    T, N = Y.shape
    k = F_hat.shape[1]
    
    # Compute residuals
    X_2 = np.stack(X, axis=2)
    p = X_2.shape[2]
    residuals = _residuals(Y, X_2, beta, F_hat, L_hat)
    # Compute degrees of freedom
    df = _degrees_of_freedom(N, T, k, p, fixed_effects)

    # Compute D0 and its inverse
    D0, inv_D0, A, M_F, Z = _est_D0_F(X_2, F_hat, L_hat, df)
    
    if variance_type == 'iid':
        cov = _cov_iid(residuals, inv_D0, df)

    elif variance_type == 'heteroskedastic':
        
        inv_LL = np.linalg.inv(L_hat.T @ L_hat/N) 
        beta_adj, cov = _cov_het(residuals, Z, inv_D0, beta, A, X_2, M_F, F_hat, L_hat, inv_LL, df)
        beta = beta_adj
    
    return beta, cov, residuals, A, D0, inv_D0, X_2, Z, df

# i.i.d. variance-covariance matrix #
def _residuals(Y, X, beta, F_hat, L_hat):
    T, N = Y.shape
    #X_2 = np.stack(X, axis=2)
    p = X.shape[2]
    # contract over regressor dimension p with coefficient vector beta -> shape (T, N)
    Xbeta = np.einsum('itp,p->it', X, beta.flatten())
    Xbeta = Xbeta.reshape(T, N)
    E = Y - Xbeta - F_hat @ L_hat.T
    return E

def _degrees_of_freedom(N, T, k, p, fixed_effects):    
    w = k + 1 if fixed_effects == 'twoways' else k
    df = N*T - p - w*(N + T) 
    
    return df

def _est_D0_F_alt(X, F_hat, L_hat):
    # Infer dimensions dynamically from input to ensure robustness
    T, N, p = X.shape
    k = F_hat.shape[1]

    #calculate projection matrices
    A = L_hat @ np.linalg.inv(L_hat.T @ L_hat/N) @ L_hat.T
    M_F = np.identity(T) - F_hat @ F_hat.T/T

    # 1. Precompute the linear transformation for all N
    # Replaces the loop: M_FX_i = M_F @ X_i
    # M_F is (T, T), X is (T, N, p). 
    # Contracting axis 1 of M_F with axis 0 of X results in (T, N, p).
    M_FX = np.tensordot(M_F, X, axes=(1, 0))

    # 2. Precompute the sum over tau
    # Replaces the inner loop over 'tau'. Result shape: (T, p)
    sum_M_FXk = np.sum(M_FX, axis=1)

    # 3. Vectorize the subtraction and scaling
    # scaling_factors shape: (N,) derived from A[:, k]
    M_FXA = np.einsum('tnp, nj -> tjp', M_FX, A) # Shape (T, N, p)

    # Use broadcasting to compute Z for all i at once.
    # X_transformed: (T, N, p)
    # sum_transformed: (T, 1, p) -> Broadcasts over N
    # scaling_factors: (1, N, 1) -> Broadcasts over T and p
    Z = M_FX - (1 / N) * M_FXA

    # 4. Compute D using a single matrix multiplication
    # Reshape Z to (T*N, p) to perform the equivalent of sum(Z_i.T @ Z_i)
    Z_flat = Z.reshape(-1, p)
    sum_zz = Z_flat.T @ Z_flat

    # 5. Final Normalization
    D0 = sum_zz / (N*T) #((N-k)*(T-k) - p)
    
    inv_D0 = np.linalg.inv(D0)

    return D0, inv_D0, A, M_F, Z

def _est_D0_F(X, F_hat, L_hat, df):
    T = F_hat.shape[0]
    N = L_hat.shape[0]
    p = X.shape[2]
    k = F_hat.shape[1]

    M_F = np.identity(T) - F_hat @ F_hat.T/T
    A = L_hat @ np.linalg.inv(L_hat.T @ L_hat/N) @ L_hat.T
    # Step 1: Apply M_F to X (broadcasting over p dimension)
    M_FX = np.einsum('tj, jnp -> tnp', M_F, X)  # Shape (T, N, p)

    # Step 2: Compute (X @ A) along N dimension
    M_FXA = np.einsum('tnp, nk -> tkp', M_FX, A)  # Shape (T, N, p)

    # Step 4: Compute final Z
    Z = M_FX - (1 / N) * M_FXA

    # Step 5: calculate the correct degree of freedom
    D0 = np.einsum('tnp , tnq -> pq', Z, Z) / df 
    inv_D0 = np.linalg.inv(D0)

    return D0, inv_D0, A, M_F, Z

def _cov_iid(E, inv_D0, df):

    Sigma2 = np.sum(E**2)/df
    cov = inv_D0*Sigma2

    return cov

# Heteroskedasticity variance-covariance matrix #
#  with bias correction and without correlation #

def _B(residuals, inv_D0, A, X, F_hat, L_hat, inv_LL):
        
    T, N, p = X.shape
    V = np.einsum('ij,tjp-> tip', A, X)/N
    sigma2_i = np.mean(residuals**2, axis=0) # Shape (N,)

    # 1. Calculate the difference matrix
    W = X - V  # Shape (T, N, p)

    # 2. Vectorized loop using einsum
    # This one line replaces the entire for loop
    DB = np.einsum('tip, tk, kl, il, i -> p',
                W, F_hat, inv_LL, L_hat, sigma2_i) / T

    # 3. Final calculation remains the same
    B = -inv_D0/N @ DB

    return B.reshape(-1, 1)

def _C(residuals, inv_D0, X, M_F, F_hat, L_hat, inv_LL):
    T, N, p = X.shape
    sigma2_t = np.mean(residuals**2, axis=1) # Shape (T,)
    Sigma = np.diag(sigma2_t)

    K = M_F @ Sigma @ F_hat @ inv_LL  # Shape (T, r)

    DC = np.einsum('tip, tk, ik -> p', X, K, L_hat)
    C = -inv_D0/(N*T) @ DC
        
    return C.reshape(-1, 1)

def _D3(residuals, Z, df):
    T, N = residuals.shape

    sigmati = residuals**2
    D3 = np.einsum('tip, tiq, ti -> pq', Z, Z, sigmati)/df

    return D3

def _cov_het(residuals, Z, inv_D0, beta, A, X, M_F, F_hat, L_hat, inv_LL, df):
    # Compute bias correction
    T, N = residuals.shape
    k = F_hat.shape[1]
    p = X.shape[2]
    B = _B(residuals=residuals, inv_D0=inv_D0, A=A, X=X, F_hat=F_hat, L_hat=L_hat, inv_LL=inv_LL)
    C = _C(residuals=residuals, inv_D0=inv_D0, X=X, M_F=M_F, F_hat=F_hat, L_hat=L_hat, inv_LL=inv_LL)
    beta_adj = beta - (1/N)*B - (1/T)*C

    # Compute covariance matrix
    D3 = _D3(residuals, Z, df)
    cov = inv_D0 @ D3 @ inv_D0

    return beta_adj, cov
