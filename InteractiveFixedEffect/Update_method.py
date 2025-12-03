import numpy as np
import numpy.linalg as LA


class SOR_update_method:
    def __init__(self, p, w: float = 1.2, w_max: float = None,
                 inc_factor: float = 1.1, dec_factor: float = 0.5):
        self.w = w
        self.w_max = w_max if w_max is not None else w
        self.inc_factor = inc_factor
        self.dec_factor = dec_factor
        self.beta_old = None
        self.p = p
        # Use a scalar check to avoid issues with vector length
        self.delta_last = np.array([0]*p)

    def apply(self, beta_new, ki=None, f_val=None):

        if beta_new.flatten().shape[0] != self.p:
            beta_new = beta_new[:self.p].reshape(-1, 1)

        if self.beta_old is None:
            self.beta_old = beta_new
            return beta_new

        delta = beta_new - self.beta_old

        # Check the product of the last update and the current one
        # This assumes 'beta' is a 1D numpy array
        current_delta_product = np.dot(self.delta_last.ravel(), delta.ravel())

        if current_delta_product < 1e-8:
            # Oscillation: decrease w
            self.w = max(self.w * self.dec_factor, 1.0)
        else:
            # Stable: increase w
            self.w = min(self.w * self.inc_factor, self.w_max)

        beta_new = self.beta_old + self.w * delta
        
        # Store the last update vector for the next check
        self.delta_last = delta 
        self.beta_old = beta_new
        return beta_new

# ------------------------ #

class AndersonAcceleration:
    """Anderson acceleration for fixed-point iteration

    (Regularized) Anderson acceleration algorithm, also known as Approximate
    Maximal Polynomial Extrapolation (AMPE). The goal is to find a fixed point
    to some Lipschitz continuous function `g`, that is, find an `x` such that 
    `g(x) = x`. AMPE uses some previous iterates and residuals to solve for 
    coefficients, and then use them to extrapolate to the next iterate. The
    parameters used in this implementation are in [1].

    Parameters
    ----------
    window_size : int (optional, default=5)
        The number of previous iterates to use in the extrapolation. This is
        `m` in the algorithm.

    reg : float (optional, default=0)
        The L2 regularization parameter. This is `lambda` in the algorithm.

    mixing_param : float (optional, default=1)
        The mixing parameter. Must be between 0 and 1. This is `beta` in the
        algorithm.

    Attributes
    ----------
    x_hist_ : list
        History of the previous accelerated iterates. 
        Maximum size = `window_size` + 1.

    gx_hist_ : list
        History of the previous function applications. These are 
        pre-accelerated iterates.
        Maximum size = `window_size` + 1.

    residuals_hist_ : list
        History of the previous residuals.
        Maximum size = `window_size` + 1.

    param_shape_ : tuple
        Shape of the parameters, defined when the first iterate is applied.

    References
    ----------
    [1] T. D. Nguyen, A. R. Balef, C. T. Dinh, N. H. Tran, D. T. Ngo, 
        T. A. Le, and P. L. Vo. "Accelerating federated edge learning," 
        in IEEE Communications Letters, 25(10):3282–3286, 2021.
    [2] D. Scieur, A. d’Aspremont, and F. Bach, “Regularized nonlinear 
        acceleration,” in Advances in Neural Information Processing Systems,
        2016.
    [3] A. d’Aspremont, D. Scieur, and A. Taylor, “Acceleration methods,”
        arXiv preprint arXiv:2101.09545, 2021.
    [4] "Anderson Acceleration," Stanford University Convex Optimization Group.
        https://github.com/cvxgrp/aa

    Examples
    --------
    >>> # x is a d-dimensional numpy array, produced by applying
    >>> # the function g to the previous iterate x_{t-1}
    >>> acc = AndersonAccelerationModule(reg=1e-8)
    >>> x_acc = acc.apply(x)  # accelerate from x 
    """

    def __init__(self, window_size=5, reg=0, mixing_param=1.0):
        
        window_size = int(window_size)
        assert window_size > 0, "Window size must be positive"
        self.window_size = int(window_size)

        assert reg >= 0, "Regularization parameter must be non-negative"
        self.reg = reg

        assert 0 <= mixing_param <= 1, "Mixing parameter must be between 0 and 1"
        self.mixing_param = mixing_param

        # History of function applications
        self.gx_hist_ = []

        # History of iterates
        self.x_hist_ = []

        # History of residuals
        self.residuals_hist_ = []

        # Shape of the parameters, defined when the first iterate is applied
        self.param_shape_ = None

        # Previous residual norm for safeguard detection
        self.prev_res_norm = np.inf    

        # Decrease_obj_test up
        self.decrease_obj = decrease_obj_test(kth_max=3, max_history=10, tolerance=1e-5)

        # previous ki
        self.previous_ki = None

    def apply_int(self, x):
        """Perform acceleration on an iterate.

        Parameters
        ----------
        x : numpy array
            The iterate to accelerate. This is the application of `g` to the
            previous iterate.

        Returns
        -------
        x_acc : numpy array
            The accelerated iterate of the same shape as `x`.
        """

        if len(self.x_hist_) <= 0:
            # First iteration, so no acceleration can be done
            self.x_hist_.append(x)
            self.param_shape_ = x.shape
            return x

        # Check the shape of the iterate
        assert x.shape == self.param_shape_, \
            "Iterate shape must be the same as the previous iterate"

        x_prev = self.x_hist_[-1]

        residual = x - x_prev
        self.residuals_hist_.append(residual)
        if len(self.residuals_hist_) > self.window_size + 1:
            self.residuals_hist_.pop(0)

        self.gx_hist_.append(x)
        if len(self.gx_hist_) > self.window_size + 1:
            self.gx_hist_.pop(0)

        # Solve for alpha: min ||alpha_i F_{t-i}||
        Ft = np.stack(self.residuals_hist_)  # shape = (m_t + 1, dim)
        RR = Ft @ Ft.T
        RR += self.reg * np.eye(RR.shape[0])
        try:
            RR_inv = LA.inv(RR)
            alpha = np.sum(RR_inv, 1)
        except LA.LinAlgError:
            #print("Warning: Singular matrix encountered in Anderson Acceleration.")
            # Singular matrix, so solve least squares instead
            alpha = LA.lstsq(RR, np.ones(Ft.shape[0]), -1)[0]
            
        # Normalize alpha
        alpha /= alpha.sum() + 1e-16

        # Extrapolate to get accelerated iterate
        if len(self.x_hist_) <= 0:
            x_acc = x
        else:
            x_acc = 0
            for alpha_i, x_i, Gx_i in zip(alpha, self.x_hist_, self.gx_hist_):
                x_acc += (1 - self.mixing_param) * alpha_i * x_i
                x_acc += self.mixing_param * alpha_i * Gx_i

        self.x_hist_.append(x_acc)
        if len(self.x_hist_) > self.window_size + 1:
            self.x_hist_.pop(0)

        return x_acc

    def reset(self):
        """Empty the histories of iterates and residuals.
        """
        self.x_hist_ = []
        self.gx_hist_ = []
        self.residuals_hist_ = []
        self.prev_res_norm = np.inf
        self.param_shape_ = None

    def apply(self, x, ki, f_val):
        
        if ki != self.previous_ki:
            self.reset()

        decreasing = self.decrease_obj.apply(f_val, x)

        if not decreasing:
            return None
        
        x_acc = self.apply_int(x)
        self.previous_ki = ki

        return x_acc

class decrease_obj_test():
    def __init__(self, kth_max = 3, max_history=10, tolerance = 1e-5):
        
        self.max_history = max_history
        self.kth_max = kth_max
        self.tolerance = tolerance
        self.reset()

    def reset(self):
        self.f_eval_hist = []
        self.beta_hist = []

    def apply(self, f_eval, beta):

        if not len(self.f_eval_hist) < self.kth_max:

            f_eval_prev = self.f_eval_hist[-1]
            delta = f_eval - f_eval_prev
            delta /= abs(f_eval_prev)

            f_val_m = f_eval - np.array(self.f_eval_hist).reshape(-1)
            sorted_indices = np.argsort(f_val_m)
            mth_max = sorted_indices[-self.kth_max]

            delta_mth_max = f_val_m[mth_max] / abs(f_eval_prev)
            if delta_mth_max >= self.tolerance:
                return False

        self.f_eval_hist.append(f_eval)
        self.beta_hist.append(beta)
        if len(self.f_eval_hist) > self.max_history:
            self.f_eval_hist.pop(0)
            self.beta_hist.pop(0)
        
        return True

# ------------------------ #

# ------------------------ #        
# ---- Set Convergence --- #
# ------  Method  -------- #
# ------------------------ #

def set_convergence_alg(p, convergence_method: str = 'SOR', **options_convergence_method):

    convergence_method = convergence_method.strip().lower()

    if convergence_method == 'none':
        w = 1 # No over-relaxation
        w_max = 1
        up_method = up_method = SOR_update_method(p = p,
                    w=w, w_max=w_max,
                    inc_factor=1, dec_factor=1)
        
        number_of_draws = 0
        dist_random_draw = 'normal'

    elif convergence_method in ['sor', 'random_sor']: 
        w = options_convergence_method.get('SOR_hyperparam', 1.5)
        w_max = options_convergence_method.get('max_SOR_hyperparam', 2.0)
        inc_factor = options_convergence_method.get('inc_factor', 1.1)
        dec_factor = options_convergence_method.get('dec_factor', 0.5)
        
        up_method = SOR_update_method(p = p,
                                  w=w, w_max=w_max,
                                  inc_factor=inc_factor, dec_factor=dec_factor)
        
        if convergence_method == 'random_sor':
            number_of_draws = options_convergence_method.get('number_of_draws', 3)
            dist_random_draw = options_convergence_method.get('dist_random_draw', 'uniform')
        else:    
            number_of_draws = 0 
            dist_random_draw = 'normal'

    
    elif convergence_method == 'andersonacceleration':
        windows_size = options_convergence_method.get('window_size', 5)
        reg = options_convergence_method.get('regularization_parameter', 0)
        mixing_param = options_convergence_method.get('mixing_parameter', 1.0)
        up_method = AndersonAcceleration(window_size=windows_size, reg=reg, mixing_param=mixing_param)

        number_of_draws = 0
        dist_random_draw = 'normal'

    else:
        raise ValueError("convergence_method must be either 'None', 'SOR', 'Random_SOR' or 'AndersonAcceleration'")

    return up_method, number_of_draws, dist_random_draw      

