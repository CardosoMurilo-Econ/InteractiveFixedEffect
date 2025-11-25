import numpy as np

class k_class(int):
    def __new__(cls, input_int: int, N, T):

        if input_int is None:
            return None
        if not isinstance(input_int, int):
            raise TypeError("k must be an integer or None.")
        if input_int < 1:
            raise ValueError("k must be greater than 0.")
        if input_int >= min(N, T):
            raise ValueError("k must be less than the minimum of N and T.")
        return int(input_int)

class k_max_class(int):
    def __new__(cls, input_int: int, N, T):
        if not isinstance(input_int, int):
            raise TypeError("k_max must be an integer.")
        if input_int < 1:
            raise ValueError("k_max must be greater than 0.")
        if input_int >= min(N, T):
            raise ValueError("k_max must be less than the minimum of N and T.")
        return int(input_int)

class criteria_class(list):
    def __new__(cls, input_list: list[str]):
        if not isinstance(input_list, list):
            if isinstance(input_list, str):
                input_list = [input_list]
            else:
                raise TypeError("Input must be a list.")
        if not all(isinstance(x, str) for x in input_list):
            raise TypeError("All elements must be strings.")
        
        valid_criteria = ['PC1', 'PC2', 'PC3', 'IC1', 'IC2', 'IC3']
        if not all(x in valid_criteria for x in input_list):
            raise ValueError("Invalid criteria. Criteria must be a list with the following options: 'PC1', 'PC2', 'PC3', 'IC1', 'IC2', 'IC3'.")
        return list(input_list)

class Matrix(np.ndarray):
    def __new__(cls, input_array: np.ndarray):
        if not isinstance(input_array, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if input_array.ndim != 2:
            raise ValueError("Input must be a 2D matrix (T x N).")
        
        obj = np.asarray(input_array).view(cls)  # Create subclassed array
        return obj

class var_type(str):
    def __new__(cls, input_str: str):
        if not isinstance(input_str, str):
            raise TypeError("Input must be a string.")
        
        valid_types = ['iid', 'heteroskedastic']

        if input_str not in valid_types:
            raise ValueError(f"Invalid input. Must be one of the following:  {', '.join(valid_types)}")
        return str(input_str)

class fixed_effect(str):
    def __new__(cls, input_str: str):
        
        if not isinstance(input_str, str):
            raise TypeError("Input must be a string.")
        
        valid_input = ['none', 'demeaned', 'twoways']
        if input_str not in valid_input:
            raise ValueError("Invalid fixed_effect. fixed_effect must be ", ', '.join(valid_input))
        
        return str(input_str)

class criteria_conv_class(list):
    def __new__(cls, input_list: list[str]):
        if not isinstance(input_list, list):
            if isinstance(input_list, str):
                input_list = [input_list]
            else:
                raise TypeError("Input must be a list.")
        if not all(isinstance(x, str) for x in input_list):
            raise TypeError("All elements must be strings.")
        
        valid_criteria = ['Relative_norm', 'Obj_fun', 'Grad_norm']
        if not all(x in valid_criteria for x in input_list):
            raise ValueError("Invalid criteria. Criteria must be a list with the following options: 'PC1', 'PC2', 'PC3', 'IC1', 'IC2', 'IC3'.")
        return list(input_list)

class restrict_PCA_class(str):
    def __new__(cls, input_str: str):
        if not isinstance(input_str, str):
            raise TypeError("Input must be a string.")
        
        valid_types = ['optimize', 'common', 'loading']

        if input_str not in valid_types:
            raise ValueError(f"Invalid input. Must be one of the following:  {', '.join(valid_types)}")
        return str(input_str)
    
class tolerance_class(np.ndarray):
    def __new__(cls, input_array: np.ndarray, convergence_criteria):
        
        if not isinstance(input_array, np.ndarray):
            if isinstance(input_array, list):
                input_array = np.array(input_array)
            elif isinstance(input_array, (float, int)):
                input_array = np.array([input_array]*len(convergence_criteria))
            else:
                raise TypeError("Tolerance must be a NumPy array, a list or a single float.")
        
        if input_array.ndim != 1:
            raise ValueError("Input must be a 1D array.")
        
        if input_array.shape[0] == 1:
            input_array = np.repeat(input_array, len(convergence_criteria))
        elif input_array.shape[0] != len(convergence_criteria):
            raise ValueError("Length of input array must be 1 or match the number of convergence criteria.")

        return input_array
    
class SOR_hyperparam_class(float):
    def __new__(cls, input: float, max_SOR_hyperparam, inc_factor, dec_factor):
        
        if not isinstance(input, (float, int)):
            raise TypeError("SOR_hyperparam must be a float.")
        
        if not isinstance(max_SOR_hyperparam, (float, type(None))):
            raise TypeError("max_SOR_hyperparam must be a float or None.")
        
        if not isinstance(inc_factor, float):
            raise TypeError("inc_factor must be a float.")
        
        if not inc_factor > 1:
            raise ValueError("inc_factor must be greater than 1.")
        
        if not isinstance(dec_factor, float):
            raise TypeError("dec_factor must be a float.")
        if not 0 < dec_factor < 1:
            raise ValueError("dec_factor must be between 0 and 1.")

        if input < 1:
            raise ValueError("SOR_hyperparam must be greater than 1.")
        if max_SOR_hyperparam is not None and input >= max_SOR_hyperparam:
            raise ValueError("SOR_hyperparam must be less or equal to max_SOR_hyperparam.")
        return float(input), max_SOR_hyperparam, inc_factor, dec_factor 
    
class boll_class():
    def __new__(cls, input_bool: bool, name = ''):
        if not isinstance(input_bool, bool):
            raise TypeError(f"{name} must be a boolean. (True or False)")
        return input_bool