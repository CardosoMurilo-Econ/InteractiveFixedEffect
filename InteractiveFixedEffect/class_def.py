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
        
        valid_criteria = ['None', 'twoways']
        if input_str not in valid_criteria:
            raise ValueError("Invalid criteria. Criteria must be ", ', '.join(valid_criteria))
        
        return str(input_str)
