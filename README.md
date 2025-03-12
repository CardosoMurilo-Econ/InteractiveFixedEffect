# InteractiveFixedEffect

This Python library provides functions to:

1) Compute **Principal Components** using the functions *PCA* or *factor_dimensionality* to extract factors from a dataset by maximizing tr(F'X'XF) or tr(L'X'XL), where F is the common factor, L is the loading factor, and X is the data matrix. This method follows the approach proposed by [Bai and Ng (2002)](https://doi.org/10.1111/1468-0262.00273).

2) Estimate the **Interactive Fixed Effects** (IFE) model using the *IFE* function for large panel datasets based on the following equations:

   <p align="center">
        <img src="https://quicklatex.com/cache3/37/ql_7d2c8093abe1106c499aef4ac2e1ae37_l3.png" alt="Equation">
    </p>
   
   or

   <p align="center">
        <img src="https://quicklatex.com/cache3/3a/ql_5296184788c6b40e449e57949e3abe3a_l3.png" alt="Equation">
    </p>

   The algorithm is based on [Bai (2009)](https://doi.org/10.3982/ECTA6135).

All functions can be optimized using **GPU acceleration** for improved computational efficiency, particularly for large T and N. To enable GPU support, set: `Torch_cuda = True`.

![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue)
![GPU Available](https://img.shields.io/badge/GPU-Available-green)

## Installation

## Usage

To use `factorAnalysis`, import it as follows:

```python
import factorAnalysis as fa
```

### Estimating PCA

To estimate $\hat{F}_t$, $\hat{\lambda}_i$, and $\hat{k}$, consider the following model:

<p align="center">
        <img src="https://quicklatex.com/cache3/7b/ql_ceac7db3998b0272b70ab6ca4ddb3a7b_l3.png" alt="Equation">
</p>

where $F_{t}$ and $\lambda_i$ are $k \times 1$ matrices.

```python
# Defining a random X as an example
import numpy as np
T, N, k = 1000, 1000, 2
F = np.random.normal(1, 1, (T, k))
L = np.random.normal(-2, 1, (N, k))
X = F @ L.T + np.sqrt(k) * np.random.normal(0, 1, (T, N))

# Estimating factors and dimensionality
PCA = fa.factor_dimensionality(X)

# Summarizing the results
PCA.summary()
```

### Estimating the Interactive Fixed Effects Model

To estimate the interactive fixed effects model:
<p align="center">
        <img src="https://quicklatex.com/cache3/d5/ql_88885fa8fe17ebe7ce4363aaebcc2bd5_l3.png" alt="Equation">
</p>


1) Assuming $\epsilon_{it}$ is i.i.d:
      <p align="center">
        <img src="https://quicklatex.com/cache3/1e/ql_6d3321d6ab63556005f4ee6cf5a9e31e_l3.png" alt="Equation">
      </p>
    
    ```python
    # Creating a random dataset
    import numpy as np
    T, N, k = 200, 100, 2
    F = np.random.normal(1, 1, (T, k))
    L = np.random.normal(-2, 1, (N, k))
    E = np.random.normal(0, 1, (T, N))
    Lx1 = np.random.normal(2, 1, (N, k))
    Lx2 = np.random.normal(-1, 1, (N, k))
    X1 = 1 + F @ Lx1.T + np.random.normal(0, 1, (T, N))
    X2 = 2 + F @ Lx2.T + np.random.normal(0, 1, (T, N))

    alpha, beta1, beta2 = -2, -1, 2
    Y = alpha + beta1 * X1 + beta2 * X2 + F @ L.T + E

    # Estimating the IFE model
    Output = fa.IFE(Y, [X1, X2], fixed_effects='twoways', Variance_type='iid')

    # Summarizing the results
    Output.summary()
    ```

2) Assuming heteroskedastic variance of $\epsilon_{it}$:

    A bias-correction procedure is applied:

    <p align="center">
        <img src="https://quicklatex.com/cache3/cb/ql_711fea253d9a3a2517a02a154060c1cb_l3.png" alt="Equation">
     </p>

    where

    <p align="center">
        <img src="https://quicklatex.com/cache3/ea/ql_a13af8777a3892268f78c36c981972ea_l3.png" alt="Equation">
     </p>

    with ![Equation](https://quicklatex.com/cache3/16/ql_8cd17681c860d94c42b7155ae651c916_l3.png).

    ```python
    # Creating a dataset with heteroskedastic errors
    import numpy as np
    T, N, k = 150, 200, 2
    F = np.random.normal(1, 1, (T, k))
    L = np.random.normal(-2, 1, (N, k))
    Lx1 = np.random.normal(2, 1, (N, k))
    Lx2 = np.random.normal(-1, 1, (N, k))
    X1 = 1 + F @ Lx1.T + np.random.normal(0, 1, (T, N))
    X2 = 2 + F @ Lx2.T + np.random.normal(0, 1, (T, N))

    alpha, beta1, beta2 = -2, 2, 1
    
    sigmait = np.random.chisquare(2, (T, N)).reshape((T, N))
    E = np.random.normal(0, 1, (T, N)) * sigmait

    Y = alpha + beta1 * X1 + beta2 * X2 + F @ L.T + E

    # Estimating the IFE model with heteroskedastic variance
    Output = fa.IFE(Y, [X1, X2], fixed_effects='twoways', Variance_type='heteroskedastic')

    # Summarizing the results
    Output.summary()
    ```

## Features

| Feature                | Description |
|------------------------|-------------|
| ✅ Easy to Use        | Simple function usage |
| 🚀 Fast Execution     | Optimized for performance with and without GPU acceleration |
| 🚀 GPU Computation | Allow PyTorch GPU computation for all functions if a GPU is available and PyTorch CUDA 11.8 or later is installed |
| ✅ Bias Correction    | Automatically applies bias correction for $\hat{\beta}$ when heteroskedastic covariance calculation is set  |

## GPU Acceleration Guide

This package utilizes PyTorch for GPU computation. To install PyTorch with CUDA, follow the instructions at [PyTorch](https://pytorch.org/get-started/locally/). Remember to uninstall Torch from your virtual environment before proceeding with the installation of PyTorch with CUDA.

To verify installation:

```python
import torch
torch.cuda.is_available()
```

If `True`, CUDA is available, and functions can be run with `Torch_cuda = True`.

```python
import factorAnalysis as fa

Output = fa.IFE(Y, [X1, X2], fixed_effects='twoways', Variance_type='heteroskedastic', Torch_cuda=True)
```

### Performance Comparison

Above, we present a performance comparison between 100 runs of the algorithm in the IFE function using CPU vs. GPU.

| N   | T    | CPU Time (s) | GPU Time (s) |
|-----|------|--------------|--------------|
| 100 | 100  | 5.26         | 6.51         |
| 100 | 500  | 102.27       | 30.62        |
| 100 | 1000 | 322.42       | 23.48        |
| 500 | 100  | 34.09        | 6.69         |
| 500 | 500  | 268.51       | 27.99        |
| 500 | 1000 | 617.09       | 21.15        |
| 1000| 100  | 124.92       | 6.17         |
| 1000| 500  | 744.43       | 25.25        |
| 1000| 1000 | 1597.97      | 24.95        |

\* The efficiency between CPU and GPU may vary depending on your hardware.

\* Hardware used in the test:

    CPU: AMD Ryzen™ 9 5900X - Clock: 4.8 GHz (max) / 3.7 GHz (basic)
    GPU: Nvidia Geforce RTX 4070 - Boost Clock: 2.48 GHz / Cuda Cores: 5888

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## Contact

For issues, open an issue on GitHub or reach out via murilo.s.cardoso@outlook.com.
