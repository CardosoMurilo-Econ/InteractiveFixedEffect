# InteractiveFixedEffect

This Python library provides functions to:

1) Compute **Principal Components** using the functions *PCA* or *factor_dimensionality* to extract factors from a dataset by maximizing tr(F'X'XF) or tr(L'X'XL), where F is the common factor, L is the loading factor, and X is the data matrix. This method follows the approach proposed by [Bai and Ng (2002)](https://doi.org/10.1111/1468-0262.00273).

2) Estimate the **Interactive Fixed Effects** (IFE) model using the *IFE* function for large panel datasets based on the following equations:

  $$ X_{it} = \lambda_i^\prime F_t + \varepsilon_{it} $$

  or 

  $$ Y_{it} = X_{it}^\prime \beta + \alpha_i + \gamma_t + \lambda_i^\prime F_t + \varepsilon_{it} $$

   The algorithm is based on [Bai (2009)](https://doi.org/10.3982/ECTA6135).

All functions can be optimized using **GPU acceleration** for improved computational efficiency, particularly for large T and N. To enable GPU support, set: `Torch_cuda = True`.

![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue)
![GPU Available](https://img.shields.io/badge/GPU-Available-green)

## Installation

To install the package, run the following command:

```bash
pip install git+https://github.com/CardosoMurilo-Econ/InteractiveFixedEffect.git
```

**Note**: Git must be installed on your system before proceeding. To check if Git is installed, run:

```bash
 git --version
```

If Git is not installed, download and install it from [git-scm.com](https://git-scm.com/downloads).

## Usage

To use `InteractiveFixedEffect`, import it as follows:

```python
import InteractiveFixedEffect as IFE
```

### Estimating PCA

To estimate $\hat{F}_t$, $\hat{\lambda}_i$, and $\hat{k}$, consider the following model:

$$ X_{it} = \lambda_i^\prime F_t + \varepsilon_{it} $$

where $F_{t}$ and $\lambda_i$ are $k \times 1$ matrices.

```python
# Defining a random X as an example
import numpy as np
T, N, k = 1000, 1000, 2
F = np.random.normal(1, 1, (T, k))
L = np.random.normal(-2, 1, (N, k))
X = F @ L.T + np.sqrt(k) * np.random.normal(0, 1, (T, N))

# Estimating factors and dimensionality
PCA = IFE.factor_dimensionality(X)

# Summarizing the results
PCA.summary()
```

### Estimating the Interactive Fixed Effects Model

To estimate the interactive fixed effects model:

$$ Y_{it} = X_{it}^\prime \beta + \alpha_i + \gamma_t + \lambda_i^\prime F_t + \varepsilon_{it} $$

1) Assuming $\epsilon_{it}$ is i.i.d:

  $$ \sqrt{NT} (\hat{\beta} - \beta) \sim N\left(0, \sigma^2 D^{-1}\right) $$
    
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
  Output = IFE.IFE(Y, [X1, X2], fixed_effects='twoways', Variance_type='iid')

  # Summarizing the results
  Output.summary()
  ```

3) Assuming heteroskedastic variance of $\epsilon_{it}$:

  A bias-correction procedure is applied:

  $$ \beta^\dagger = β - (1/N) B - (1/T) C $$

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
  Output = IFE.IFE(Y, [X1, X2], fixed_effects='twoways', Variance_type='heteroskedastic')

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
import InteractiveFixedEffect as IFE

Output = IFE.IFE(Y, [X1, X2], fixed_effects='twoways', Variance_type='heteroskedastic', Torch_cuda=True)
```

## 🚀 Performance Comparison  

Below, we present a performance comparison between **100 runs** of the algorithm in the IFE function using **CPU vs. GPU**.  
As you can see, **GPU acceleration significantly outperforms the CPU**, especially as `N` and `T` increase. ⚡🔥  

| N    | T    | CPU Time (s) ⏳ | GPU Time (s) ⚡ | CPU/GPU Speedup 🚀 |
|------|------|--------------|--------------|------------------|
| 100  | 100  | 8.99         | 6.11         | 1.5x             |
| 100  | 500  | 122.26       | 29.29        | 4.2x             |
| 100  | 1000 | 360.91       | 24.59        | 14.7x 🚀        |
| 500  | 100  | 63.71        | 6.50         | 9.8x  🚀        |
| 500  | 500  | 409.91       | 26.64        | 15.4x 🚀        |
| 500  | 1000 | 902.78       | 23.55        | 38.3x 🚀⚡      |
| 1000 | 100  | 206.17       | 6.22         | 33.1x 🚀⚡      |
| 1000 | 500  | 1159.62      | 25.85        | 44.9x 🚀⚡      |
| 1000 | 1000 | 2413.17      | 30.09        | 80.2x 💀➡️🚀    |

**Note:** 

\* The CPU/GPU Speedup column represents how many times faster the GPU is compared to the CPU.

\* The efficiency between CPU and GPU may vary depending on your hardware.

\* Hardware used in the test:

    CPU: AMD Ryzen™ 9 5900X - Clock: 4.8 GHz (max) / 3.7 GHz (basic)
    GPU: Nvidia Geforce RTX 4070 - Boost Clock: 2.48 GHz / Cuda Cores: 5888

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## Contact

For issues, open an issue on GitHub or reach out via murilo.s.cardoso@outlook.com.
