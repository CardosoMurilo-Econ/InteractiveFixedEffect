from setuptools import find_packages, setup


setup(
    name='InteractiveFixedEffect',
    packages=find_packages(),
    version='1.2.0',
    description='1 - Estimates the **number of factors (k)** in large-dimensional factor models, along with the **common factors (F)** and **factor loadings (L)** for a given matrix T x N (**X**). The estimation is based on one of the criteria proposed by Bai and Ng (2002): https://doi.org/10.1111/1468-0262.00273. \n'
    '2 - Estimates the interactive fixed effect model in large-panel data, along with the equation $y_{it} = X_{it} \\beta + \lambda_i F_t _e_{it}$. The estimation is based in Bai (2009): https://doi.org/10.3982/ECTA6135',
    author='Murilo Cardoso',
    author_email='murilo.s.cardoso@outlook.com',

    install_requires=[
        'numpy==2.0.2', 
        'pandas==2.2.3',
        'scipy==1.13.1',
        'torch'
        ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1']
)