# PPHSE: Privacy-Preserving Hierarchical State Estimation [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository includes the codes of PPHSE, a Python framework designed for conducting privary-preserving hierarchical state estimation for interconnected power systems in untrustworthy cloud environments.

## Citation

Please cite the following paper when you refer to the framework.

> J. Wang, D. Shi, J. Chen, and C.-C. Liu, "Privacy-Preserving Hierarchical State Estimation in Untrustworthy Cloud Environments," *IEEE Trans. Smart Grid*, 2020, Early Access.

## Dependencies

The codes are tested based on Python 3.8. Compatibility with previous versions is not guaranteed. The following Python packages need to be installed beforehand:

- **Scientific computation:** numpy, scipy, scikit-learn
- **Data management and visualization:** pandas, prettytable
- **Cryptography:** gmpy2, cryptography
- **Power system computation:** pandapower

### Notes related to gmpy2

The Python package gmpy2 drastically improves the performance of cryptographic computations. However, it is usually not easy to install gmpy2, especially on the Windows platform. Please follow the instructions on the [documentation of gmpy2](https://gmpy2.readthedocs.io/en/latest/intro.html#installation) to make sure the package, along with its dependencies, is successfully installed.

### System requirements

The framework supports multiprocessing. More CPU cores are favorable for improving the efficiency of the framework.

## Getting started

```
$ python main.py
```

## About us

We are from Power System Analysis Research Group of Huazhong University of Science and Technology. Our research interests include power system computations, cybersecurity, and artificial intelligence applications. We welcome contributions and suggestions to this framework of any kind. If you have any question, please send emails to [jywang@hust.edu.cn](mailto://jywang@hust.edu.cn).
