# GBSGE

This repository contains python code for approximating the Gaussian expectation problems using GBS-I, GBS-P, and MC methods. 

## Features

- Generate Gaussian expectation problems.
- Simulate and compare GBS-I, GBS-P, and MC methods.
- Create figures and tables in the companion paper.

## Installation

You can install this software directly from GitHub:

```bash
pip install git+https://github.com/sshanshans/GBEGE.git
```

## Usage

### 1. Generate a Gaussian Expectation Problem
To generate a Gaussian expectation problem, use one of the following scripts:
- `script/model_haf.py`
- `script/model_hafsq.py`

---

### 2. Simulate Comparisons
To simulate and compare the performance of GBS-I, GBS-P, and MC methods, run:
- `script/run_est_haf.py`
- `script/run_est_hafsq.py`

---

### 3. Generate Figures
To produce figures for analyzing the convergence behaviors, use the script:
- `script/make_fig_haf.py`
- `script/make_fig_hafsq.py`

---

### 4. Generate Tables
To reproduce tables displayed in the paper, run one of the example scripts:
- `script/example1.py`
- `script/example2.py`
- `script/example3.py`
- `script/example4.py`


## License
This software is licensed under the GPL-3.0 License. See the LICENSE file for details.

## Citations
If you use this software in your research, please cite the associated paper:

**Using Gaussian Boson Samplers for Approximate Gaussian Expectation Problems**  
Jørgen Ellegaard Andersen, Shan Shan (2025)


## Contributions
Contributions are welcome! Please submit a pull request or open an issue if you encounter bugs or have suggestions.