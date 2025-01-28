from src.utils.CoeffDict import CoeffDict
from src.utils.Estimator import Estimator
from src.utils.Plotter import Plotter
import pickle
import os
from os.path import join
import numpy as np
from src._helpers.check import *

N = 3
K = 10
exp_id = f'haf-N_{N}-K_{K}'
path_to_folder = f'/work/GBSGE/exp/{exp_id}'
with open(f"{path_to_folder}/{exp_id}.pkl", "rb") as f:
    T = pickle.load(f)
print('gt', T.gt)

total_sample_used = int(1e9)
num_threads = 1
thread_size = int(total_sample_used/num_threads)
step_size = 1000

# load the estimator
for which_rslt in ['val_est', 'mul_err', 'add_err']:
    path_to_estimates=f"{path_to_folder}/estimates/GBSP"
    path_to_file = join(path_to_estimates, f'{which_rslt}.pkl')
    with open(path_to_file, "rb") as f:
        est_gbsi = pickle.load(f)
    
    path_to_estimates=f"{path_to_folder}/estimates/MC"
    path_to_file = join(path_to_estimates, f'{which_rslt}.pkl')
    with open(path_to_file, "rb") as f:
        est_mc = pickle.load(f)
    
    plotter = Plotter()
    check_and_create_folder(f'{path_to_folder}/figures/')
    plotter.plot_threads_single_method(est_gbsi, color='red', label="GBS-P")
    plotter.plot_threads_single_method(est_mc, color='blue', label="MC")
    plotter.plot_single_line(T.gt, label="Ground Truth")
    plotter.set_labels(title="", xlabel="n (number of samples)", ylabel=create_ylabel(which_rslt))
    plotter.set_ylim(crete_ylim(which_rslt, T.gt, 1))
    plotter.set_tick_labels(total_sample_used, step_size)
    plotter.save_plot(f'{path_to_folder}/figures/{which_rslt}.png')