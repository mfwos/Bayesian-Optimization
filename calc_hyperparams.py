import numpy as np
import pandas as pd
import os 
import glob
from scipy.optimize import Bounds, dual_annealing
import datetime


model_type = 'XGB'
streamline_part = 'Baseline'
rand_seed = 100


if os.getcwd()[1:7] == 'kaggle':
    SCRIPT_PATH = '/kaggle/input/icr-optimization/' + model_type + '/' + streamline_part + '/'
    DATA_PATH = '/kaggle/input/icr-identify-age-related-conditions/'
    OUTPUT_PATH = '/kaggle/working/'
else: 
    SCRIPT_PATH = 'C:/Users/mfwos/Documents/ML Projects/ICR - Age-Related Conditions/icr-optimization/' + model_type + '/' + streamline_part + '/'
    DATA_PATH = 'C:/Users/mfwos/Documents/ML Projects/ICR - Age-Related Conditions/data/'
    OUTPUT_PATH = SCRIPT_PATH

csv_files = glob.glob(SCRIPT_PATH + 'created_points_' + model_type + '_' + streamline_part + '_*')
df_evaluation = pd.DataFrame()

for file in csv_files:
    new_part = pd.read_csv(file)
    df_evaluation = pd.concat((df_evaluation, new_part))


eval_params = df_evaluation.drop(['cv_score','cv_std'], axis = 1).to_numpy()
scores = df_evaluation['cv_score'].to_numpy()
num_points = df_evaluation.shape[0]
num_vars = eval_params.shape[1]

def f_likelihood (x):
    
    mu = x[0]
    alpha0 = x[1]
    alphas = x[2:]
    
    sigma = np.diag(np.repeat(1, num_points)).astype(float)
    
    for i in range(num_points):
        for j in range(num_points):
        
            dist = (alphas * (eval_params[i,:] - eval_params[j,:])**2).sum()
            sigma[i,j] = alpha0 * np.exp(-dist)
    
    sigma = sigma + np.diag(np.repeat(1,num_points)).astype(float) * 1e-6
    
    sigma_inv = np.linalg.inv(sigma)
    sigma_det = np.linalg.det(sigma)
    
    likelihood = 0.5 * (scores - mu) @ (sigma_inv @ (scores - mu)) + np.log(np.sqrt(sigma_det))
    
    return likelihood

hparam_bounds = Bounds(np.repeat(1e-8, num_vars + 2), ub = np.repeat(3, num_vars + 2))
toc = datetime.datetime.now()
min_obj = dual_annealing(f_likelihood, bounds = hparam_bounds, seed = rand_seed)  
tic = datetime.datetime.now()
print(tic - toc)
    
hyper_params = min_obj.x

np_output_name = 'hyper_params_' + model_type + '_' + streamline_part + '.csv'
np.save(file = OUTPUT_PATH + np_output_name, arr = hyper_params)

df_output_name = 'df_evaluation_initial_' + model_type + '_' + streamline_part + '.csv'
df_evaluation.to_csv(OUTPUT_PATH + df_output_name, index = False)