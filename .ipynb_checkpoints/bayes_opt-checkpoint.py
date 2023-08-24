import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.optimize import Bounds, dual_annealing
from sklearn.model_selection import StratifiedKFold
import datetime
from scipy.stats import norm as norm_dist

def f_bayes_opt(X, y, hyper_params, num_new_points, df_evaluation, rand_seed):

    def eval_model (X_train, X_val, y_train, y_val, model_params, num_round, verbose):
        
        dtrain = xgb.DMatrix(X_train, label = y_train)
        dvalid = xgb.DMatrix(X_val, label = y_val)
        evallist = [(dtrain, 'train'), (dvalid, 'eval')]
        model_params_input = model_params.copy()
        model_params_input['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()
        
        bst = xgb.train(model_params_input, dtrain, num_round, evals = evallist, verbose_eval = verbose)

        preds = bst.predict(dvalid)

        preds = preds + 0.05 * (1-preds)
        blogloss = (- 1/(len(y_val) - y_val.sum()) * ((1-y_val) * np.log(1 - preds)).sum()  - 1/y_val.sum() * (y_val * np.log(preds)).sum() ) / 2
        
        return blogloss

    def cv_params (X, y, model_params, num_round, num_folds, rand_seed, verbose):
        
        rand_state = np.random.RandomState(rand_seed)
        skf = StratifiedKFold(n_splits = num_folds, random_state = rand_state, shuffle = True)
        all_blls = np.ones(num_folds) * 999
        
        for i, (train_idxs, val_idxs) in enumerate(skf.split(X,y)):
            
            curr_blogloss = eval_model(X.iloc[train_idxs,:], X.iloc[val_idxs,:], y[train_idxs], y[val_idxs], model_params, num_round, False)
            all_blls[i] = curr_blogloss
            
            if verbose:
                print("Balanced logloss for iteration", i+1, ": ", curr_blogloss)
        
        return all_blls.mean()


    def multiple_cv (X, y, model_params, num_round, num_folds, num_cvs, start_seed, verbose):
        
        cv_scores = np.ones(num_cvs) * 999
        for i,j in enumerate(range(start_seed,start_seed + num_cvs)):
            cv_scores[i] = cv_params(X, y, model_params, num_round, num_folds, j, verbose)
        
        results = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
        
        return results    

    def f_expected_improvement(x):
        
        prior_points = df_evaluation[['max_depth','colsample_bytree','lambda','alpha','eta','log_num_round']].to_numpy()
        prior_score = df_evaluation.loc[:,'cv_score'].to_numpy()
        n = prior_points.shape[0]
        
        sigma0_x_xn = hyper_params[1] * np.exp(-(hyper_params[2:] * (x - prior_points)**2).sum(1))
        sigma0_xn_xn = hyper_params[1] * np.exp(-(hyper_params[2:] * (np.tile(prior_points, n).reshape(n,n,6) - prior_points)**2).sum(2))
        sigma0_xn_xn_inv = np.linalg.inv(sigma0_xn_xn)
        sigma_x_x = hyper_params[1] * np.exp(-(hyper_params[2:] * (x - x)**2).sum())
        
        mu_n = sigma0_x_xn @ (sigma0_xn_xn_inv @ (prior_score - hyper_params[0])) + hyper_params[0]
        std_n = np.sqrt(sigma_x_x - sigma0_x_xn @ (sigma0_xn_xn_inv @ sigma0_x_xn))
        
        delta_n = prior_score.min() - mu_n # Looking for minimum of the function
        ds_ratio = delta_n / std_n
        
        EI = np.maximum(delta_n, 0) + std_n * norm_dist.pdf(ds_ratio) - np.abs(delta_n) * norm_dist.cdf(ds_ratio)
        
        # The optimization algorithm we will use is a minimization algorithm, hence to find the
        # maximum of expected improvement we will return its negative
        
        return -EI 
    
    var_bounds = Bounds(lb = np.array([2,0.05,0,0,0.05,0]), ub = np.array([7,1,10,10,1,7.5]))
    
    for m in range(num_new_points):
    
        toc = datetime.datetime.now()
        min_obj = dual_annealing(f_expected_improvement, bounds = var_bounds, seed = rand_seed)  
        
        new_point = min_obj.x
        
        # evaluate new point
        
        model_params = {'max_depth': np.round(new_point[0]).astype(int), 'objective': 'binary:logistic',
                 'colsample_bytree': new_point[1], 
                 'lambda': new_point[2], 'alpha': new_point[3], 
                 'eta': new_point[4],
                 'eval_metric': 'logloss', 'seed': 100}
        
        num_round = (np.round(np.exp(new_point[5]))).astype(int)
        eval_result = multiple_cv(X,y, model_params, num_round, 6, 30, 0, False)
        
        new_line = {'max_depth': new_point[0], 'colsample_bytree': new_point[1],
                    'lambda': new_point[2], 'alpha': new_point[3],
                    'eta': new_point[4], 'log_num_round': new_point[5],
                    'cv_score': eval_result['cv_mean'],
                    'cv_std': eval_result['cv_std']}
        df_evaluation = df_evaluation.append(new_line, ignore_index = True)
        
        tic = datetime.datetime.now()
        print("Punkt ", m + 1, " berechnet. Verstrichene Zeit: ", tic - toc)
    
    return df_evaluation