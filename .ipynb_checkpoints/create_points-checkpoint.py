import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def f_create_points(X, y, rand_seed, num_points):

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

    choice_state = np.random.RandomState(rand_seed)
   
    init_max_depth = choice_state.uniform(2, 7, size = num_points)
    init_colsample_bytree = choice_state.uniform(0.05, 1, size = num_points)
    init_lambda = choice_state.uniform(0, 10, size = num_points)
    init_alpha = choice_state.uniform(0, 10, size = num_points)
    init_eta = choice_state.uniform(0.05, 1, size = num_points)
    init_num_round = choice_state.uniform(0, 7.5, size = num_points)

    df_evaluation = pd.DataFrame({'max_depth': init_max_depth,
                     'colsample_bytree': init_colsample_bytree,
                     'lambda': init_lambda,
                     'alpha': init_alpha,
                     'eta': init_eta,
                     'log_num_round': init_num_round,
                     'cv_score': np.nan,
                     'cv_std': np.nan})

    for i in range(len(df_evaluation)):
        
        model_params = {'max_depth': np.round(df_evaluation.loc[i,'max_depth']).astype(int), 'objective': 'binary:logistic',
                 'colsample_bytree': df_evaluation.loc[i,'colsample_bytree'], 
                 'lambda': df_evaluation.loc[i,'lambda'], 'alpha': df_evaluation.loc[i,'alpha'], 
                 'eta': df_evaluation.loc[i,'eta'],
                 'eval_metric': 'logloss', 'seed': 100}
        
        num_round = (np.round(np.exp(df_evaluation.loc[i,'log_num_round']))).astype(int)
        
        mcv_results = multiple_cv(X,y, model_params, num_round, 6, 30, 0, False)
        df_evaluation.loc[i,'cv_score'] = mcv_results['cv_mean']
        df_evaluation.loc[i,'cv_std'] = mcv_results['cv_std']
        
        print("Finished run for point number ", i+1)
    
    return df_evaluation