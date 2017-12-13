'''
Uses sklearns GridSearchCV to perform an exhaustive search over specified
parameter values for an estimator. GridSearchCV implements a “fit” and a “score”
method. It also implements “predict”, “predict_proba”, “decision_function”,
“transform” and “inverse_transform” if they are implemented in the estimator used.

The parameters of the estimator used to apply these methods are optimized by
cross-validated grid-search over a parameter grid. It is computationally
intensive and intended to be run on AWS servers.
'''

from sklearn.model_selection import GridSearchCV

def fit_models(model_parameters, X, y):
    '''
    Process each of the models with GridSearchCV and return a dictionary where
    the key is the estimator abreviation and the value is a GridSearch object
    fit to the best parameters.

    This will be a serial for loop to start but my intention is to parallelize
    this operation to compute the fits as fast as possible.

    GridSearchCV is provided a list of scorers for flexibility, however, 'f1'
    is anticipated to be the best score given an expected imbalance in the
    y labels (~20% survived the sinking of the titanic)
    '''
    best_models_dict = {}
    for model_name, (model, param_grid) in model_parameters.items():
        gs = GridSearchCV(estimator=model,
                param_grid=param_grid,
                scoring=['accuracy','average_precision','f1','precision','recall','roc_auc'],
                fit_params=None,            # deprecated
                n_jobs=1,                   # come back to this for parallelization
                iid=False,                  # I think I want mean loss across folds
                refit='f1',                 # fit to the best f1 score
                cv=None,                    # use the default 3-fold CV
                verbose=False,              # run in AWS, don't need log
                pre_dispatch='2*n_jobs',    # come back to this for parallelization
                error_score='raise',        # raise errors
                return_train_score=True)    # I want training scores
        gs.fit(X, y)
        best_models_dict[model_name] = gs

    return best_models_dict
