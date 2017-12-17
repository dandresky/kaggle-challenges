'''
fit_models uses sklearns GridSearchCV to perform an exhaustive search over specified
parameter values for an estimator. GridSearchCV implements a “fit” and a “score”
method. It also implements “predict”, “predict_proba”, “decision_function”,
“transform” and “inverse_transform” if they are implemented in the estimator used.

The parameters of the estimator used to apply these methods are optimized by
cross-validated grid-search over a parameter grid. It is computationally
intensive and intended to be run on AWS servers.

eval_model implements a model directly in lieu of GridSearchCV to compare results.
'''

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from model_parameters import RANDOM_FOREST
from model_parameters import K_NEAREST
from model_parameters import GRADIENT_BOOSTING

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

def eval_model(model_name, X, y, splits=5):
    '''
    Split the data into K folds and perform cross validation on a model.
    Parameters for the model are set prior to calling this function.
    '''
    # build the model
    model = None
    if model_name == RANDOM_FOREST:
        model = get_rf_model()
    if model_name == K_NEAREST:
        model = get_knn_model()
    if model_name == GRADIENT_BOOSTING:
        model = get_gbc_model()
    model.fit(X, y)
    print(model.score(X, y))
    return model

def get_rf_model():
    model = RandomForestClassifier(n_estimators=30,
            criterion='entropy', # 'gini', 'entropy'
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto', # 'sqrt', 'log2', None, int, float
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=1,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None)
    return model

def get_knn_model():
    model = KNeighborsClassifier(n_neighbors=7,
            weights='distance', #'distance', 'uniform'
            algorithm='auto', # auto, ball_tree, kd_tree, brute
            leaf_size=30,
            p=2, # int (power metric for minkowski)
            metric='minkowski',
            metric_params=None,
            n_jobs=1)
    return model

def get_gbc_model():
    model = GradientBoostingClassifier(loss='deviance', # deviance or exponential (loss function to be optimized)
            learning_rate=1.20,
            n_estimators=300,
            subsample=1.0,
            criterion='mse',
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_depth=3,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            init=None,
            random_state=None,
            max_features=None,
            verbose=0,
            max_leaf_nodes=None,
            warm_start=False,
            presort='auto')
    return model
