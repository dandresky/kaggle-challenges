'''
Functions to build dictionaries of model parameter permutations to be used by
sklearns GridSearchCV cross-validation library.
Call get_model_parameters to get a master dictionary of all models.
Call any one of the other functions to get a model/parameter tuple for a
specific model.
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def get_rf_parameters():
    ''' Random Forest Parameters
    '''
    param = {"n_estimators": [300, 500, 700],
             "criterion": ['gini','entropy'],  # 'gini', 'entropy'
             "max_features": ['sqrt', 'log2', None],    # 'sqrt', 'log2', None, int, float
             "max_depth": [None],   # None or int
             "min_samples_split": [2],  # int or float (percent)
             "min_samples_leaf": [1],   # int or float (percent)
             "min_weight_fraction_leaf": [0.0], # float
             "max_leaf_nodes": [None],  # int or none
             "min_impurity_decrease": [0.0],    #float
             "bootstrap": [True],   # True False
             "oob_score": [False],  # True False
             "n_jobs": [-1],    # int, -1 means use all available cores
             "random_state": [39],  # int (seed) or None
             "verbose": [0],    # int 0 = no, 1 = yes, do ints increase verbosity?
             "warm_start": [False], # True (add trees to previous) False (start over)
             "class_weight": [None] # None (equal weights to all classes) or dict of weights
             }
    model = RandomForestClassifier()
    return (model, param)

def get_ada_parameters(base_estimator=None):
    ''' AdaBoost Parameters
    Input: base_estimator will be Decision Tree by default, this allows to specify other
    '''
    param = {'base_estimator': [base_estimator],  # object, None (Desicion Tree)
             'n_estimators': [100,200,300,400,500],    # int
             'learning_rate': [0.1,0.25,0.5,0.75,1],   # int
             'algorithm': ['SAMME.R'],   # SAMME.R, SAMME
             "random_state": [39]  # int (seed) or None
             }
    model = AdaBoostClassifier()
    return (model, param)

def get_svm_parameters():
    ''' SVM Parameters
    '''
    param = {'C': [0.0001],
             'kernel': ['rbf', 'sigmoid'],  # 'linear', 'poly', 'rbf', 'sigmoid'
             'degree': [3], # int (degree of polynomial)
             'gamma': ['auto'], # 'auto' (1/n) or float (rbf, poly, sigmoid)
             'coef0': [0.0],    # float (poly or sigmoid)
             'probability': [False],    # boolean
             'shrinking': [True],   # boolean
             'tol': [0.001],    # float
             'cache_size': [200],   # float (kernel cache im MB)
             'class_weight': [None],    # dict of class weights
             'verbose': [False],    # boolean
             'max_iter': [-1],  # int, -1 means no limit
             'decision_function_shape': ['ovr'],    # 'ovo', 'ovr'
             "random_state": [39]  # int (seed) or None
             }
    model = SVC()
    return (model, param)

def get_knn_parameters():
    ''' KNN Parameters
    '''
    knn_parameters = {'n_neighbors': [10, 15, 20],  # int
                      'weights': ['distance', 'uniform'],   #'distance', 'uniform'
                      'algorithm': ['auto'],  # auto, ball_tree, kd_tree, brute
                      'leaf_size': [30],    # int
                      'p': [2], # int (power metric for minkowski)
                      'metric': ['minkowski'],  # minkowski or callable
                      'n_jobs': [-1]    # int, -1 means use all available cores
                      }
    model = KNeighborsClassifier()
    return (model, knn_parameters)

def get_lr_parameters():
    ''' Logistic Regression model parameters to choose from
    penalty=’l2’,
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver=’liblinear’,
    max_iter=100,
    multi_class=’ovr’,
    verbose=0,
    warm_start=False,
    n_jobs=1
    '''
    lr_parameters = {"penalty": ['l1', 'l2'],
                     "C": [1,2, 3]}
    model = LogisticRegression()
    return (model, lr_parameters)

def get_dt_parameters():
    ''' Decision Tree parameters to choose from
    criterion=’gini’,
    splitter=’best’,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    class_weight=None,
    presort=False
    '''
    dt_parameters = {"splitter": ['best', 'random'],
                     "criterion": ['gini', 'entropy']}
    model = DecisionTreeClassifier()
    return (model, dt_parameters)

def get_mlp_parameters():
    ''' MLP parameters to choose from
    hidden_layer_sizes=(100, ),
    activation=’relu’,
    solver=’adam’,
    alpha=0.0001,
    batch_size=’auto’,
    learning_rate=’constant’,
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=200,
    shuffle=True,
    random_state=None,
    tol=0.0001,
    verbose=False,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=False,
    validation_fraction=0.1,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08
    '''
    mlp_parameters = {"hidden_layer_sizes": [(30,), (40,)],
                      "activation": ['tanh', 'relu'],
                      "solver": ['sgd'],
                      "alpha": [0.0005, 0.001]}
    model = MLPClassifier()
    return (model, mlp_parameters)

def get_gbc_parameters():
    '''
    loss=’deviance’,
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    criterion=’friedman_mse’,
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
    presort=’auto’
    '''
    gbc_parameters = {"n_estimators": [250, 300],
                      "learning_rate": [0.1,0.05,0.15],
                      'max_depth': [2,3,4],
                      'max_features': ['auto', 'log2']}
    model = GradientBoostingClassifier()
    return (model, gbc_parameters)

def get_model_parameters(desired_estimators=['rf','knn','gbc']):
    '''
    return a dictionary - keys are the classifier abreviations, values are
    model/parameter tuples
    '''
    parameter_dict = {}
    if 'rf' in desired_estimators:
        parameter_dict.update({"rf": get_rf_parameters()})
    if 'svm' in desired_estimators:
        parameter_dict.update({"svm": get_svm_parameters()})
    if 'knn' in desired_estimators:
        parameter_dict.update({"knn": get_knn_parameters()})
    if 'lr' in desired_estimators:
        parameter_dict.update({"lr": get_lr_parameters()})
    if 'dt' in desired_estimators:
        parameter_dict.update({"dt": get_dt_parameters()})
    if 'mlp' in desired_estimators:
        parameter_dict.update({"mlp": get_mlp_parameters()})
    if 'gbc' in desired_estimators:
        parameter_dict.update({"gbc": get_gbc_parameters()})
    if 'ada' in desired_estimators:
        parameter_dict.update({'ada': get_ada_parameters()})
    return parameter_dict
