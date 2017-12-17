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

# constant definitions
RANDOM_FOREST = 'rf'
SUPPORT_VECTOR_MACHINE = 'svm'
K_NEAREST = 'knn'
LOGISTIC_REGRESSION = 'lr'
DECISION_TREE = 'dt'
GRADIENT_BOOSTING = 'gbc'
ADA_BOOST = 'ada'

VERBOSE_FALSE = False
VERBOSE_TRUE = True

def get_rf_parameters():
    ''' Random Forest Parameters
    '''
    param = {"n_estimators": [20, 30, 40],
             "criterion": ['gini','entropy'],  # 'gini', 'entropy'
             "max_features": ['auto'],    # 'sqrt', 'log2', None, int, float
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
             "verbose": [VERBOSE_FALSE],    # int 0 = no, 1 = yes, do ints increase verbosity?
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
             "class_weight": [None, 'balanced'],    #  None or balanced
             'verbose': [VERBOSE_FALSE],    # boolean
             'max_iter': [-1],  # int, -1 means no limit
             'decision_function_shape': ['ovr'],    # 'ovo', 'ovr'
             "random_state": [39]  # int (seed) or None
             }
    model = SVC()
    return (model, param)

def get_knn_parameters():
    ''' KNN Parameters
    '''
    param = {'n_neighbors': [10, 15, 20],  # int
             'weights': ['distance', 'uniform'],   #'distance', 'uniform'
             'algorithm': ['auto'],  # auto, ball_tree, kd_tree, brute
             'leaf_size': [30],    # int
             'p': [2], # int (power metric for minkowski)
             'metric': ['minkowski'],  # minkowski or callable
             'n_jobs': [-1]    # int, -1 means use all available cores
             }
    model = KNeighborsClassifier()
    return (model, param)

def get_lr_parameters():
    ''' Logistic Regression parameters
    '''
    param = {"penalty": ['l2'],   # l1 or l2
             "dual": [False],   # sklearn prefers False when samples > features
             "tol": [0.0001],   # floatt (tolerance for stopping criteria)
             "C": [0.25, 0.5, 0.75, 1.0],   # float (Inverse of regularization strength, smaller value means high reg)
             "fit_intercept": [True, False],
             "intercept_scaling": [1,2,3,4],  # float (Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True)
             "class_weight": [None, 'balanced'],    #  None or balanced
             "random_state": [39],  # int (seed) or None
             "solver": ['liblinear', 'lbfgs', 'newton-cg'],   # ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
             "max_iter": [50, 100], # Useful only for the newton-cg, sag and lbfgs solvers
             "multi_class": ['ovr'],    # str, {‘ovr’, ‘multinomial’}, default: ‘ovr’
             "verbose": [VERBOSE_FALSE],
             "warm_start": [False], # True (build on previous results) False (start over)
             "n_jobs": [1]  # int (Number of CPU cores used when parallelizing over classes if multi_class=’ovr’)
             }
    model = LogisticRegression()
    return (model, param)

def get_dt_parameters():
    ''' Decision Tree Classifier parameters
    '''
    param = {'criterion': ['gini','entropy'],  # gini or entropy
             'splitter': ['best', 'random'],    # best or random
             'max_depth': [None],  # int or None
             'min_samples_split': [2,4,6],  # minimum number of samples required to split an internal node
             'min_samples_leaf': [1],   # minimum number of samples required to be at a leaf node
             'min_weight_fraction_leaf': [0.0], # minimum weighted fraction of the sum total of weights
             'max_features': [None],    # number of features to consider when looking for the best split
             'random_state': [39],  # int (seed) or None
             'max_leaf_nodes': [None],  # int or None
             'min_impurity_decrease': [0.0],    # A node will be split if this split induces a decrease of the impurity greater than or equal to this value
             'min_impurity_split': [None],  # Threshold for early stopping in tree growth
             "class_weight": [None, 'balanced'],    #  None or balanced
             'presort': [False, True] # presort the data to speed up the finding of best splits in fitting
             }
    model = DecisionTreeClassifier()
    return (model, param)

def get_gbc_parameters():
    ''' Gradient Boosting parameters
    '''
    param = {'loss': ['deviance', 'exponential'], # deviance or exponential (loss function to be optimized)
             'learning_rate': [0.1,0.15,0.20],    # learning rate shrinks the contribution of each tree
             'n_estimators': [100,150,200,250],    # The number of boosting stages to perform
             'max_depth': [2,3], # limits the number of nodes in the tree
             'criterion': ['friedman_mse', 'mse', 'mae'],   # friedman_mse, mse, mae
             'min_samples_split': [2], # minimum number of samples required to split an internal node
             'min_samples_leaf': [2], # minimum number of samples required to be at a leaf node
             'min_weight_fraction_leaf': [0.0],    # minimum weighted fraction of the sum total of weights
             'subsample': [1.0],   #  fraction of samples to be used for fitting the individual base learners
             'max_features': ['sqrt', 'log2', None], # int, sqrt, log2, None
             'max_leaf_nodes': [None],
             'min_impurity_split': [None], #
             'min_impurity_decrease': [0.0],   #
             'init': [None],   # estimator object that is used to compute the initial predictions
             'verbose': [VERBOSE_FALSE],
             'warm_start': [False], # True (build on previous results) False (start over)
             'random_state': [39],  # int (seed) or None
             'presort': ['auto'] # bool or 'auto' - presort the data to speed up the finding of best splits in fitting
             }
    model = GradientBoostingClassifier()
    return (model, param)

def get_model_parameters(desired_estimators=['rf','knn','gbc']):
    '''
    return a dictionary - keys are the classifier abreviations, values are
    model/parameter tuples
    '''
    parameter_dict = {}
    if RANDOM_FOREST in desired_estimators:
        parameter_dict.update({RANDOM_FOREST: get_rf_parameters()})
    if SUPPORT_VECTOR_MACHINE in desired_estimators:
        parameter_dict.update({SUPPORT_VECTOR_MACHINE: get_svm_parameters()})
    if K_NEAREST in desired_estimators:
        parameter_dict.update({K_NEAREST: get_knn_parameters()})
    if LOGISTIC_REGRESSION in desired_estimators:
        parameter_dict.update({LOGISTIC_REGRESSION: get_lr_parameters()})
    if DECISION_TREE in desired_estimators:
        parameter_dict.update({DECISION_TREE: get_dt_parameters()})
    if GRADIENT_BOOSTING in desired_estimators:
        parameter_dict.update({GRADIENT_BOOSTING: get_gbc_parameters()})
    if ADA_BOOST in desired_estimators:
        parameter_dict.update({ADA_BOOST: get_ada_parameters()})
    return parameter_dict
