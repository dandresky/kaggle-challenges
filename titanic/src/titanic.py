import pandas as pd
import model_parameters as parameters
import cross_val as cv

from model_parameters import RANDOM_FOREST
from model_parameters import SUPPORT_VECTOR_MACHINE
from model_parameters import K_NEAREST
from model_parameters import LOGISTIC_REGRESSION
from model_parameters import DECISION_TREE
from model_parameters import GRADIENT_BOOSTING
from model_parameters import ADA_BOOST


def get_data(train_filename, test_filename):
    train_df = pd.read_csv(train_filename)
    test_df = pd.read_csv(test_filename)
    return train_df, test_df

def clean_data(df):
    # drop PassengerID, Name and Ticket
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    # convert Sex to binary values 1=male, 0=female
    df.loc[df['Sex']=='male', 'Sex'] = 1
    df.loc[df['Sex']=='female', 'Sex'] = 0
    # fill in missing age values (see notebook)
    df.loc[(df.Age.isnull()) & (df.Parch > 0), 'Age'] = 40
    df.loc[(df.Age.isnull()) & (df.Pclass == 1), 'Age'] = 38
    df.loc[(df.Age.isnull()) & (df.Pclass == 2), 'Age'] = 30
    df.loc[(df.Age.isnull()) & (df.Pclass == 3), 'Age'] = 26
    # Change Age values to ordinals based on Age Band computed in the notebook
    df.loc[df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[df['Age'] > 64, 'Age'] = 4
    # Change Fare values to ordinals based on Fare Band computed in the notebook
    df.loc[df['Fare'] <= 75, 'Fare'] = 0
    df.loc[df['Fare'] > 75, 'Fare'] = 1
    # Change Embarked values to ordinals based on examination in the notebook
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 2
    df.loc[df.Embarked.isnull(), 'Embarked'] = 0
    return df


def main():
    # read the raw data
    train_filename = '../data/train.csv'
    test_filename = '../data/test.csv'
    train_df, test_df = get_data(train_filename, test_filename)
    # clean the data (see jupyter notebook for analysis)
    train_df_cln = clean_data(train_df)
    # the test data has a missing fare value. The 50th percentile of the fare
    # training data is $14.45 so that is what will be assigned to the missing
    # value
    test_df.loc[test_df.Fare.isnull(), 'Fare'] = 14.45
    test_df_cln = clean_data(test_df)
    # create the training and test data sets for regression. K-Fold splits for
    # cross validation will be handled in functions specific to each classifier
    X_trn = train_df_cln.drop("Survived", axis=1)
    print("\nEngineered Training Data Set (see Jupyter Notebook)\n", X_trn.head(10))
    print("\n", X_trn.describe())
    y_trn = train_df_cln["Survived"]
    print("\n", y_trn.head(10))
    X_tst = test_df_cln
    print("\nEngineered Test Data Set (see Jupyter Notebook)\n", X_tst)
    # Perform model evaluation using sklearns GridSearchCV, first request dictionary
    # of parameter permutations
    # print("\n\n\nModel under test: ", GRADIENT_BOOSTING, "\n")
    # model_parameters = parameters.get_model_parameters(desired_estimators=[GRADIENT_BOOSTING])
    # best_estimators = cv.fit_models(model_parameters, X_trn, y_trn)
    # for model_name, gs in best_estimators.items():
    #     print("\n{} has f1 score of {}".format(model_name, gs.best_score_))
    #     print("\n{} best parameters: {}".format(model_name, gs.best_params_))
    #     #print("{} CV results: {}".format(model_name, gs.cv_results_))
    # pass
    # Alternatively, perform model evaluations one at a time with fixed parameters
    print("\n\n\nModel under test: ", RANDOM_FOREST, "\n")
    model = cv.eval_model(RANDOM_FOREST, X_trn, y_trn, splits=5)
    # compute predictions for test data and prepare submission file
    Y_pred = model.predict(X_tst)
    prediction_df = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
    prediction_df.to_csv('../output/submission.csv', index=False)


if __name__ == '__main__':
    main()
