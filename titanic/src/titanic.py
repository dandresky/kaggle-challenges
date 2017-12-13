import pandas as pd
import model_parameters as parameters
import cross_val as cv


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
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    # create the training and test data sets for regression. K-Fold splits for
    # cross validation will be handled in functions specific to each classifier
    X_trn = train_df.drop("Survived", axis=1)
    print(X_trn.info())
    y_trn = train_df["Survived"]
    X_tst = test_df
    # get the parameters and models for the desired estimators
    model_parameters = parameters.get_model_parameters(desired_estimators=['knn'])
    best_estimators = cv.fit_models(model_parameters, X_trn, y_trn)
    for model_name, gs in best_estimators.items():
        print("{} has f1 score of {}".format(model_name, gs.best_score_))
        print("{} best parameters: {}".format(model_name, gs.best_params_))
        #print("{} CV results: {}".format(model_name, gs.cv_results_))
    pass


if __name__ == '__main__':
    main()
