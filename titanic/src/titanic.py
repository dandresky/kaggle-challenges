import pandas as pd
import numpy as np


def get_data(train_filename, test_filename):
    train_df = pd.read_csv(train_filename)
    test_df = pd.read_csv(test_filename)
    return train_df, test_df

def clean_data(df):
    # drop PassengerID and Ticket
    df = df.drop(['PassengerId', 'Ticket'], axis=1)
    return df


def main():
    # read the raw data
    train_filename = '../data/train.csv'
    test_filename = '../data/test.csv'
    train_df, test_df = get_data(train_filename, test_filename)
    # clean the data (see jupyter notebook for analysis)
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    pass


if __name__ == '__main__':
    main()
