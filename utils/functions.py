'''
NAME
    funciones

AUTHOR
    Miguel Chamochin

DATE
    10 Abril 2022

VERSION
    v0 preliminar

DESCRIPTION
    This file contains the declared functions
    
REQUIREMENTS
    Ver fichero ../notebooks/project_resume.ipynb: 

FILE
./utils/funciones.py

FUNCTIONS
'''

#
# Imports
#
import pandas as pd

from sklearn.model_selection import train_test_split

import pickle
import json

import warnings
warnings.filterwarnings("ignore")

from variables import data_path, model_path, Res_Nonhydro_Capita_2021, SDGs_Scores_2021
'''
    Read the files containing the independent and dependent var, and it splits then into 
    datasets for training and testing for regression
    Args:
        fileX (str): name of the file containing the independents vars
        fileY (str): name of the file containing the dependent vars
    Returns:
        X_train (DataFrame): Cointains the independents vars splitted for training
        X_test (DataFrame):  Cointains the independents vars splitted for test
        y_train (DataFrame): Cointains the dependents vars splitted for training
        y_test (DataFrame): Cointains the dependents vars splitted for test
'''
def read_train_test_split_for_regression(fileX, fileY):
    # Read the data
    SDGs_Scores_2021 = pd.read_excel (data_path + fileX, index_col = 0)
    Res_Nonhydro_Capita = pd.read_excel (data_path + fileY, index_col = 0)

    # Only consider the 2021 year
    Res_Nonhydro_Capita_2021 = Res_Nonhydro_Capita[[2021]]
    Res_Nonhydro_Capita_2021 = Res_Nonhydro_Capita_2021.rename(columns={ 2021: 'Res_Nonhydro_Capita_2021'})

    # Do not consider the first column
    SDGs_Scores_2021 = SDGs_Scores_2021.iloc[:,1:]

    # Append the target to the Datasets
    SDGs_Scores_2021 = pd.concat([SDGs_Scores_2021, Res_Nonhydro_Capita_2021], axis=1)

    X = SDGs_Scores_2021.drop('Res_Nonhydro_Capita_2021',1)
    y = SDGs_Scores_2021['Res_Nonhydro_Capita_2021']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train, X_test, y_train, y_test

'''
    Read the files containing the independent and dependent var, and it splits then into 
    datasets for training and testing for classification
    It dicothomizes the target variable using the median
    Args:
        fileX (str): name of the file containing the independents vars
        fileY (str): name of the file containing the dependent vars
    Returns:
        X_train (DataFrame): Cointains the independents vars splitted for training
        X_test (DataFrame):  Cointains the independents vars splitted for test
        y_train (DataFrame): Cointains the dependents vars splitted for training
        y_test (DataFrame): Cointains the dependents vars splitted for test
'''
def read_train_test_split_for_classification(fileX, fileY):
    # Read the data
    SDGs_Scores_2021 = pd.read_excel (data_path + fileX, index_col = 0)
    Res_Nonhydro_Capita = pd.read_excel (data_path + fileY, index_col = 0)

    # Only consider the 2021 year
    Res_Nonhydro_Capita_2021 = Res_Nonhydro_Capita[[2021]]
    Res_Nonhydro_Capita_2021 = Res_Nonhydro_Capita_2021.rename(columns={ 2021: 'Res_Nonhydro_Capita_2021'})

    # Do not consider the first column
    SDGs_Scores_2021 = SDGs_Scores_2021.iloc[:,1:]

    # Append the target to the Datasets
    SDGs_Scores_2021 = pd.concat([SDGs_Scores_2021, Res_Nonhydro_Capita_2021], axis=1)

    # Dicotomizing Res_Nonhydro_Capita_2021 in Wealthy and Unwealthy Countries
    median = Res_Nonhydro_Capita_2021['Res_Nonhydro_Capita_2021'].median()
    Res_Nonhydro_Capita_2021['Res_Nonhydro_Capita_2021'] = Res_Nonhydro_Capita_2021.Res_Nonhydro_Capita_2021.map(lambda x: 1 if x >= median else 0)
    Res_Nonhydro_Capita_2021 = Res_Nonhydro_Capita_2021.rename(columns={'Res_Nonhydro_Capita_2021': 'Wealthy'})

    # Append the target column "Wealthy"
    SDGs_Scores_2021 = pd.concat([SDGs_Scores_2021, Res_Nonhydro_Capita_2021], axis=1)

    X = SDGs_Scores_2021.drop('Wealthy',1)
    y = SDGs_Scores_2021['Wealthy']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train, X_test, y_train, y_test

'''
    Dump the model 
    This function creates thre files:
        <modelname>: the model dump with pickle
        <modelname>.json: Cointains a JSON with the description. 
        <modelname>.csv: Cointains the data with the dependent and independent vars
    Args:
        model (sklearn.pipeline.Pipeline): Cointains the model
        modelname (str):  Name of the model
        model_description (DataFrame): Cointains a JSON with the description
        test_csv (DataFrame): Cointains the dataframe with the dependent and independent vars
    Returns:
        n/a
'''
def dump_model(model, modelname, model_description, test_csv):
    with open(model_path + modelname, 'wb') as outfile1:
        pickle.dump(model, outfile1)
    
    with open(model_path + modelname + '.json', 'w') as outfile2:
        json.dump(model_description, outfile2)
    
    test_csv.to_csv(model_path + modelname + '.csv', index = True)

'''
    Load the model 
    Args:
        modelname (str):  Name of the model
    Returns:
        loaded_model (sklearn.pipeline.Pipeline): Cointains the model
        X_test (DataFrame):  Cointains the independents varswhich were splitted for testing
        y_test (DataFrame): Cointains the dependents vars which were splitted for testing
'''
def load_model(modelname):
    with open(model_path + modelname, 'rb') as inputfile:
        loaded_model = pickle.load(inputfile)
    df_test = pd.read_csv(model_path + modelname + '.csv', index_col = 'Country')

    # Leemos el json con la info
    with open(model_path + modelname + '.json') as json_file:
        info = json.load(json_file)
    target = info['target']
    X_test = df_test.drop(target,1)
    y_test = df_test[target]

    return loaded_model, X_test, y_test