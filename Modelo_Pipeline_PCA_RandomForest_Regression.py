# Import the required libraries
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
import os

import json

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '\\utils')

from functions import read_train_test_split_for_regression, read_train_test_split_for_classification, dump_model, load_model
from variables import fileXStrScores, fileXStrRanks, fileYStr

X_train, X_test, y_train, y_test = read_train_test_split_for_regression(fileXStrScores, fileYStr)

pipeline = Pipeline(steps = [
    ("scaler", StandardScaler()), # primero escalo
    ("pca", PCA()), # segundo aplica PCA 
    ("randomForestRegressor", RandomForestRegressor()) # Despues un RandomForestRegressor
])

pipeline_param = {
    'pca__n_components' :  [3],
    'pca__random_state' :  [42],
    'randomForestRegressor__n_estimators' :  [10],
    'randomForestRegressor__max_leaf_nodes' :  [25],   
    'randomForestRegressor__random_state' :  [42]
}

# pipeline.fit(X_train, y_train)
# pipeline.score(X_test, y_test)
search = GridSearchCV(pipeline, pipeline_param, cv=5).fit(X_train, y_train)

print("Train: Coeficiente de determinacion de la predicción:", search.best_estimator_.score(X_train, y_train))
print("Test: Coeficiente de determinacion de la predicción:", search.best_estimator_.score(X_test, y_test))

#
# Save Model
#
# Data to be written
model_description ={
    "nombre_alumno" : "Miguel Chamochin",
    "titulo" : "Conectando los Objetivos de Desarrollo Sostenible con el cambio climático y la transición energética",
    "tipo_ml" : "R",
    "target" : "Res_Nonhydro_Capita_2021"
}

test_csv = pd.concat([X_test, y_test], axis=1)

dump_model(search.best_estimator_, 'Modelo_Pipeline_PCA_RandomForest_Regression', model_description, test_csv)

loaded_model, X_test, y_test = load_model('Modelo_Pipeline_PCA_RandomForest_Regression')

print('loaded_model.score', loaded_model.score(X_test, y_test))


