'''
NAME
    train

AUTHOR
    Miguel Chamochin

DATE
    10 Abril 2022

VERSION
    v0 preliminar

DESCRIPTION
    Main program ML

REQUIREMENTS
    Ver fichero ./notebooks/project_resume.ipynb

FILE
./train.py

Este script entrena el modelo entrenamiento del modelo elegido 'my_model'
El modelo elegido es el modelo ya entrenado en 'Modelo_Pipeline_PCA_DecissionTree_Regression.py' 
y listo para poner en producción 

El coeficiente de determinacion de la predicción en Test de este modelo es 0.9484216064860829
'''

import sys
import os

import json

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '\\utils')

from functions import load_model



loaded_model, X_test, y_test = load_model('my_model')

print('El coeficiente de determinacion de la predicción en Test (loaded_model.score) es:', loaded_model.score(X_test, y_test))