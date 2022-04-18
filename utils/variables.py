'''
NAME
    variables

AUTHOR
    Miguel Chamochin

DATE
    10 Abril 2022

VERSION
    v0 preliminar

DESCRIPTION
    This file contains the declared constants
    
REQUIREMENTS
    Ver fichero ../notebooks/project_resume.ipynb: 

FILE
./utils/variables.py

VARIABLES
'''
#
# Imports
#
import sys
import os

import pandas as pd
import numpy as np
import sys
import os

import pandas as pd
import numpy as np


# Data Directoriess
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + '\\..\\data\\processed\\'
model_path = dir_path + '\\..\\model\\'

# Name of files containing the dependent and independet variables
fileXStrScores = "SDGs_Scores_2021.xlsx"
fileXStrRanks = "SDGs_Ranks_2021.xlsx"
fileYStr = "Res_Nonhydro_Capita.xlsx"

# Global variables
SDGs_Scores_2021 = pd.DataFrame()
Res_Nonhydro_Capita_2021 = pd.DataFrame()

