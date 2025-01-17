from fastapi import FastAPI, Request
from typing import List
from pydantic import BaseModel
import pickle 

import warnings
warnings.filterwarnings('ignore')
from numpy import set_printoptions
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
pd.options.display.max_columns=None
import seaborn as sns 
from pandas import read_csv
import io
import base64
import json
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage

import os
import os.path
from unidecode import unidecode

from urllib.error import HTTPError

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from mlens.ensemble import SuperLearner

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_log_error, median_absolute_error
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from sklearn.neighbors import KNeighborsRegressor
from fastapi import FastAPI, HTTPException
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# CARGUE DE DATOS
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

app = FastAPI()
carrera = ""
anio = ""
mejor_modelo = None

variables_por_carrera = { 
    'industrial': {
        '1': ['PG_ICFES','CON_MAT_ICFES','FISICA_ICFES','QUIMICA_ICFES','IDIOMA_ICFES','PROMEDIO_UNO','LOCALIDAD_COLEGIO','BIOLOGIA_ICFES',
              'LOCALIDAD','CAR_UNO','NCC_UNO','NAA_UNO','NOTA_DIFERENCIAL','NOTA_DIBUJO','NOTA_QUIMICA','NOTA_CFJC','NOTA_TEXTOS','NOTA_SEMINARIO','NOTA_EE_UNO',
              'PROMEDIO_DOS','PROMEDIO_TRES'],

        '2': ['PG_ICFES', 'CON_MAT_ICFES', 'FISICA_ICFES','QUIMICA_ICFES','IDIOMA_ICFES','LOCALIDAD', 'PROMEDIO_UNO', 'LOCALIDAD_COLEGIO', 
              'BIOLOGIA_ICFES', 'CAR_UNO', 'NCC_UNO', 'NAA_UNO', 'NOTA_DIFERENCIAL', 'NOTA_DIBUJO', 'NOTA_QUIMICA', 'NOTA_CFJC', 'NOTA_TEXTOS', 'NOTA_SEMINARIO', 'NOTA_EE_UNO',
              'PROMEDIO_DOS','NCC_DOS', 'NCA_DOS', 'NAA_DOS', 'NOTA_ALGEBRA', 'NOTA_INTEGRAL', 'NOTA_MATERIALES', 'NOTA_PBASICA', 'NOTA_EE_DOS', 
              'PROMEDIO_TRES','NAA_TRES', 'NOTA_MULTIVARIADO', 'NOTA_ESTADISTICA_UNO', 'NOTA_TERMODINAMICA', 'NOTA_TGS', 'NOTA_EE_TRES','PROMEDIO_CUATRO', 'PROMEDIO_CINCO'],

        '3': ['PG_ICFES', 'CON_MAT_ICFES', 'FISICA_ICFES','QUIMICA_ICFES','IDIOMA_ICFES','LOCALIDAD','PROMEDIO_UNO', 'LOCALIDAD_COLEGIO', 'BIOLOGIA_ICFES', 'CAR_UNO', 
              'NCC_UNO', 'NAA_UNO', 'NOTA_DIFERENCIAL', 'NOTA_DIBUJO', 'NOTA_QUIMICA', 'NOTA_CFJC', 'NOTA_TEXTOS', 'NOTA_SEMINARIO', 'NOTA_EE_UNO','PROMEDIO_DOS', 'NCC_DOS', 
              'NCA_DOS', 'NAA_DOS', 'NOTA_ALGEBRA', 'NOTA_INTEGRAL', 'NOTA_MATERIALES', 'NOTA_PBASICA', 'NOTA_EE_DOS','PROMEDIO_TRES', 'NAA_TRES', 'NOTA_MULTIVARIADO', 'NOTA_ESTADISTICA_UNO', 
              'NOTA_TERMODINAMICA', 'NOTA_TGS', 'NOTA_EE_TRES','PROMEDIO_CUATRO', 'NOTA_ECUACIONES', 'NOTA_ESTADISTICA_DOS', 'NOTA_FISICA_DOS', 'NOTA_MECANICA', 'NOTA_PROCESOSQ', 
              'PROMEDIO_CINCO', 'NOTA_EE_CUATRO', 'NOTA_PROCESOSM', 'NOTA_ADMINISTRACION', 'NOTA_LENGUA_UNO', 'NOTA_EI_UNO', 'NOTA_EI_DOS','PROMEDIO_SEIS', 'NCA_SEIS', 'NOTA_PLINEAL', 'NOTA_DISENO', 
              'NOTA_EI_TRES','PROMEDIO_SIETE', 'NOTA_IECONOMICA', 'NAA_SIETE', 'NOTA_GRAFOS', 'NOTA_CALIDAD_UNO', 'NOTA_ERGONOMIA', 'NOTA_EI_CINCO','PROMEDIO_OCHO','PROMEDIO_NUEVE']
    },
    'sistemas': {
        '1': ['CON_MAT_ICFES','IDIOMA_ICFES','LOCALIDAD_COLEGIO','BIOLOGIA_ICFES','QUIMICA_ICFES','PG_ICFES','LITERATURA_ICFES',
              'PROMEDIO_UNO','NOTA_DIFERENCIAL','NOTA_CATEDRA_DEM','NOTA_PROG_BASICA','NAA_UNO','CAR_UNO','NOTA_TEXTOS','NOTA_LOGICA','NOTA_SEMINARIO',
              'VECES_LOGICA','NAC_UNO','NCC_UNO','NCA_UNO','VECES_CATEDRA_DEM','NOTA_EE_UNO','NAP_UNO','VECES_DIFERENCIAL',
              'NOTA_CATEDRA_CON','VECES_EE_UNO','PROMEDIO_TRES'],
        
        '2': ['CON_MAT_ICFES','IDIOMA_ICFES','LOCALIDAD_COLEGIO','BIOLOGIA_ICFES','QUIMICA_ICFES','PG_ICFES','LITERATURA_ICFES',
              'PROMEDIO_UNO','NOTA_DIFERENCIAL','NOTA_CATEDRA_DEM','NOTA_PROG_BASICA','NAA_UNO','CAR_UNO','NOTA_TEXTOS','NOTA_LOGICA','NOTA_SEMINARIO',
              'VECES_LOGICA','NAC_UNO','NCC_UNO','NCA_UNO','VECES_CATEDRA_DEM','NOTA_EE_UNO','NAP_UNO','VECES_DIFERENCIAL',
              'NOTA_CATEDRA_CON','VECES_EE_UNO','PROMEDIO_DOS','NOTA_TGS','NOTA_PROG_AVANZADA','NOTA_FISICA_DOS','NOTA_MULTIVARIADO',
              'NOTA_SEGUNDA_LENGUA_DOS','VECES_FISICA_DOS','NOTA_EE_DOS','VECES_PROG_AVANZADA','NAA_DOS','VECES_TGS','NCA_DOS','VECES_MULTIVARIADO','NOTA_ECUACIONES','NAC_DOS',
              'NOTA_METODOS_NUM','NOTA_A_SISTEMAS','NOTA_MOD_PROG_UNO','NOTA_MAT_DIS','NOTA_HOMBRE','NOTA_GT_DOS','VECES_A_SISTEMAS','VECES_HOMBRE','NCC_TRES','NAA_TRES',
              'NOTA_FUND_BASES','NAC_TRES','PROMEDIO_CINCO'],

        '3': ['CON_MAT_ICFES','IDIOMA_ICFES','LOCALIDAD_COLEGIO','BIOLOGIA_ICFES','QUIMICA_ICFES','PG_ICFES','LITERATURA_ICFES',
              'PROMEDIO_UNO','NOTA_DIFERENCIAL','NOTA_CATEDRA_DEM','NOTA_PROG_BASICA','NAA_UNO','CAR_UNO','NOTA_TEXTOS','NOTA_LOGICA','NOTA_SEMINARIO',
              'VECES_LOGICA','NAC_UNO','NCC_UNO','NCA_UNO','VECES_CATEDRA_DEM','NOTA_EE_UNO','NAP_UNO','VECES_DIFERENCIAL',
              'NOTA_CATEDRA_CON','VECES_EE_UNO','PROMEDIO_DOS','NOTA_TGS','NOTA_PROG_AVANZADA','NOTA_FISICA_DOS',
              'NOTA_MULTIVARIADO','NOTA_SEGUNDA_LENGUA_DOS','VECES_FISICA_DOS','NOTA_EE_DOS','VECES_PROG_AVANZADA','NAA_DOS','VECES_TGS','NCA_DOS','VECES_MULTIVARIADO','NOTA_ECUACIONES',
              'NAC_DOS','NOTA_METODOS_NUM','NOTA_A_SISTEMAS','NOTA_MOD_PROG_UNO','NOTA_MAT_DIS','NOTA_HOMBRE','NOTA_GT_DOS','VECES_A_SISTEMAS',
              'VECES_HOMBRE','NCC_TRES','NAA_TRES','NOTA_FUND_BASES','NAC_TRES','PROMEDIO_TRES','NOTA_PROB','NOTA_CIENCIAS_COMP_UNO',
              'NOTA_MOD_PROG_DOS','NOTA_ARQ_COMP','NOTA_INV_OP_UNO','NOTA_FISICA_TRES','VECES_INV_OP_UNO','VECES_PROB','VECES_MOD_PROG_DOS','VECES_FISICA_TRES',
              'NOTA_HISTORIA_COL','NOTA_ESTADISTICA','NOTA_CIBERNETICA_UNO','NOTA_INV_OP_DOS','NOTA_REDES_COM_UNO','NOTA_GI_UNO','VECES_HISTORIA_COL','VECES_CIBERNETICA_UNO',
              'NOTA_CIENCIAS_COMP_DOS','VECES_REDES_COM_UNO','VECES_GI_UNO','VECES_ESTADISTICA','VECES_CIENCIAS_COMP_DOS','VECES_INV_OP_DOS','CAR_CINCO',
              'PROMEDIO_CINCO','NOTA_F_ING_SOF','NOTA_GT_TRES','NOTA_ECONOMIA','NOTA_EI_DOS','NOTA_REDES_COM_DOS','VECES_ECONOMIA','NOTA_CIBERNETICA_UDOS','NOTA_INV_OP_TRES',
              'VECES_F_ING_SOF','VECES_REDES_COM_DOS','VECES_EI_DOS','VECES_CIBERNETICA_DOS','NCP_SEIS','NOTA_GI_DOS','PROMEDIO_SEIS','NOTA_DIS_ARQ_SOFT','NOTA_EI_OP_AI',
              'NOTA_ING_ECONOMICA','NOTA_REDES_COM_TRES','NOTA_CIBERNETICA_TRES','VECES_EI_OP_AI','VECES_DIS_ARQ_SOFT','VECES_ING_ECONOMICA','VECES_EI_OP_BI',
              'VECES_CIBERNETICA_TRES','PROMEDIO_NUEVE']  
    },
    'catastral': {
        '1': ['CON_MAT_ICFES','IDIOMA_ICFES','BIOLOGIA_ICFES','GENERO','FILOSOFIA_ICFES','LOCALIDAD','LITERATURA_ICFES','QUIMICA_ICFES','PG_ICFES',
              'PROMEDIO_UNO','NOTA_ALGEBRA','NOTA_DITO','NOTA_TEXTOS','NOTA_CD&C','NCA_UNO','NAA_UNO','NOTA_HIST','NOTA_SEM',
              'NOTA_DIF','NAP_UNO','VECES_HIST','VECES_ALGEBRA','NCP_UNO','PROMEDIO_TRES'],

        '2': ["CON_MAT_ICFES","IDIOMA_ICFES","BIOLOGIA_ICFES","GENERO","FILOSOFIA_ICFES","LOCALIDAD","LITERATURA_ICFES","QUIMICA_ICFES","PG_ICFES","PROMEDIO_UNO",
              "NOTA_ALGEBRA","NOTA_DITO","NOTA_TEXTOS","NOTA_CD&C","NCA_UNO","NAA_UNO","NOTA_HIST","NOTA_SEM","NOTA_DIF","NAP_UNO","VECES_HIST","VECES_ALGEBRA","NCP_UNO",
              "PROMEDIO_DOS","EPA_DOS","NOTA_TOP","NOTA_PROGB","NOTA_FIS_UNO","NOTA_INT","VECES_TOP","NOTA_EYB","PROMEDIO_TRES","NOTA_SUELOS","NOTA_POO","NOTA_FIS_DOS",
              "NOTA_ECUA","NCA_TRES","NOTA_FOTO","NOTA_CFJC","VECES_SUELOS","PROMEDIO_CINCO"],
        
        '3': ["CON_MAT_ICFES","IDIOMA_ICFES","BIOLOGIA_ICFES","GENERO","FILOSOFIA_ICFES","LOCALIDAD","LITERATURA_ICFES","QUIMICA_ICFES","PG_ICFES","PROMEDIO_UNO",
              "NOTA_ALGEBRA","NOTA_DITO","NOTA_TEXTOS","NOTA_CD&C","NCA_UNO","NAA_UNO","NOTA_HIST","NOTA_SEM","NOTA_DIF","NAP_UNO","VECES_HIST","VECES_ALGEBRA","NCP_UNO",
              "PROMEDIO_DOS","EPA_DOS","NOTA_TOP","NOTA_PROGB","NOTA_FIS_UNO","NOTA_INT","VECES_TOP","NOTA_EYB","PROMEDIO_TRES","NOTA_SUELOS","NOTA_POO","NOTA_FIS_DOS",
              "NOTA_ECUA","NCA_TRES","NOTA_FOTO","NOTA_CFJC","VECES_SUELOS","PROMEDIO_CUATRO","NOTA_GEOGEO","NCA_CUATRO","NOTA_PRII","EPA_CUATRO",
              "NOTA_PROB","PROMEDIO_CINCO","NOTAS_SISC","VECES_BD","NOTA_ME","CAR_CINCO","NOTA_CARTO","VECES_CARTO","PROMEDIO_SEIS","NOTA_GEO_FIS","NOTA_L_DOS","NOTA_LEGIC",
              "NOTA_INGECO","VECES_L_DOS","PROMEDIO_SIETE","NOTA_ECONOMET","NOTA_AVAP","NOTA_EI_UNO","PROMEDIO_NUEVE"],
    },
    'electrica':{
        '1': ['IDIOMA_ICFES','PG_ICFES','FISICA_ICFES','ANO_INGRESO','APT_VERB_ICFES','QUIMICA_ICFES','FILOSOFIA_ICFES','GENERO','BIOLOGIA_ICFES',
              'PROMEDIO_UNO','NOTA_POO','NOTA_INT','NOTA_FIS_DOS','NOTA_EYB','NOTA_DEM','NOTA_CIR_UNO','NOTA_HCI','VECES_INT','VECES_FIS_DOS','PROMEDIO_TRES'],

        '2': ["IDIOMA_ICFES","PG_ICFES","FISICA_ICFES","ANO_INGRESO","APT_VERB_ICFES","QUIMICA_ICFES","FILOSOFIA_ICFES","GENERO","BIOLOGIA_ICFES","PROMEDIO_UNO",
              "NOTA_POO","NOTA_INT","NOTA_FIS_DOS","NOTA_EYB","NOTA_DEM","NOTA_CIR_UNO","NOTA_HCI","VECES_INT","VECES_FIS_DOS","PROMEDIO_DOS",
              "NOTA_ECUA","NOTA_TERMO","NOTA_CIR_DOS","NOTA_EL_UNO","VECES_ECUA","NOTA_FIS_TRES","NOTA_MULTI","NCP_DOS","NCA_DOS","VECES_TERMO","NOTA_EL_DOS","NOTA_MPI","NOTA_ME",
              "NOTA_PYE","NOTA_ELD","VECES_ME","VECES_EL_DOS","NOTA_GEE","NOTA_CIR_TRES","VECES_PYE","PROMEDIO_CINCO"],   
        
        '3': ["IDIOMA_ICFES","PG_ICFES","FISICA_ICFES","ANO_INGRESO","APT_VERB_ICFES","QUIMICA_ICFES","FILOSOFIA_ICFES","GENERO","BIOLOGIA_ICFES","PROMEDIO_UNO","NOTA_POO",
              "NOTA_INT","NOTA_FIS_DOS","NOTA_EYB","NOTA_DEM","NOTA_CIR_UNO","NOTA_HCI","VECES_INT","VECES_FIS_DOS","PROMEDIO_DOS","NOTA_ECUA","NOTA_TERMO",
              "NOTA_CIR_DOS","NOTA_EL_UNO","VECES_ECUA","NOTA_FIS_TRES","NOTA_MULTI","NCP_DOS","NCA_DOS","VECES_TERMO","NOTA_EL_DOS","NOTA_MPI","NOTA_ME","NOTA_PYE","NOTA_ELD","VECES_ME","VECES_EL_DOS",
              "NOTA_GEE","NOTA_CIR_TRES","VECES_PYE","PROMEDIO_CUATRO","NOTA_ASD","NOTA_CEM","NOTA_IO_UNO","NOTA_INSE","NOTA_IYM","NOTA_EE_UNO","NOTA_GENH","VECES_CEM",
              "VECES_IYM","VECES_INSE","NOTA_INGECO","NOTA_DDP","NOTA_CTRL","NOTA_CONEM","NOTA_TDE","VECES_INGECO","VECES_IO_DOS","VECES_DDP","NAC_CINCO","NOTAS_RDC","VECES_ECO","NOTA_L_UNO",
              "NOTA_ASP","NOTA_ELECPOT","NOTA_ECO","NOTA_MAQEL","VECES_ASP","NOTA_AUTO","PROMEDIO_SIETE","NOTA_FGEP","NOTA_HIST","NOTA_CCON","NOTA_L_DOS","NOTA_AIELEC","NOTA_SUBEL","VECES_SUBEL",
              "VECES_FGEP","VECES_L_DOS","VECES_HIST","NOTA_L_TRES","NOTA_HSE","NOTA_ADMIN","NOTA_TGUNO","VECES_HSE","NOTA_EI_TRES","NOTA_EE_DOS","NOTA_PROTELEC","NCC_OCHO","VECES_L_TRES","VECES_PROTELEC",
              "CAR_OCHO","VECES_ADMIN","PROMEDIO_NUEVE"]
    },
    'electronica':{
        '1': ["CON_MAT_ICFES","QUIMICA_ICFES","PG_ICFES","FISICA_ICFES","IDIOMA_ICFES","BIOLOGIA_ICFES","GENERO","LOCALIDAD","LITERATURA_ICFES","PROMEDIO_UNO",
              "NOTA_DIF","EPA_UNO","NOTA_PROGB","NOTA_FIS_UNO","NCP_UNO","NOTA_TEXTOS","NOTA_CFJC","NAP_UNO","PROMEDIO_TRES"],
        
        '2': ["CON_MAT_ICFES","QUIMICA_ICFES","PG_ICFES","FISICA_ICFES","IDIOMA_ICFES","BIOLOGIA_ICFES","GENERO","LOCALIDAD","LITERATURA_ICFES","PROMEDIO_UNO",
              "NOTA_DIF","EPA_UNO","NOTA_PROGB","NOTA_FIS_UNO","NCP_UNO","NOTA_TEXTOS","NOTA_CFJC","NAP_UNO","PROMEDIO_DOS","NAA_DOS","NOTA_INT",
              "VECES_INT","NCA_DOS","NOTA_FIS_DOS","CAR_DOS","PROMEDIO_TRES","NCA_TRES","NOTA_FIS_TRES","NOTA_EL_UNO","VECES_FIS_TRES","NOTA_DEM","PROMEDIO_CINCO"],

        '3': ["CON_MAT_ICFES","QUIMICA_ICFES","PG_ICFES","FISICA_ICFES","IDIOMA_ICFES","BIOLOGIA_ICFES","GENERO","LOCALIDAD","LITERATURA_ICFES","PROMEDIO_UNO",
              "NOTA_DIF","EPA_UNO","NOTA_PROGB","NOTA_FIS_UNO","NCP_UNO","NOTA_TEXTOS","NOTA_CFJC","NAP_UNO","PROMEDIO_DOS","NAA_DOS","NOTA_INT",
              "VECES_INT","NCA_DOS","NOTA_FIS_DOS","CAR_DOS","PROMEDIO_TRES","NCA_TRES","NOTA_FIS_TRES","NOTA_EL_UNO","VECES_FIS_TRES","NOTA_DEM",
              "PROMEDIO_CUATRO","NOTA_EL_DOS","NOTA_CAMPOS","NOTA_FCD","VECES_FCD","NOTA_PROGAPL","NOTA_VARCOM","NCP_CUATRO","PROMEDIO_CINCO","NOTA_EE_DOS","NCC_CINCO","NOTA_ADMICRO","NAP_CINCO",
              "NOTA_TFM","VECES_ADMICRO","NAC_CINCO","PROMEDIO_SEIS","NOTA_COMANA","NOTA_MOYGE","NOTA_CCON","VECES_COMANA","VECES_DDM","NOTAS_SYS",
              "PROMEDIO_SIETE","NOTA_SISDIN","NOTA_ELECPOT","NOTA_COMDIG","NOTA_FDS","NCC_SIETE","PROMEDIO_OCHO","NOTA_STG","NOTA_C_UNO","NOTA_INSIND","NAP_OCHO","NOTA_INGECO","PROMEDIO_NUEVE"]
    }
}



class InputData(BaseModel):
    carrera: str
    anio: str

@app.post("/procesar_datos/")
async def procesar_datos(data: InputData):
    global carrera, anio
    carrera = data.carrera
    anio = data.anio
    print("Carrera actualizada:", carrera)
    print("anio actualizado:", anio)
    cargar_entrenar_modelo()

@app.post("/predict")
def predict(data: dict):
    global mejor_modelo
    if mejor_modelo:
        try:
            prediction = int(mejor_modelo.predict([list(data.values())])[0])
            return {"prediction": prediction}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=404, detail="No hay modelo entrenado")
    
def cargar_datos(carrera, anio):
    ruta_archivo = f'C:/Users/URIELDARIO/Desktop/MODULO_ANALITICA_PREDICTIVA/DATOS/{carrera}Anio{anio}.csv'
    datos = pd.read_csv(ruta_archivo, sep=";")
    return datos

def transformacion_johnson(X):
    transformador_johnson = PowerTransformer(method='yeo-johnson', standardize=True).fit(X)
    datos_transformados = transformador_johnson.transform(X)
    datos_transformados_df = pd.DataFrame(data=datos_transformados, columns=X.columns)
    return datos_transformados_df

def numero_a_letras(numero):
    numeros_letras = ['cero', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez']
    return numeros_letras[int(numero)]

def cargar_entrenar_modelo():
    global mejor_modelo

    if anio == "1":
        rendimiento_columna = 'PROMEDIO_TRES'
        print(rendimiento_columna)
    elif anio == "2":
        rendimiento_columna = 'PROMEDIO_CINCO'
        print(rendimiento_columna)
    elif anio == "3":
        rendimiento_columna = 'PROMEDIO_NUEVE'
        print(rendimiento_columna)
    else:
        raise ValueError("anio no válido. Solo se permiten los anios 1, 2 y 3.")

    try:
        datos = cargar_datos(carrera, anio)
        columnas_filtradas = variables_por_carrera[carrera][anio]
        df = datos[columnas_filtradas].astype(int)
        anio_en_letras = numero_a_letras(anio)
        X = df.loc[:, ~df.columns.str.contains(rendimiento_columna)]
        Y = df.loc[:, df.columns.str.contains(rendimiento_columna)]
        X_transformado = transformacion_johnson(X)
        X_trn, X_tst, Y_trn, Y_tst = train_test_split(X_transformado, Y, test_size=0.3, random_state=2)
        modelo_knn, r2_knn, mejores_hiperparametros_knn = entrenar_modelo_knn_con_transformacion(X_trn, Y_trn, X_tst, Y_tst)
        modelo_svc, r2_svc, mejores_hiperparametros_svc= entrenar_modelo_svc_con_transformacion(X_trn, Y_trn, X_tst, Y_tst)
        modelo_tree, r2_tree, mejores_hiperparametros_tree=entrenar_modelo_tree_con_transformacion(X_trn, Y_trn, X_tst, Y_tst)
        modelo_gaussian, r2_gaussian, mejores_hiperparametros_gaussian=entrenar_modelo_gaussian_con_transformacion(X_trn, Y_trn, X_tst, Y_tst)
        modelo_LDA, r2_LDA, mejores_hiperparametros_LDA=entrenar_modelo_LDA_con_transformacion(X_trn, Y_trn, X_tst, Y_tst)
        modelo_BG, r2_BG, mejores_hiperparametros_BG=entrenar_modelo_BG_con_transformacion(X_trn, Y_trn, X_tst, Y_tst, mejores_hiperparametros_tree)
        modelo_random, r2_random, mejores_hiperparametros_random=entrenar_modelo_random_con_transformacion(X_trn, Y_trn, X_tst, Y_tst)
        modelo_extra, r2_extra, mejores_hiperparametros_extra= entrenar_modelo_extra_con_transformacion(X_trn, Y_trn, X_tst, Y_tst)
        modelo_ADA, r2_ADA, mejores_hiperparametros_ADA= entrenar_modelo_ADA_con_transformacion(X_trn, Y_trn, X_tst, Y_tst)
        modelo_GD, r2_GD, mejores_hiperparametros_GD= entrenar_modelo_GD_con_transformacion(X_trn, Y_trn, X_tst, Y_tst)
        modelo_XB, r2_XB, mejores_hiperparametros_XB= entrenar_modelo_XB_con_transformacion(X_trn, Y_trn, X_tst, Y_tst)
        modelo_CB, r2_CB, mejores_hiperparametros_CB=entrenar_modelo_CB_con_transformacion(X_trn, Y_trn, X_tst, Y_tst)
        modelo_LIGHT, r2_LIGHT, mejores_hiperparametros_LIGHT= entrenar_modelo_LIGHT_con_transformacion(X_trn, Y_trn, X_tst, Y_tst)
        
        modelo_voting, r2_voting= entrenar_modelo_voting_con_transformacion(X_trn, Y_trn, X_tst, Y_tst,
                                                mejores_hiperparametros_GD,
                                                mejores_hiperparametros_tree,
                                                mejores_hiperparametros_ADA,
                                                mejores_hiperparametros_extra,
                                                mejores_hiperparametros_random,
                                                mejores_hiperparametros_BG,
                                                mejores_hiperparametros_XB)
        modelo_stacking_lineal, r2_stacking_lineal, mejores_hiperparametros_stacking_lineal=entrenar_modelo_stacking_lineal_con_transformacion(X_trn, Y_trn, X_tst, Y_tst,
                                                mejores_hiperparametros_tree,
                                                mejores_hiperparametros_ADA,
                                                mejores_hiperparametros_extra,
                                                mejores_hiperparametros_random)
        modelo_stacking_nolineal, r2_stacking_nolineal, mejores_hiperparametros_stacking_nolineal=entrenar_modelo_stacking_nolineal_con_transformacion(X_trn, Y_trn, X_tst, Y_tst,
                                                mejores_hiperparametros_tree,
                                                mejores_hiperparametros_ADA,
                                                mejores_hiperparametros_extra,
                                                mejores_hiperparametros_random)
        modelo_super_aprendiz, r2_super_aprendiz=entrenar_modelo_super_aprendiz_con_transformacion(X_trn, Y_trn, X_tst, Y_tst,
                                mejores_hiperparametros_GD,
                            mejores_hiperparametros_tree,mejores_hiperparametros_extra,
                            mejores_hiperparametros_random)
        modelo_super_aprendiz_dos_capas, r2_super_aprendiz_dos_capas=entrenar_modelo_super_aprendiz_dos_capas(X_trn, Y_trn, X_tst, Y_tst, mejores_hiperparametros_tree,
                                            mejores_hiperparametros_extra,mejores_hiperparametros_random)

        modelos = {
            'KNeighbors': (modelo_knn, r2_knn),
            'SVC':(modelo_svc, r2_svc),
            'DecisionTree':(modelo_tree, r2_tree),
            'NaiveBayes':(modelo_gaussian,r2_gaussian),
            'LDA': (modelo_LDA, r2_LDA),
            'Bagging':(modelo_BG, r2_BG),
            'RandomForest': (modelo_random, r2_random),
            'Extratrees': (modelo_extra, r2_extra),
            'ADA':(modelo_ADA, r2_ADA),
            'GradientBoosting': (modelo_GD, r2_GD),
            'XGB': (modelo_XB, r2_XB),
            'CatBoost': (modelo_CB, r2_CB),
            'LIGHT': (modelo_LIGHT, r2_LIGHT),
            'Voting':(modelo_voting, r2_voting),
            'Stacking_Lineal':(modelo_stacking_lineal, r2_stacking_lineal),
            'Stacking_noLineal':(modelo_stacking_nolineal, r2_stacking_nolineal),
            'Super_Aprendiz': (modelo_super_aprendiz,r2_super_aprendiz),
            'Super_Aprendiz_dos_Capas':(modelo_super_aprendiz_dos_capas,r2_super_aprendiz_dos_capas)
        }
        mejor_modelo_nombre = max(modelos, key=lambda x: modelos[x][1])
        mejor_modelo, mejor_precision = modelos[mejor_modelo_nombre]
        print("Mejor modelo:", mejor_modelo_nombre)
        print("Exactitud del modelo:", mejor_precision)
    except Exception as e:
        print("Error al cargar y entrenar el modelo:", e)
        
def entrenar_modelo_knn_con_transformacion(X_trn, Y_trn, X_tst, Y_tst):
    parameters = {
        'n_neighbors': [i for i in range(1, 18, 1)],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'algorithm': ['auto'],
        'p': [i for i in range(1, 6)],
        'weights': ['uniform']
    }
    modelo = KNeighborsRegressor()
    semilla = 5
    num_folds = 10
    kfold =StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica = 'neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejores_hiperparametros_knn = grid_resultado.best_params_
    mejor_modelo = KNeighborsRegressor(**grid_resultado.best_params_)
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_knn

def entrenar_modelo_svc_con_transformacion(X_trn, Y_trn, X_tst, Y_tst):
    parameters = { 'kernel':  ['rbf', 'poly', 'sigmoid','linear'], 
            'C': [i/10000 for i in range(8,12,1)],
            'max_iter':[i for i in range(1,3,1)],
            'gamma' : [i/100 for i in range(90,110,5)]}
    modelo = SVR()
    semilla=5
    num_folds=10
    kfold =StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica = 'neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejores_hiperparametros_svc = grid_resultado.best_params_
    mejor_modelo = SVR(**grid_resultado.best_params_)
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_svc

def entrenar_modelo_tree_con_transformacion(X_trn, Y_trn, X_tst, Y_tst):
    parameters = {          
            'max_depth':[i for i in range(1,7,1)], 
            'min_samples_leaf' : [i for i in range(1,7,1)], 
            'max_features' : [i for i in range(1,7,1)], 
            'splitter': ["best", "random"],
            'random_state': [i for i in range(1,7,1)]}
    modelo = DecisionTreeRegressor()
    semilla=5
    num_folds=10
    kfold = StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica = 'neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejores_hiperparametros_tree = grid_resultado.best_params_
    mejor_modelo = DecisionTreeRegressor(**grid_resultado.best_params_)
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_tree

def entrenar_modelo_gaussian_con_transformacion(X_trn, Y_trn, X_tst, Y_tst):
    parameters = {
        'alpha': [1e-10, 1e-5, 1e-2, 1e-1],
        'n_restarts_optimizer': [0, 1, 2, 3],
        'normalize_y': [True, False],
        'optimizer': ['fmin_l_bfgs_b']
    }
    modelo = GaussianProcessRegressor()
    semilla=5
    num_folds=10
    kfold =StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica = 'neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejor_modelo = GaussianProcessRegressor(**grid_resultado.best_params_)
    mejores_hiperparametros_gaussian=grid_resultado.best_params_
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_gaussian

def entrenar_modelo_LDA_con_transformacion(X_trn, Y_trn, X_tst, Y_tst):
    parameters = { 'solver':  ['svd','lsqr','eigen'],
            'n_components':[1,2,3,4,5,6,7,8,9,10],
            'shrinkage': ['auto', 0.001, 0.01, 0.1, 0.5,1,10,100,1000]
            }
    modelo = LinearDiscriminantAnalysis()
    semilla=5
    num_folds=10
    kfold = StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica= 'neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejor_modelo = LinearDiscriminantAnalysis(**grid_resultado.best_params_)
    mejores_hiperparametros_LDA=grid_resultado.best_params_
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_LDA

def entrenar_modelo_BG_con_transformacion(X_trn, Y_trn, X_tst, Y_tst,mejores_hiperparametros_tree):
    parameters = {'n_estimators': [i for i in range(750,760,5)],
            'max_samples' : [i/100.0 for i in range(70,90,5)],
            'max_features': [i/100.0 for i in range(75,85,5)],
            'bootstrap': [True], 
            'bootstrap_features': [True]}
    base_estimator= DecisionTreeRegressor(**mejores_hiperparametros_tree)
    semilla=5
    modelo = BaggingRegressor(estimator=base_estimator)
    num_folds=10
    kfold = StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica =  'neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejores_hiperparametros_BG=grid_resultado.best_params_
    mejor_modelo = BaggingRegressor(**grid_resultado.best_params_)
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_BG
    
def entrenar_modelo_random_con_transformacion(X_trn, Y_trn, X_tst, Y_tst):
    parameters = { 
                'min_samples_split' : [1, 2 , 3,  4 , 6 , 8 , 10 , 15, 20 ],  
                'min_samples_leaf' : [ 1 , 3 , 5 , 7 , 9, 12, 15 ],
            }
    modelo = RandomForestRegressor()
    semilla=5
    num_folds=10
    kfold = StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica =  'neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejores_hiperparametros_random=grid_resultado.best_params_
    mejor_modelo = RandomForestRegressor(**grid_resultado.best_params_)
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_random

def entrenar_modelo_extra_con_transformacion(X_trn, Y_trn, X_tst, Y_tst):
    parameters = {'min_samples_split' : [i for i in range(1,10,1)], 
                'criterion':('absolute_error', 'squared_error', 'friedman_mse', 'poisson')}
    semilla=5            
    modelo = ExtraTreesRegressor(random_state=semilla, 
                                n_estimators=40,
                                bootstrap=True) 
    num_folds=10
    kfold = StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica ='neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejores_hiperparametros_extra=grid_resultado.best_params_
    mejor_modelo = ExtraTreesRegressor(**grid_resultado.best_params_)
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_extra

def entrenar_modelo_ADA_con_transformacion(X_trn, Y_trn, X_tst, Y_tst):
    parameters = {'learning_rate' : [i/10000.0 for i in range(5,20,5)],
                'n_estimators':[i for i in range(1,50,1)]}
    semilla=5            
    modelo = AdaBoostRegressor(estimator = None,random_state= None) 
    num_folds=10
    kfold = StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica ='neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejores_hiperparametros_ADA=grid_resultado.best_params_
    mejor_modelo = AdaBoostRegressor(**grid_resultado.best_params_)
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_ADA

def entrenar_modelo_GD_con_transformacion(X_trn, Y_trn, X_tst, Y_tst):
    parameters = { 
                'learning_rate' : [0.01, 0.05, 0.1,0.15],
                'loss':('absolute_error', 'squared_error', 'quantile', 'huber'),
                'criterion':['friedman_mse']}
    semilla=5
    modelo = GradientBoostingRegressor(random_state=semilla,
                                    n_estimators= 100,learning_rate= 0.1,max_depth= 2,
                                    min_samples_split= 2, min_samples_leaf= 3,max_features= 2)
    semilla=5
    num_folds=10
    kfold = StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica ='neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejores_hiperparametros_GD=grid_resultado.best_params_
    mejor_modelo = GradientBoostingRegressor(**grid_resultado.best_params_)
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_GD

def entrenar_modelo_XB_con_transformacion(X_trn, Y_trn, X_tst, Y_tst):
    parameters = {'reg_alpha': [0,0.1,0.2,0.3,0.4,0.5],
                'reg_lambda':  [i/1000.0 for i in range(100,150,5)],
                'colsample_bytree': [0.1,0.3, 0.5,0.6,0.7,0.8, 0.9, 1,1.1],
                'max_features':('sqrt','log2')
                }
    semilla=5
    modelo = XGBRegressor(random_state=semilla,subsample =1,max_depth =2)
    num_folds=10
    kfold = StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica ='neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejor_modelo= XGBRegressor(**grid_resultado.best_params_)
    mejores_hiperparametros_XB=grid_resultado.best_params_
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_XB

def entrenar_modelo_CB_con_transformacion(X_trn, Y_trn, X_tst, Y_tst):
    parameters = {} 
    semilla=5
    modelo = CatBoostRegressor(random_state=semilla, verbose =0)
    semilla=5
    num_folds=10
    kfold = StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica ='neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejores_hiperparametros_CB=grid_resultado.best_params_
    mejor_modelo = CatBoostRegressor(verbose=0,**grid_resultado.best_params_)
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_CB

def entrenar_modelo_LIGHT_con_transformacion(X_trn, Y_trn, X_tst, Y_tst):
    parameters = {
    'min_child_samples' : [i for i in range(1, 1000, 100)],'colsample_bytree': [0.6, 0.8, 1.0,1.5],
    'boosting_type': ['gbdt', 'dart', 'goss'],'objective': ['binary', 'multiclass'],'random_state': [42]}
    semilla=5
    modelo = LGBMRegressor(random_state=semilla,                           
                            num_leaves =  10,max_depth = 1, n_estimators = 100,    
                            learning_rate = 0.1 ,class_weight=  None, subsample = 1,
                            colsample_bytree= 1, reg_alpha=  0, reg_lambda = 0,
                            min_split_gain = 0, boosting_type = 'gbdt')
    semilla=5
    num_folds=10
    kfold = StratifiedKFold(n_splits=num_folds, random_state=semilla, shuffle=True)
    metrica ='neg_mean_squared_error'
    grid = GridSearchCV(estimator=modelo, param_grid=parameters, scoring=metrica, cv=kfold, n_jobs=-1)
    grid_resultado = grid.fit(X_trn, Y_trn)
    mejores_hiperparametros_LIGHT=grid_resultado.best_params_
    mejor_modelo = LGBMRegressor(verbose=-1,**grid_resultado.best_params_)
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_LIGHT

def entrenar_modelo_voting_con_transformacion(X_trn, Y_trn, X_tst, Y_tst,
                                                mejores_hiperparametros_GD,
                                                mejores_hiperparametros_tree,
                                                mejores_hiperparametros_ADA,
                                                mejores_hiperparametros_extra,
                                                mejores_hiperparametros_random,
                                                mejores_hiperparametros_BG,
                                                mejores_hiperparametros_XB):
    semilla= 5 
    kfold = StratifiedKFold(n_splits=10, random_state=semilla, shuffle=True)
    modelo1 = GradientBoostingRegressor(**mejores_hiperparametros_GD)
    base_estimator=DecisionTreeRegressor(**mejores_hiperparametros_tree)
    modelo2 = AdaBoostRegressor(**mejores_hiperparametros_ADA)
    modelo3 = ExtraTreesRegressor(**mejores_hiperparametros_extra)
    modelo4 = RandomForestRegressor(**mejores_hiperparametros_random)
    model = DecisionTreeRegressor(**mejores_hiperparametros_tree)
    modelo5 = BaggingRegressor(**mejores_hiperparametros_BG)
    modelo6 = DecisionTreeRegressor(**mejores_hiperparametros_tree)
    modelo7 = XGBRegressor(**mejores_hiperparametros_XB)
    metrica ='neg_mean_squared_error'
    mejor_modelo = VotingRegressor(
    estimators=[('Gradient', modelo1), ('Adaboost', modelo2), 
                                    ('Extratrees', modelo3),('Random Forest',modelo4),
                                    ('Bagging',modelo5),('Decision tree',modelo6),
                                    ('XGB',modelo7)]) 
    mejor_modelo.fit(X_trn, Y_trn)
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2

def entrenar_modelo_stacking_lineal_con_transformacion(X_trn, Y_trn, X_tst, Y_tst,
                                                mejores_hiperparametros_tree,
                                                mejores_hiperparametros_ADA,
                                                mejores_hiperparametros_extra,
                                                mejores_hiperparametros_random):
    semilla= 5 
    kfold = StratifiedKFold(n_splits=10, random_state=semilla, shuffle=True)
    base_estimator=DecisionTreeRegressor(**mejores_hiperparametros_tree)
    modelo2 = AdaBoostRegressor(**mejores_hiperparametros_ADA)
    modelo3 = ExtraTreesRegressor(**mejores_hiperparametros_extra)
    modelo4 = RandomForestRegressor (**mejores_hiperparametros_random)
    model = DecisionTreeRegressor(**mejores_hiperparametros_tree)
    modelo5 = DecisionTreeRegressor(**mejores_hiperparametros_tree)
    estimador_final = LinearRegression()
    metrica ='neg_mean_squared_error'
    mejor_modelo = StackingRegressor(
    estimators=[ ('Adaboost', modelo2), ('Extratrees', modelo3),
                ('Random Forest',modelo4),
                ('Decision tree',modelo5)
                ], final_estimator=estimador_final) 
    mejor_modelo.fit(X_trn, Y_trn)
    mejores_hiperparametros_stacking_lineal=mejor_modelo.get_params
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_stacking_lineal

def entrenar_modelo_stacking_nolineal_con_transformacion(X_trn, Y_trn, X_tst, Y_tst,
                                                mejores_hiperparametros_tree,
                                                mejores_hiperparametros_ADA,
                                                mejores_hiperparametros_extra,
                                                mejores_hiperparametros_random):
    semilla= 5 
    kfold = StratifiedKFold(n_splits=10, random_state=semilla, shuffle=True)
    base_estimator=DecisionTreeRegressor(**mejores_hiperparametros_tree)
    modelo2 = AdaBoostRegressor(**mejores_hiperparametros_ADA)
    modelo3 = ExtraTreesRegressor(**mejores_hiperparametros_extra)
    modelo4 = RandomForestRegressor (**mejores_hiperparametros_random)
    model = DecisionTreeRegressor(**mejores_hiperparametros_tree)
    modelo5 = DecisionTreeRegressor(**mejores_hiperparametros_tree)
    estimador_final = ExtraTreesRegressor()
    metrica ='neg_mean_squared_error'
    mejor_modelo = StackingRegressor(
    estimators=[ ('Adaboost', modelo2), ('Extratrees', modelo3),
                ('Random Forest',modelo4),
                ('Decision tree',modelo5)
                ], final_estimator=estimador_final) 
    mejor_modelo.fit(X_trn, Y_trn)
    mejores_hiperparametros_stacking_nolineal=mejor_modelo.get_params
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2, mejores_hiperparametros_stacking_nolineal

def entrenar_modelo_super_aprendiz_con_transformacion(X_trn, Y_trn, X_tst, Y_tst,
                            mejores_hiperparametros_GD,
                            mejores_hiperparametros_tree,mejores_hiperparametros_extra,
                            mejores_hiperparametros_random):
    semilla = 5 
    kfold = StratifiedKFold(n_splits=10, random_state=semilla, shuffle=True)
    modelo1 = ExtraTreesRegressor(**mejores_hiperparametros_extra)
    modelo2 = RandomForestRegressor(**mejores_hiperparametros_random)
    model = DecisionTreeRegressor(**mejores_hiperparametros_tree)
    modelo4 = DecisionTreeRegressor(**mejores_hiperparametros_tree)
    modelo5 = GradientBoostingRegressor(**mejores_hiperparametros_GD) 
    estimadores = [('Extratrees', modelo1), 
                ('Random Forest', modelo2), 
                ('Decision tree', modelo4),
                ('Gradient',modelo5)]
    mejor_modelo = SuperLearner(folds=10, random_state=semilla, verbose=2)
    mejor_modelo.add(estimadores)
    estimador_final = ExtraTreesRegressor(n_estimators=100, max_features=None,
                                        bootstrap=False, max_depth=11, min_samples_split=4, 
                                        min_samples_leaf=1)
    mejor_modelo.add_meta(estimador_final)
    mejor_modelo.fit(X_trn, Y_trn)
    mejores_hiperparametros_super_learner=mejor_modelo.get_params
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2

def entrenar_modelo_super_aprendiz_dos_capas(X_trn, Y_trn, X_tst, Y_tst, 
                                            mejores_hiperparametros_tree,mejores_hiperparametros_extra,
                                            mejores_hiperparametros_random):
    semilla = 5 
    kfold = StratifiedKFold(n_splits=10, random_state=semilla, shuffle=True)
    modelo1 = ExtraTreesRegressor(**mejores_hiperparametros_extra)
    modelo2 = RandomForestRegressor(**mejores_hiperparametros_random)
    model = DecisionTreeRegressor(**mejores_hiperparametros_tree)
    modelo3 = DecisionTreeRegressor(**mejores_hiperparametros_tree)
    estimadores = [('Extratrees', modelo1), 
                ('Random Forest', modelo2), 
                ('Decision tree', modelo3)]
    mejor_modelo = SuperLearner(folds=10, random_state=semilla, verbose=2)
    mejor_modelo.add(estimadores)
    mejor_modelo.add(estimadores)
    estimador_final = ExtraTreesRegressor(n_estimators=100, max_features=None,
                                        bootstrap=False, max_depth=11, min_samples_split=4, 
                                        min_samples_leaf=1)
    mejor_modelo.add_meta(estimador_final)
    mejor_modelo.fit(X_trn, Y_trn)
    mejores_hiperparametros_superaprendiz_dos_capas=mejor_modelo.get_params
    predicciones = mejor_modelo.predict(X_tst)
    r2 = r2_score(Y_tst, predicciones)
    return mejor_modelo, r2

@app.get("/")
def read_root():
    return {"message": "API para predecir calificación de estudiantes"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)