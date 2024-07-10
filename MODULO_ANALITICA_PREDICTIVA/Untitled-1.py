import warnings
warnings.filterwarnings('ignore')
from numpy import set_printoptions
from pandas.plotting import scatter_matrix
import pandas as pd 
pd.options.display.max_columns=None
import seaborn as sns 
from pandas import read_csv
import io
import os
import os.path
from unidecode import unidecode
from urllib.error import HTTPError
from fastapi import FastAPI, Request
from typing import List
from pydantic import BaseModel
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, normalized_mutual_info_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

app = FastAPI()
carrera = ""
semestre = ""
modelos_seleccionados=[]

class InputData(BaseModel):
    
    carrera: str
    semestre: str
    modelos_seleccionados: List[str]

@app.post("/procesar_datos/")
async def procesar_datos(data: InputData):
    global modelos_seleccionados, carrera, semestre
    modelos_seleccionados= data.modelos_seleccionados
    carrera = data.carrera
    semestre = data.semestre
    print("Diccionario seleccion actualizado:", modelos_seleccionados)
    print("Carrera actualizada:", carrera)
    print("Semestre actualizado:", semestre)



    variables_por_carrera = {
    'industrial': {
        '1': ['PG_ICFES', 'CON_MAT_ICFES', 'FISICA_ICFES','QUIMICA_ICFES','IDIOMA_ICFES','LOCALIDAD','RENDIMIENTO_UNO'],
        '2': ['LOCALIDAD_COLEGIO', 'PG_ICFES', 'CON_MAT_ICFES', 'FISICA_ICFES', 'BIOLOGIA_ICFES', 'IDIOMA_ICFES
'LOCALIDAD', 'RENDIMIENTO_UNO', 'RENDIMIENTO_DOS'],
        '3': ['LOCALIDAD_COLEGIO', 'PG_ICFES', 'CON_MAT_ICFES', 'FISICA_ICFES', 'BIOLOGIA_ICFES', 'IDIOMA_ICFES',
              'LOCALIDAD', 'RENDIMIENTO_UNO', 'RENDIMIENTO_DOS']
    },
    'sistemas': {
        '1': ['LOCALIDAD_COLEGIO', 'PG_ICFES', 'CON_MAT_ICFES', 'IDIOMA_ICFES', 'RENDIMIENTO_DOS'],
        '2': ['LOCALIDAD_COLEGIO', 'PG_ICFES', 'CON_MAT_ICFES', 'FISICA_ICFES', 'IDIOMA_ICFES', 'RENDIMIENTO_DOS'],
        '3': ['LOCALIDAD_COLEGIO', 'PG_ICFES', 'CON_MAT_ICFES', 'FISICA_ICFES', 'QUIMICA_ICFES', 'IDIOMA_ICFES',
              'RENDIMIENTO_DOS']
    }
}
    #Cargamos la data en un dataframe
    print("Procesando datos...")
    dataframe = load_data(carrera, semestre)
    print("Data procesada exitosamente.")
    print("Aplicando preprocesamiento...")
    dataframe = preprocess_data(dataframe, carrera, semestre)
    print("Preprocesamiento completado.")
    print("Entrenando modelos...")
    train_models(dataframe, carrera, semestre)
    print("Entrenamiento de modelos completado.")

    return {"message": "Proceso completado exitosamente."}

def load_data(carrera, semestre):
    data = pd.read_csv(f'{save_path}/data_{carrera}_{semestre}.csv')
    return data

def preprocess_data(data, carrera, semestre):
    #Eliminamos columnas innecesarias
    data = data.drop(['Unnamed: 0'], axis=1)
    #Seleccionamos las variables adecuadas para la carrera y el semestre
    selected_variables = variables_por_carrera[carrera][semestre]
    data = data[selected_variables]
    #Rellenamos valores nulos
    data.fillna(data.mean(), inplace=True)
    #Codificamos las variables categóricas
    data = pd.get_dummies(data)
    return data

def train_models(data, carrera, semestre):
    X = data.drop(['RENDIMIENTO_DOS'], axis=1)
    y = data['RENDIMIENTO_DOS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if 'knn' in modelos_seleccionados:
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)
        knn_predictions = knn_model.predict(X_test)
        print("KNN metrics:")
        print("Precision:", precision_score(y_test, knn_predictions))
        print("Recall:", recall_score(y_test, knn_predictions))
        print("F1 Score:", f1_score(y_test, knn_predictions))
    if 'svm' in modelos_seleccionados:
        svm_model = SVC()
        svm_model.fit(X_train, y_train)
        svm_predictions = svm_model.predict(X_test)
        print("SVM metrics:")
        print("Precision:", precision_score(y_test, svm_predictions))
        print("Recall:", recall_score(y_test, svm_predictions))
        print("F1 Score:", f1_score(y_test, svm_predictions))
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

