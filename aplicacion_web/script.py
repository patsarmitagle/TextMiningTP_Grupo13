# Script final de la aplicación, devuelve json en lugar de un csv para mejorar la oportunidad de integración
# encoding: utf-8 
"""
TP_Text_Mining_Final.ipynb
Requiere:
# !pip install spacy
# !python -m spacy download es_core_news_sm
# !pip install transformers
# !pip install lightgbm


# Importación de librerías
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
import sys

# Check if a file path parameter is provided
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    print("Please provide a file path as a parameter.")
    sys.exit(1)

if len(sys.argv) > 2:
    os.chdir(sys.argv[2])
else:
    print("Please provide a path as execution path parameter.")
    sys.exit(1)

import re
import string
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('all')

from transformers import pipeline

# import spacy
# nlp = spacy.load('es_core_news_sm')

"""# Cargamos el dataset"""

df = pd.read_csv(file_path, sep=';', encoding='utf-8')
pd.set_option('display.max_columns', None)

model_path = 'model.pkl'
model = joblib.load(model_path)


"""# Preparación de datos
## Unificación de comentarios
"""

df_grouped = df.groupby('IdTracking').apply(
    lambda x: pd.Series({
        'Des': x['Des'].iloc[0],  # Cargar 'Des' solo una vez (el primer valor del grupo)
        'Des_Combined': ' '.join(x['Des.1'].astype(str)),
        'IdTipoTracking': x['IdTipoTracking'].iloc[0],
        'IdPrioridad': x['IdPrioridad'].iloc[0],
        'IdTipoTarea': x['IdTipoTarea'].iloc[0],
        'IdComponente': x['IdComponente'].iloc[0],
        'IdPropuesta': x['IdPropuesta'].iloc[0],
        'IdCuenta': x['IdCuenta'].iloc[0],
        'FecAlt': x['FecAlt'].iloc[0],
        'IdUsuarioAlt': x['IdUsuarioAlt'].iloc[0],
        'FecAsi': x['FecAsi'].iloc[0],
        'IdUsuarioAsi': x['IdUsuarioAsi'].iloc[0],
        'IdEstado': x['IdEstado'].iloc[0],
        'IdTipoEstado': x['IdTipoEstado'].iloc[0],
        'HorEst': x['HorEst'].iloc[0],
        'MinEst': x['MinEst'].iloc[0],
        'Inter': x['Inter'].iloc[0],
        'Exter': x['Exter'].iloc[0],
        'IdHito': x['IdHito'].iloc[0],
        'IdVersion': x['IdVersion'].iloc[0],
        'FecVto': x['FecVto'].iloc[0],
        'HorDes': x['HorDes'].iloc[0],
        'IdProducto': x['IdProducto'].iloc[0],
        'IdCategoria': x['IdCategoria'].iloc[0],
        'AplMan': x['AplMan'].iloc[0],
        'Hordesdes': x['Hordesdes'].iloc[0],
        'Hordestes': x['Hordestes'].iloc[0],
        'Horesttes': x['Horesttes'].iloc[0],
        'Horestdes': x['Horestdes'].iloc[0],
        'HorCli': x['HorCli'].iloc[0],
        'ConCan': x['ConCan'].iloc[0],
        'Obs': x['Obs'].iloc[0],
        'IdTareaProyecto': x['IdTareaProyecto'].iloc[0],
        'IdSubtipoProducto': x['IdSubtipoProducto'].iloc[0],
        'IdTrackingRelacionado': x['IdTrackingRelacionado'].iloc[0],
        'IdComplejidad': x['IdComplejidad'].iloc[0],
        'FecUltMod': x['FecUltMod'].iloc[0],
        'IdServidor': x['IdServidor'].iloc[0],
        'Facturable': x['Facturable'].iloc[0]
    })
).reset_index()


"""## Preparación del texto"""

def limpiar_texto(text):
  # Convertir a minúsculas
  text = text.lower()
  # Eliminar signos de puntuación
  text = text.translate(str.maketrans('', '', string.punctuation))
  # Reemplazar tildes
  text = text.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('Á', 'A').replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U')
  # Eliminar números y caracteres especiales
  text = re.sub(r'[^a-zA-Z\s]', '', text)
  # Eliminar espacios en blanco adicionales
  text = ' '.join(text.split())
  # Tokenizar el texto
  tokens = word_tokenize(text, language='spanish')
  # Eliminar stop words
  stop_words = set(stopwords.words('spanish'))
  tokens = [word for word in tokens if not word in stop_words]
  # Lematizar las palabras
  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  # Unir los tokens de nuevo en una cadena
  processed_text = ' '.join(tokens)
  return processed_text

df_grouped_copia = df_grouped.copy()

df_grouped['Des'] = df_grouped['Des'] + ' ' + df_grouped['Des_Combined']
df_grouped['Des'] = df_grouped['Des'].apply(limpiar_texto)
df_grouped = df_grouped.drop(columns=['Des_Combined'])

"""### Realizamos un análisis de Sentimientos de la variable Des"""

# Cargar el modelo de análisis de sentimientos
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Función para analizar el sentimiento de un texto
def analyze_sentiment(text):
  #Se trunca el texto a un máximo de 512 tokens para que se ajuste al modelo BERT.
  result = sentiment_pipeline(text[:512])
  return result[0]['label'], result[0]['score']

# Aplicar la función al DataFrame
df_grouped[['sentiment_label', 'sentiment_score']] = df_grouped['Des'].apply(lambda x: pd.Series(analyze_sentiment(x)))

# Mapear las etiquetas de sentimiento a valores numéricos
sentiment_mapping = {'1 star': 1, '2 stars': 2, '3 stars': 3, '4 stars': 4, '5 stars': 5}
df_grouped['des_sentimiento'] = df_grouped['sentiment_label'].map(sentiment_mapping)

"""### Realizamos una Clasificación de texto de la variable Des"""

# Cargar el pipeline para clasificación de texto con BERT
classifier = pipeline('text-classification', model='dccuchile/bert-base-spanish-wwm-uncased')

# Función para clasificar el texto
def classify_text(text):
  result = classifier(text[:775])
  return result[0]['label'], result[0]['score']

# Aplicar la función al DataFrame
df_grouped[['classification_label', 'classification_score']] = df_grouped['Des'].apply(lambda x: pd.Series(classify_text(x)))
df_grouped['des_clasificacion'] = df_grouped['classification_label'].map({'LABEL_0': False, 'LABEL_1': True})

"""## Agregamos la variable FecAlt_diff_minutes que representa el tiempo en minutos desde que un issue fue creado hasta su primera atención"""

# Convertir las columnas 'FecAlt' y 'FecAlt.1' a formato datetime
df['FecAlt'] = pd.to_datetime(df['FecAlt'], format='%d/%m/%Y %H:%M')
df['FecAlt.1'] = pd.to_datetime(df['FecAlt.1'], format='%d/%m/%Y %H:%M')

# Find the minimum FecAlt for each IdTracking in df_grouped
min_fecalt = df.groupby('IdTracking')['FecAlt.1'].min().reset_index()
df_diff_min = df_grouped
# Merge the minimum FecAlt values back into the original DataFrame
df_diff_min = df_diff_min.merge(min_fecalt, on=['IdTracking'], how='left')

#Convertimos las columnas a fechas
df_diff_min['FecAlt'] = pd.to_datetime(df_diff_min['FecAlt'], format='%d/%m/%Y %H:%M')
df_diff_min['FecAlt.1'] = pd.to_datetime(df_diff_min['FecAlt.1'], format='%d/%m/%Y %H:%M')

# Calculate the difference in minutes between FecAlt and FecAlt.1
df_grouped['FecAlt_diff_minutes'] = (df_diff_min['FecAlt.1'] - df_diff_min['FecAlt']).dt.total_seconds() / 60


"""### Eliminamos las columnas no deseadas"""

# Eliminar las columnas especificadas
df_grouped = df_grouped.drop(columns=[
      'IdTipoTarea', 'IdComponente', 'IdPropuesta', 'FecAlt','IdUsuarioAlt', 'FecAsi', 'IdUsuarioAsi', 'IdTipoEstado',
      'MinEst', 'IdHito', 'IdVersion', 'FecVto', 'HorDes', 'AplMan', 'Hordesdes',
      'Hordestes', 'Horesttes', 'Horestdes', 'HorCli', 'Obs','IdTareaProyecto', 'IdSubtipoProducto', 'IdTrackingRelacionado',
      'FecUltMod', 'IdServidor', 'sentiment_label','sentiment_score', 'classification_score', 'classification_label'
])

# Mostrar las primeras filas del DataFrame para verificar el resultado

"""## Corregimos los valores erróneos de la variable Facturable"""

# Reemplazamos los #n/a por 0
df_grouped['Facturable'] = df_grouped['Facturable'].replace('#N/D', 0)

"""## Vectorizamos la variable Des"""

# Vectorización con TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 1), tokenizer=word_tokenize)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_grouped['Des'])

# Convertir la matriz TF-IDF a un DataFrame de pandas
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Concatenar el DataFrame de TF-IDF con el DataFrame original
df_grouped = pd.concat([df_grouped, tfidf_df], axis=1)


"""## Entrenamos el modelo"""

df_grouped['Facturable'] = df_grouped['Facturable'].astype(int)
# Import the necessary library

# Define features and target variable
X = df_grouped.drop(['Facturable', 'Des'], axis=1)
y = df_grouped['Facturable']

# Predecimos el subconjunto Test
predictions = model.predict(X)

# Convert probabilities to binary predictions
predictions_binary = [1 if p > 0.50 else 0 for p in predictions]

df_grouped['Facturable_Prediction'] = predictions_binary

# Mostrar el resultado
df_final = df_grouped[['IdTracking', 'Facturable_Prediction']]
df_final = df_final.rename(columns={'Facturable_Prediction': 'Facturable'})
# Unir los datos basados en IdTracking
df_final = df_final.merge(df_grouped_copia[['IdTracking', 'Des']], on='IdTracking', how='left')
# Renombrar la columna 'Des' a 'Descripcion'
df_final = df_final.rename(columns={'Des': 'Descripcion'})
# Agregar una columna con la descripción de facturable
df_final['Facturable_Descripcion'] = df_final['Facturable'].apply(lambda x: 'Si' if x == 1 else 'No')

# Convert the result to JSON format
result_json = df_final.to_json(orient='records')

# Print the JSON result
print(result_json)