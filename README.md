# Clasificación de Trackings

## Descripción

Este proyecto de Python realiza la clasificación de trackings (incidencias) para determinar si son facturables o no. El proceso incluye preprocesamiento de texto, análisis de sentimientos, clasificación de texto y entrenamiento de un modelo de aprendizaje automático.

## Dependencias

Las siguientes bibliotecas son necesarias para ejecutar el script:

* pandas
* numpy
* scikit-learn
* nltk
* transformers
* lightgbm
* spacy
* es-core-news-sm

## Uso

1.  Asegúrate de que todas las dependencias estén instaladas.
2.  Descarga el conjunto de datos 'trackings.csv' y colócalo en el mismo directorio que el script.
3.  Ejecuta el script.

## Resultados

El script generará un modelo entrenado para la clasificación de trackings. También mostrará métricas de evaluación del modelo, incluyendo el área bajo la curva ROC y la matriz de confusión.

## Notas

* El script utiliza un modelo BERT para el análisis de sentimientos y la clasificación de texto.
* El modelo de aprendizaje automático utilizado es LightGBM.
* Se realiza un preprocesamiento de texto para limpiar y preparar los datos para el modelado.
* El script incluye comentarios explicativos para cada paso del proceso.

## Autores
* Marcos Ludueña
* Fernando Raco
* Macarena Bello Vargas 
* Nerio Espina
* Patricia Sarmiento 

## Licencia
Este proyecto está disponible bajo la [Licencia MIT](LICENSE).

