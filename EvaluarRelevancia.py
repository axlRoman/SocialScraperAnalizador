import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Datos de ejemplo: candidatos y requisitos del trabajo
candidatos = pd.DataFrame({
    'ID': [1, 2, 3],
    'Experiencia': [
        "Desarrollador de Python con 3 años de experiencia en desarrollo web.",
        "Ingeniero de software con experiencia en Java y C++.",
        "Programador junior con conocimientos de Python y JavaScript."
    ]
})

requisitos_puesto = "Buscamos un desarrollador de Python con experiencia en desarrollo web y al menos 2 años de experiencia."

# Preprocesamiento de texto
tfidf_vectorizer = TfidfVectorizer()
documentos = candidatos['Experiencia'].tolist() + [requisitos_puesto]
matriz_tfidf = tfidf_vectorizer.fit_transform(documentos)

# Calcular similitud de coseno entre la descripción del trabajo y las experiencias de los candidatos
similitudes = cosine_similarity(matriz_tfidf[:-1], matriz_tfidf[-1:])

# Calcular una puntuación de relevancia para cada candidato
puntuaciones_relevancia = similitudes.flatten()

# Asegurarse de que la longitud de puntuaciones_relevancia coincida con la longitud de candidatos
candidatos['Puntuacion_Relevancia'] = puntuaciones_relevancia[:len(candidatos)]

# Imprimir los resultados
print(candidatos[['ID', 'Experiencia', 'Puntuacion_Relevancia']])
