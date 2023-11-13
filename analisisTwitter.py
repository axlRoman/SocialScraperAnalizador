import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from langdetect import detect
from googletrans import Translator

# Supongamos que tienes un archivo de texto llamado 'datos_redes_sociales.txt' con un mensaje por línea
with open('datos_redes_sociales.txt', 'r', encoding='utf-8') as file:
    mensajes = file.readlines()

# Función para traducir un mensaje a inglés
def traducir_a_ingles(mensaje):
    translator = Translator()
    try:
        translation = translator.translate(mensaje, src=detect(mensaje), dest='en')
        return translation.text
    except:
        return mensaje

# Traducir todos los mensajes a inglés
mensajes_traducidos = [traducir_a_ingles(mensaje) for mensaje in mensajes]

# Crear un DataFrame con los mensajes traducidos
datos = pd.DataFrame({'Texto': mensajes_traducidos})

# Análisis de sentimientos
datos['Sentimiento'] = datos['Texto'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Visualizar distribución de sentimientos
plt.figure(figsize=(10, 6))
plt.hist(datos['Sentimiento'], bins=30, edgecolor='black')
plt.title('Distribución de Sentimientos')
plt.xlabel('Sentimiento')
plt.ylabel('Frecuencia')
plt.show()


# Generar una nube de palabras
#text = ' '.join(datos['Texto'])
#wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Visualizar la nube de palabras
#plt.figure(figsize=(10, 6))
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis('off')
#plt.title('Nube de Palabras')
#plt.show()
