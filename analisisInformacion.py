import spacy
 
# Cargar el modelo de procesamiento de lenguaje natural de spaCy
nlp = spacy.load("es_core_news_sm")  # Puedes usar un modelo específico para tu idioma
 
# Texto de ejemplo: Descripción del puesto y experiencia del candidato
descripcion_puesto = "Buscamos un desarrollador de Python con experiencia en desarrollo web y al menos 2 años de experiencia."
experiencia_candidato = "Desarrollador de Python con 3 años de experiencia en desarrollo web."
 
# Procesar la descripción del puesto y la experiencia del candidato
doc_puesto = nlp(descripcion_puesto)
doc_candidato = nlp(experiencia_candidato)
 
# Función para evaluar la similitud contextual
def evaluar_similitud_contextual(doc1, doc2):
	similitud = doc1.similarity(doc2)
	return similitud
 
# Evaluar la similitud contextual entre la descripción del puesto y la experiencia del candidato
similitud_contextual = evaluar_similitud_contextual(doc_puesto, doc_candidato)
 
# Establecer un umbral de similitud
umbral_similitud = 0.7  # Puedes ajustar este umbral según tus criterios
 
# Evaluar si la experiencia del candidato es coherente con la descripción del puesto
if similitud_contextual >= umbral_similitud:
	print("La experiencia del candidato es coherente con la descripción del puesto.")
else:
	print("La experiencia del candidato no es coherente con la descripción del puesto.")
 
# Imprimir la similitud contextual (puede ser útil para fines de análisis)
print(f"Similitud Contextual: {similitud_contextual:.2f}")
