import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuración de API Keys
os.environ["GROQ_API_KEY"] = ""

# 2. Leer el archivo CSV con Pandas
ruta_csv = "/base_conocimiento_csv/base_inicial.csv" # Cambiar por tu archivo
df = pd.read_csv(ruta_csv)

# Convertir el DataFrame a texto plano para que el LLM lo entienda
# Esta línea une todas las columnas fila por fila, creando un bloque de texto por registro.
textos = df.apply(lambda row: " | ".join([f"{col}: {row[col]}" for col in df.columns]), axis=1).tolist()

# 3. Guardar información en un Vector Store
# Convertimos el texto a embeddings utilizando HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Procesando y guardando información en la base de datos vectorial...")
db = Chroma.from_texts(textos, embeddings, persist_directory="./chroma_db")

# 4. Configurar el LLM con Groq
# Puedes usar modelos como 'llama3-70b-8192' o 'mixtral-8x7b-32768'
llm = ChatGroq(model="llama3-70b-8192", temperature=0.2)

# 5. Crear el sistema de preguntas y respuestas (RAG)
system_prompt = (
    "Eres un asistente de inteligencia artificial muy útil. "
    "Usa los siguientes fragmentos de contexto recuperados para responder la pregunta. "
    "Si no sabes la respuesta, di que no la sabes. "
    "Sé conciso y profesional en tus respuestas.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever(search_kwargs={"k": 5}) # Recupera las 5 filas más relevantes
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Interacción con el usuario (Bucle de preguntas)
print("\n¡Chatbot listo! Escribe 'salir' para terminar.\n")
while True:
    pregunta_usuario = input("Ingresa tu pregunta: ")
    if pregunta_usuario.lower() == 'salir':
        break
    
    # El modelo responde basándose en el contexto vectorial
    response = rag_chain.invoke({"input": pregunta_usuario})
    
    print("\nRespuesta:")
    print(response["answer"])
    print("-" * 50)