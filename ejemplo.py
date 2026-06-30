import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Configura tu API Key de Groq
os.environ["GROQ_API_KEY"] = ""

# --- CONFIGURACIONES ---
ruta_archivo = os.path.join("base_conocimiento_csv", "base_inicial.csv")

directorio_vector = "db_vectores"

# 2. Leer el CSV con Pandas (Opcional: para validaciones o lectura directa)
df = pd.read_csv(ruta_archivo)
print(f"Archivo CSV cargado. Total de filas: {len(df)}")

# 3. Cargar el documento y crear los Vectores (Vector Store)
# CSVLoader lee el CSV y convierte cada fila en un documento de texto
loader = CSVLoader(file_path=ruta_archivo, encoding='utf-8')
docs = loader.load()

# Usamos un modelo local ligero de HuggingFace para generar los embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Guardamos los vectores en una base de datos local (ChromaDB)
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=directorio_vector)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Recupera las 3 filas más relevantes

# 4. Configurar el LLM con Groq
# Usamos un modelo rápido de Groq, como llama-3.1-8b-instant
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 5. Crear la cadena de preguntas (Q&A Chain)
system_prompt = (
    "Eres un asistente experto para responder preguntas sobre datos.\n"
    "Usa los siguientes fragmentos de información recuperada del CSV para responder la pregunta.\n"
    "Si no sabes la respuesta, di que no la sabes. Sé conciso.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Interacción con el usuario (Ingreso de preguntas)
print("\n--- Sistema iniciado. Escribe 'salir' para terminar ---")
while True:
    pregunta_usuario = input("\nIngresa tu pregunta (escribe salir para terminar): ")
    if pregunta_usuario.lower() == 'salir':
        break
    
    # El LLM responde basado en la información vectorial
    response = rag_chain.invoke({"input": pregunta_usuario})
    print("\nRespuesta:")
    print(response["answer"])
