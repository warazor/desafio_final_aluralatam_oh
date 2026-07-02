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


import os
import glob
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuración obligatoria de la API Key de Groq
# En local usas tu clave; en GitHub Actions se inyecta desde los secretos automáticamente.
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "TU_API_KEY_DE_GROQ_AQUI")

# Variables de configuración globales
CARPETA_DATOS = "mis_archivos_csv"
DIRECTORIO_VECTOR = "db_vectores"

# ==========================================
# SEGMENTO 1: Lectura de archivos y Vectores
# ==========================================
@st.cache_resource # Evita que se vuelvan a procesar los vectores en cada recarga de la interfaz
def inicializar_base_conocimiento():
    """Lee todos los archivos CSV de una carpeta y los guarda en vectores."""
    # Crear la carpeta si no existe
    if not os.path.exists(CARPETA_DATOS):
        os.makedirs(CARPETA_DATOS)
        # Creamos un CSV de prueba si la carpeta está vacía
        df_ejemplo = pd.DataFrame({"producto": ["Laptop", "Mouse"], "precio": [1000, 25]})
        df_ejemplo.to_csv(f"{CARPETA_DATOS}/ejemplo.csv", index=False)

    # Buscar todos los archivos .csv en la carpeta configurada
    archivos_csv = glob.glob(os.path.join(CARPETA_DATOS, "*.csv"))
    
    todos_los_documentos = []
    for ruta_csv in archivos_csv:
        # Cargar el archivo mediante LangChain
        loader = CSVLoader(file_path=ruta_csv, encoding='utf-8')
        todos_los_documentos.extend(loader.load())
        
    if not todos_los_documentos:
        raise ValueError(f"No se encontraron archivos CSV válidos en la carpeta '{CARPETA_DATOS}'")

    # Generar embeddings y persistir de forma local en la base de datos ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        todos_los_documentos, 
        embeddings, 
        persist_directory=DIRECTORIO_VECTOR
    )
    
    # Retorna el recuperador configurado para extraer los 3 fragmentos más parecidos
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# ==========================================
# SEGMENTO 3: Obtención de respuesta por la IA
# ==========================================
def obtener_respuesta_ia(retriever, pregunta_usuario: str) -> str:
    """Configura la cadena de IA y devuelve la respuesta formulada por Groq."""
    # Instanciar el modelo LLaMA ultra veloz de Groq
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # Prompt con directrices explícitas para el sistema
    system_prompt = (
        "Eres un asistente analítico experto en datos corporativos.\n"
        "Usa exclusivamente el contexto recuperado de los archivos para responder.\n"
        "Si los datos no contienen la respuesta, di que la información no está disponible.\n\n"
        "Contexto:\n{context}"
    )
    prompt = ChatPromptTemplate.from_template(system_prompt + "\n\nPregunta: {question}")

    # Función auxiliar para formatear la salida del recuperador
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Cadena RAG moderna usando sintaxis LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Ejecuta la cadena y retorna el string plano devuelto por la IA
    return rag_chain.invoke(pregunta_usuario)

# ==========================================
# SEGMENTO 2: Interfaz para interactuar (Streamlit)
# ==========================================
def main():
    st.set_page_config(page_title="RAG Multi-CSV Chatbot", page_icon="📊")
    st.title("📊 Chatbot de IA sobre múltiples archivos CSV")
    st.subheader("Segmentación con LangChain + Groq + ChromaDB")

    # Inicializar componentes del Segmento 1
    try:
        with st.spinner("Indexando los archivos CSV de la carpeta en vectores..."):
            retriever = inicializar_base_conocimiento()
        st.success(f"Base de datos vectorial cargada y sincronizada con la carpeta '{CARPETA_DATOS}'!")
    except Exception as e:
        st.error(f"Error al inicializar los vectores: {e}")
        return

    # Historial de chat integrado en la interfaz gráfica
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes previos del historial de la sesión
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Entrada de texto interactiva para que el usuario escriba la pregunta
    if pregunta := st.chat_input("Escribe tu pregunta sobre los datos de tus archivos CSV aquí..."):
        # Mostrar la pregunta en la pantalla del chat
        with st.chat_message("user"):
            st.write(pregunta)
        st.session_state.messages.append({"role": "user", "content": pregunta})

        # Invocar la función del Segmento 3 para obtener y devolver la respuesta de la IA
        with st.chat_message("assistant"):
            with st.spinner("La IA está consultando la base de datos vectorial..."):
                respuesta_final = obtener_respuesta_ia(retriever, pregunta)
                st.write(respuesta_final)
        
        # Guardar la respuesta de la IA en la sesión
        st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

if __name__ == "__main__":
    main()