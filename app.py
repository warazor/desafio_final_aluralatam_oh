import os
import glob
import hashlib
import shutil
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
CARPETA_DATOS = "base_conocimiento_csv"
DIRECTORIO_VECTOR = "db_vectores"
NOMBRE_COLECCION = "base_conocimiento"

# ==========================================
# SEGMENTO 0: Utilidades de la base vectorial
# ==========================================
def _obtener_embeddings():
    """Carga el modelo de embeddings una sola vez."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _cargar_documentos_de_csv(ruta_csv):
    """Carga un CSV como documentos y les asigna un id e info de origen estables.

    Cada fragmento recibe:
      - metadata['source'] = nombre del archivo (permite borrarlo luego)
      - id = 'archivo::indice' (determinista, evita duplicados al re-agregar)
    """
    nombre_archivo = os.path.basename(ruta_csv)
    loader = CSVLoader(file_path=ruta_csv, encoding="utf-8")
    documentos = loader.load()

    ids = []
    for i, doc in enumerate(documentos):
        doc.metadata["source"] = nombre_archivo
        ids.append(f"{nombre_archivo}::{i}")
    return documentos, ids


@st.cache_resource
def obtener_vectorstore():
    """Devuelve la vectorstore Chroma persistente (se crea una sola vez por sesión).

    A diferencia de la versión anterior, NO reconstruye la base en cada recarga:
    abre la colección persistida y, si está vacía, la siembra con los CSV existentes.
    """
    if not os.path.exists(CARPETA_DATOS):
        os.makedirs(CARPETA_DATOS)

    embeddings = _obtener_embeddings()
    vectorstore = Chroma(
        collection_name=NOMBRE_COLECCION,
        embedding_function=embeddings,
        persist_directory=DIRECTORIO_VECTOR,
    )

    # Siembra inicial solo si la colección está vacía
    try:
        vacia = vectorstore._collection.count() == 0
    except Exception:
        vacia = True

    if vacia:
        archivos_csv = glob.glob(os.path.join(CARPETA_DATOS, "*.csv"))
        for ruta_csv in archivos_csv:
            documentos, ids = _cargar_documentos_de_csv(ruta_csv)
            if documentos:
                vectorstore.add_documents(documents=documentos, ids=ids)
        vectorstore.persist()

    return vectorstore


def listar_documentos_indexados(vectorstore):
    """Devuelve la lista de archivos (source) presentes en la base vectorial."""
    try:
        datos = vectorstore.get(include=["metadatas"])
        fuentes = {m.get("source") for m in datos.get("metadatas", []) if m and m.get("source")}
        return sorted(fuentes)
    except Exception:
        return []


def agregar_documento(vectorstore, ruta_csv):
    """Agrega/actualiza un CSV en la base vectorial.

    Primero elimina cualquier versión previa del mismo archivo (para actualizar en
    lugar de duplicar) y luego inserta los fragmentos nuevos.
    """
    nombre_archivo = os.path.basename(ruta_csv)
    # Actualización idempotente: borra la versión anterior si existía
    eliminar_documento(vectorstore, nombre_archivo, borrar_csv=False)

    documentos, ids = _cargar_documentos_de_csv(ruta_csv)
    if not documentos:
        raise ValueError(f"El archivo '{nombre_archivo}' no contiene filas válidas.")

    vectorstore.add_documents(documents=documentos, ids=ids)
    vectorstore.persist()
    return len(documentos)


def eliminar_documento(vectorstore, nombre_archivo, borrar_csv=True):
    """Elimina de la base vectorial todos los fragmentos de un archivo dado.

    Si borrar_csv=True, también elimina el CSV físico de la carpeta de datos.
    """
    vectorstore._collection.delete(where={"source": nombre_archivo})
    vectorstore.persist()

    if borrar_csv:
        ruta_csv = os.path.join(CARPETA_DATOS, nombre_archivo)
        if os.path.exists(ruta_csv):
            os.remove(ruta_csv)


# ==========================================
# SEGMENTO 3: Obtención de respuesta por la IA
# ==========================================
def obtener_respuesta_ia(retriever, pregunta_usuario: str) -> str:
    """Configura la cadena de IA y devuelve la respuesta formulada por Groq."""
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    system_prompt = (
        "Eres un asistente de atención al cliente de una tienda online que vende "
        "ropa y accesorios de superhéroes.\n"
        "Usa exclusivamente el contexto recuperado de los archivos para responder.\n"
        "Si los datos no contienen la respuesta, di que la información no está disponible.\n\n"
        "Contexto:\n{context}"
    )
    prompt = ChatPromptTemplate.from_template(system_prompt + "\n\nPregunta: {question}")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(pregunta_usuario)


# ==========================================
# SEGMENTO 2: Interfaz para interactuar (Streamlit)
# ==========================================
def barra_lateral_gestion_documentos(vectorstore):
    """Panel lateral para agregar y eliminar documentos de la base vectorial."""
    st.sidebar.header("📁 Gestión de documentos")

    # ---- Agregar documentos ----
    st.sidebar.subheader("➕ Agregar documento")
    archivos_subidos = st.sidebar.file_uploader(
        "Sube uno o más archivos CSV",
        type=["csv"],
        accept_multiple_files=True,
        key="uploader_docs",
    )
    if archivos_subidos and st.sidebar.button("Indexar archivos subidos"):
        for archivo in archivos_subidos:
            ruta_destino = os.path.join(CARPETA_DATOS, archivo.name)
            with open(ruta_destino, "wb") as f:
                f.write(archivo.getbuffer())
            try:
                n = agregar_documento(vectorstore, ruta_destino)
                st.sidebar.success(f"'{archivo.name}' indexado ({n} fragmentos).")
            except Exception as e:
                st.sidebar.error(f"Error al indexar '{archivo.name}': {e}")
        st.rerun()

    # ---- Eliminar documentos ----
    st.sidebar.subheader("🗑️ Eliminar documento")
    documentos = listar_documentos_indexados(vectorstore)
    if documentos:
        a_eliminar = st.sidebar.selectbox("Selecciona un documento", documentos)
        if st.sidebar.button("Eliminar documento seleccionado"):
            try:
                eliminar_documento(vectorstore, a_eliminar)
                st.sidebar.success(f"'{a_eliminar}' eliminado de la base vectorial.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error al eliminar '{a_eliminar}': {e}")
    else:
        st.sidebar.info("No hay documentos indexados todavía.")

    # ---- Estado actual ----
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Documentos indexados: {len(documentos)}")
    for d in documentos:
        st.sidebar.write(f"• {d}")


def main():
    st.set_page_config(page_title="Chatbot Tienda Superhéroes", page_icon="🦸")
    st.title("🦸 Chatbot de la Tienda de Superhéroes")
    st.subheader("Atención al cliente con LangChain + Groq + ChromaDB")

    # Inicializar la base vectorial persistente
    try:
        with st.spinner("Cargando la base de datos vectorial..."):
            vectorstore = obtener_vectorstore()
        st.success("Base de datos vectorial lista.")
    except Exception as e:
        st.error(f"Error al inicializar los vectores: {e}")
        return

    # Panel lateral de gestión de documentos (agregar / eliminar)
    barra_lateral_gestion_documentos(vectorstore)

    # Recuperador con los 3 fragmentos más parecidos
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if pregunta := st.chat_input("Escribe tu pregunta sobre la tienda aquí..."):
        with st.chat_message("user"):
            st.write(pregunta)
        st.session_state.messages.append({"role": "user", "content": pregunta})

        with st.chat_message("assistant"):
            with st.spinner("La IA está consultando la base de datos vectorial..."):
                respuesta_final = obtener_respuesta_ia(retriever, pregunta)
                st.write(respuesta_final)

        st.session_state.messages.append({"role": "assistant", "content": respuesta_final})


if __name__ == "__main__":
    main()
