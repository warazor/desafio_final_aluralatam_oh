# desafio_final_aluralatam_oh
Repositorio de almacenamiento de todo el proyecto asociado al desafio final de aluralatam

# 🦸 Chatbot RAG — Tienda de Superhéroes

Asistente de atención al cliente basado en IA para una tienda online que vende **ropa y accesorios de superhéroes**. El bot responde preguntas de los clientes (envíos, pagos, devoluciones, términos y condiciones, etc.) usando exclusivamente la información contenida en la base de conocimiento de la tienda.

---

## 📝 Descripción del sistema

El sistema es un **chatbot RAG (Retrieval-Augmented Generation)**: en lugar de dejar que el modelo de lenguaje responda "de memoria", primero recupera los fragmentos más relevantes de una base de conocimiento propia (archivos CSV con políticas, FAQs y guías de la tienda) y luego el LLM redacta la respuesta apoyándose únicamente en ese contexto. Esto reduce las alucinaciones y mantiene las respuestas alineadas con la información oficial de la tienda.

La interfaz es una aplicación web en **Streamlit** con dos capacidades principales:

1. **Chat de atención al cliente**: el usuario escribe una pregunta y recibe una respuesta generada a partir de la base de conocimiento.
2. **Gestión de la base de conocimiento** (panel lateral):
   - **Agregar documentos**: subir uno o más CSV que se indexan automáticamente en la base vectorial.
   - **Eliminar documentos**: quitar un archivo y todos sus fragmentos de la base vectorial.

La base vectorial es **persistente**: no se reconstruye en cada recarga, sino que se abre desde disco y solo se modifica cuando se agrega o elimina un documento.

---

## 🏗️ Arquitectura

```
┌──────────────────────────────────────────────────────────────┐
│                      Interfaz Web (Streamlit)                  │
│                                                                │
│   ┌───────────────┐         ┌──────────────────────────────┐  │
│   │   Chat RAG    │         │  Panel de gestión (sidebar)  │  │
│   │ (preguntas)   │         │  ➕ Agregar   🗑️ Eliminar     │  │
│   └───────┬───────┘         └───────────────┬──────────────┘  │
└───────────┼─────────────────────────────────┼─────────────────┘
            │                                 │
            ▼                                 ▼
   ┌──────────────────┐            ┌─────────────────────────┐
   │  Cadena RAG      │            │  Gestión de documentos  │
   │  (LangChain LCEL)│            │  add / delete por        │
   │                  │            │  metadata 'source'       │
   └───┬─────────┬────┘            └────────────┬────────────┘
       │         │                              │
       ▼         ▼                              ▼
 ┌──────────┐ ┌──────────────┐        ┌───────────────────────┐
 │  Groq    │ │  Retriever   │◄──────►│   ChromaDB (vectores) │
 │  LLaMA   │ │  (top-k = 3) │        │   persistido en disco │
 │  3.1-8b  │ └──────────────┘        │   db_vectores/        │
 └──────────┘        ▲                └───────────┬───────────┘
                     │                            │
                     │                    ┌───────▼────────┐
                     └────────────────────┤  Embeddings    │
                                          │  HuggingFace   │
                                          │ all-MiniLM-L6  │
                                          └───────┬────────┘
                                                  │
                                          ┌───────▼────────┐
                                          │  CSVs de la    │
                                          │  base de       │
                                          │  conocimiento  │
                                          └────────────────┘
```

**Flujo de una consulta (RAG):**

1. El usuario envía una pregunta desde el chat.
2. La pregunta se convierte en embedding y se buscan en ChromaDB los **3 fragmentos más similares** (`k = 3`).
3. Esos fragmentos se inyectan como *contexto* en el prompt.
4. **Groq (LLaMA 3.1 8B)** genera la respuesta usando solo ese contexto.
5. La respuesta se muestra en el chat y se guarda en el historial de la sesión.

**Flujo de gestión de documentos:**

- **Agregar**: el CSV se guarda en `base_conocimiento_csv/`, se divide en documentos (una fila = un fragmento), a cada fragmento se le asigna `metadata['source'] = nombre_archivo` e `id = "archivo::índice"`, y se insertan en Chroma. La operación es *idempotente*: si el archivo ya existía, primero se elimina su versión previa (permite actualizar sin duplicar).
- **Eliminar**: se borran de Chroma todos los fragmentos cuyo `metadata['source']` coincide con el archivo y, opcionalmente, se elimina el CSV físico.

---

## 🛠️ Tecnologías utilizadas

| Componente            | Tecnología                                   |
|-----------------------|----------------------------------------------|
| Interfaz web          | [Streamlit](https://streamlit.io/)           |
| Orquestación RAG      | [LangChain](https://www.langchain.com/) (LCEL) |
| Modelo LLM            | [Groq](https://groq.com/) — `llama-3.1-8b-instant` |
| Embeddings            | HuggingFace `all-MiniLM-L6-v2` (`sentence-transformers`) |
| Base de datos vectorial | [ChromaDB](https://www.trychroma.com/) (persistente) |
| Procesamiento de datos | pandas                                       |
| Contenedores / CI     | Docker + GitHub Actions                       |
| Lenguaje              | Python 3.11                                   |

---

## ⚙️ Instrucciones de instalación

### Requisitos previos

- Python 3.11+
- Una **API Key de Groq** (gratuita) → obtenla en <https://console.groq.com/keys>

### Keys necesarias

| Variable de entorno | Descripción | Dónde obtenerla |
|---------------------|-------------|-----------------|
| `GROQ_API_KEY`      | Clave de acceso a la API de Groq para el modelo LLaMA. | <https://console.groq.com/keys> |

> El modelo de embeddings (HuggingFace `all-MiniLM-L6-v2`) se descarga automáticamente y **no requiere clave**.

### Opción A — Ejecución local

```bash
# 1. Clonar el repositorio
git clone https://github.com/warazor/desafio_final_aluralatam_oh.git
cd desafio_final_aluralatam_oh

# 2. (Recomendado) Crear un entorno virtual
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar la API Key de Groq
export GROQ_API_KEY="tu_clave_de_groq"     # En Windows: set GROQ_API_KEY=tu_clave

# 5. Ejecutar la aplicación
streamlit run app.py
```

La app quedará disponible en <http://localhost:8501>.

### Opción B — Docker

```bash
# Construir la imagen
docker build -t chatbot-rag-superheroes .

# Ejecutar el contenedor inyectando la API Key
docker run -d -p 8501:8501 \
  -e GROQ_API_KEY="tu_clave_de_groq" \
  --name chatbot_superheroes \
  chatbot-rag-superheroes
```

### CI/CD (GitHub Actions)

El repositorio incluye el workflow `ejecutar_bot.yml`, que compila la imagen Docker y valida que la interfaz arranque en cada `push` a `main`. La clave se inyecta desde el secreto `GROQ_API_KEY` del repositorio:
`Settings → Secrets and variables → Actions → New repository secret`.

---

## 💬 Ejemplos de preguntas

- ¿Cuánto tarda en llegar mi pedido?
- ¿Qué métodos de pago aceptan?
- ¿Puedo cancelar un pedido después de comprarlo?
- ¿Cuál es el plazo para devolver un producto?
- ¿Ofrecen envío gratuito?
- ¿Le pueden vender un traje a un supervillano?
- ¿Aceptan pagos en créditos imperiales u otras monedas extraterrestres?

---

## 🤖 Ejemplos de respuestas

> **Pregunta:** ¿Cuánto tarda en llegar mi pedido?
>
> **Respuesta:** El tiempo de entrega estándar es de [X a X] días hábiles según tu ubicación. Recibirás un correo con el número de seguimiento una vez que el paquete sea despachado. Para envíos express, el plazo es menor.

> **Pregunta:** ¿Le pueden vender un traje a un supervillano?
>
> **Respuesta:** No. Según los términos y condiciones de la tienda, solo se permite la compra de trajes a superhéroes y justicieros. Queda estrictamente prohibida la venta o entrega de cualquier producto a villanos, supervillanos o entidades asociadas al crimen organizado multiversal.

> **Pregunta:** ¿Aceptan pagos en créditos imperiales?
>
> **Respuesta:** No. Solo se aceptan monedas fiduciarias terrestres (dólares, euros, pesos, etc.) y transferencias electrónicas seguras. El uso de monedas extraterrestres, como créditos imperiales o latino-oro ferengi, no está permitido.

> **Pregunta:** ¿Cuál es la capital de Francia?
>
> **Respuesta:** Lo siento, esa información no está disponible en la base de conocimiento de la tienda.

*(Las respuestas dependen del contenido real de los CSV; los campos entre corchetes `[ ]` son plantillas que debes completar con los datos definitivos de tu tienda.)*

---

## 📂 Estructura del proyecto

```
desafio_final_aluralatam_oh/
├── app.py                     # Aplicación principal (chat + gestión de documentos)
├── requirements.txt           # Dependencias de Python
├── Dockerfile                 # Imagen del contenedor
├── ejecutar_bot.yml           # Workflow de GitHub Actions (CI)
├── base_conocimiento_csv/     # Base de conocimiento (archivos CSV)
│   ├── guia_envio_y_entregas.csv
│   ├── politica_privacidad.csv
│   ├── politica_reembolso_devoluciones.csv
│   ├── preguntas_frecuentes.csv
│   └── terminos_y_condiciones.csv
└── db_vectores/               # Base de datos vectorial persistente (ChromaDB)
```
Evidencia Ejecución OCI
URL:http://148.116.110.159:8501/

<img width="1912" height="955" alt="image" src="https://github.com/user-attachments/assets/5242d289-3047-4191-a007-c0038c308049" />


