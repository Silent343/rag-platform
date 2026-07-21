# Document RAG — Backend (FastAPI + Gemini + ChromaDB)

Backend de un sistema RAG (Retrieval-Augmented Generation): subís documentos,
hacés preguntas, y un LLM responde **citando las fuentes** — sin inventar.

## Cómo funciona (el flujo RAG)

**Ingesta:** PDF/TXT → extraer texto → partir en chunks con overlap →
embeddings (Gemini Embedding 2) → guardar vectores en ChromaDB.

**Query:** pregunta → embedding → buscar los chunks más similares →
armar prompt con ese contexto → Gemini responde solo desde el contexto.

## Arquitectura

```
app/
├── main.py              # FastAPI app + CORS
├── config.py            # settings desde .env
├── models/schemas.py    # contratos Pydantic (request/response)
├── services/
│   ├── document_loader.py  # extrae texto de PDF/TXT/MD
│   ├── chunker.py          # parte texto en chunks con overlap
│   ├── embeddings.py       # vectores con Gemini
│   ├── vector_store.py     # ChromaDB: guardar + buscar
│   └── rag_service.py      # orquesta ingesta + query + prompt
└── api/
    ├── dependencies.py  # wiring (DI)
    └── routes.py        # endpoints REST
```

## API REST

| Método | Ruta                       | Descripción                |
|--------|----------------------------|----------------------------|
| POST   | `/api/documents`           | Subir documento (multipart)|
| GET    | `/api/documents`           | Listar documentos          |
| DELETE | `/api/documents/{id}`      | Borrar documento           |
| POST   | `/api/query`               | Preguntar (body JSON)      |
| GET    | `/health`                  | Liveness                   |

## Cómo correr

```bash
# 1. Crear entorno e instalar
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r requirements.txt

# 2. Configurar la API key
copy .env.example .env          # Windows
# editar .env y poner tu GEMINI_API_KEY
# (se obtiene gratis en https://aistudio.google.com/apikey)

# 3. Levantar
uvicorn app.main:app --reload

# Docs interactivas: http://localhost:8000/docs
```

## Validado

La lógica de chunking pasa 9 tests unitarios y el flujo completo de la API
pasa 11 tests de integración (ingesta, listado, query con fuentes, borrado,
validación).

## Nota sobre el anti-alucinación

El prompt (`rag_service.py`) instruye al modelo a responder SOLO desde el
contexto y a decir explícitamente "I could not find the answer..." cuando la
respuesta no está en los documentos. Eso es el corazón de un RAG: la respuesta
viene de tus documentos, no del conocimiento general del modelo.

## Actualización de Gemini

El backend usa el SDK actual `google-genai`, con `gemini-3.5-flash` para respuestas y `gemini-embedding-2` para búsqueda. Los vectores de `gemini-embedding-001` no son compatibles con el modelo nuevo, por lo que se usa la colección `documents_gemini_embedding_2`. Volvé a subir los documentos para indexarlos allí; la colección antigua no se elimina.

## Production and Render

See [DEPLOYMENT.md](../DEPLOYMENT.md) for the Render Blueprint, required environment variables, and the persistent ChromaDB disk configuration.
