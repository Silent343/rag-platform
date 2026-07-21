# Deploying the backend on Render Free

Connect the `rag-backend` folder as the repository root in Render, or use this
folder as the root directory of the repository. Then select **New → Blueprint**
and choose `render.yaml`.

The Blueprint creates only `document-rag-api`. It does not deploy the Angular
frontend. Render will ask for `GEMINI_API_KEY` as a secret during setup.

This test deployment uses Render Free and stores ChromaDB in `/tmp/chroma`.
Documents are temporary and disappear after a restart or redeploy. Upload a
document again whenever you redeploy.

Build command: `pip install -r requirements.txt`

Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

Health check: `/health`
