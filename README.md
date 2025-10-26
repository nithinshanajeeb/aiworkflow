# AI Workflow Intake App

Streamlit-based prototype for a social support intake workflow. Applicants submit personal details and income documentation, the app evaluates eligibility, and everything is stored in PostgreSQL for follow-up. Document parsing and fraud checks rely on an Ollama-hosted `granite3.2-vision` model via the OpenAI-compatible API, while eligibility scoring comes from the simple rules engine in `eligibility.py`.

## Why Run Ollama Natively?

- Keeps sensitive applicant data on the same machine; no third-party API uploads are required.
- Gives full control over model selection, caching, and GPU usage for faster PDF transcription.
- Allows experimentation with other Ollama-hosted models without touching upstream cloud services.

## Running Ollama Locally

1. Install Ollama following the platform instructions at https://ollama.com/download.
2. Start the daemon with `ollama serve` (macOS/Linux) or launch the desktop app.
3. Pull the model used by this project:

   ```bash
   ollama pull granite3.2-vision
   ```

4. Ensure the service is reachable at `http://localhost:11434` (the default). If you run it elsewhere, set `OLLAMA_BASE_URL` for both the Streamlit app and worker containers.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) v2 (ships with recent Docker Desktop builds)

## Quick start

1. Build and start the stack:

   ```bash
   make up
   ```

   Use `make up-detach` to run in the background or inspect available targets with `make help`.

2. Open http://localhost:8501 to use the intake form. The Streamlit app connects to PostgreSQL through the `DATABASE_URL` environment variable that is provided in `docker-compose.yml`. The Ollama server is exposed at http://localhost:11434 for local LLM interactions.

   Need direct database access from your host? With Compose running you can connect to PostgreSQL at `localhost:5432` using `app_user` / `app_password` and the database `aiworkflow`.

3. Stop the stack with `Ctrl+C` or shut everything down via:

   ```bash
   make down
   ```

   To remove containers and the Postgres volume, run:

   ```bash
   make clean
   ```

## Configuration

- `DATABASE_URL` (default `postgresql://app_user:app_password@db:5432/aiworkflow`): override this to point at a different PostgreSQL instance.
- `OLLAMA_BASE_URL` (default `http://ollama:11434`): change if the app should reach a different Ollama endpoint.
- `OLLAMA_SKIP_SETUP` (default unset): set to `1` to bypass the automatic on-start model pull.
- `OLLAMA_API_KEY` (optional, default `ollama`): override if the Ollama instance enforces an OpenAI-compatible token.
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (optional): when both are set the Ollama client wraps requests with the Langfuse OpenAI integration so usage is captured in Langfuse (configure `LANGFUSE_HOST` if your Langfuse deployment is self-hosted).
- The Postgres credentials and database name can be updated in `docker-compose.yml`. Make sure the values stay in sync with `DATABASE_URL`.
- Streamlit configuration can be customised via additional environment variables on the `app` service. See the [Streamlit config reference](https://docs.streamlit.io/library/advanced-features/configuration#set-configuration-options) for details.

The document parser relies on the `granite3.2-vision` model exposed by the Ollama service. On first launch the app triggers a pull via the Ollama HTTP API (unless `OLLAMA_SKIP_SETUP=1`). The app image installs `poppler-utils` so `pdf2image` can rasterize PDFs before sending them to the model, and page transcription uses the OpenAI Python client configured against Ollama's OpenAI-compatible endpoint.

Persistent Postgres data lives in the named volume `postgres_data`. Delete it with `make clean` if you want a clean slate.

### Database schema

Starting the app creates two tables:

- `applications`: applicant submissions and eligibility decisions (existing schema).
- `applications_documents`: Markdown extracted from uploaded PDFs, linked via `application_id` foreign key.

### Useful commands

- `make logs` tails service output (add `SERVICE=app` or `SERVICE=db` to scope)
- `make ps` shows container status
- `make build` rebuilds images without starting containers
