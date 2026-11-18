# Servicio de recuperación para NFCorpus

La API carga `../nfcorpus/corpus.jsonl` al iniciar, construye el índice invertido y la matriz dispersa TF-IDF en memoria. Utiliza los mismos pasos que `EBIM1.ipynb`: minúsculas, tokenización alfanumérica, stopwords inglesas y Porter stemming; BM25 usa k1=1.2 y b=0.75.

## Ejecutar en Windows

Desde la raíz `C:\notebooks_IR`, activa el entorno que ya usa el proyecto e inicia:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn servicio_ir.main:app --host 0.0.0.0 --port 8000
```

También funciona desde Git Bash:

```bash
source .venv/Scripts/activate
python -m uvicorn servicio_ir.main:app --host 0.0.0.0 --port 8000
```

La primera carga construye los modelos, por lo que puede tardar un poco. Deja abierta esa terminal mientras se use el servicio; `Ctrl+C` lo detiene. Si el puerto está ocupado, usa `--port 8001` y ajusta la URL de las solicitudes.

## Endpoints

- `GET http://localhost:8000/health`
- `POST http://localhost:8000/search` con JSON, por ejemplo:

```json
{"query":"dietary fiber and cardiovascular health","model":"bm25","top_k":10}
```

`model` acepta `bm25` y `tfidf`; `top_k` admite de 1 a 100. Swagger queda en `http://localhost:8000/docs`.

Para evaluación pública, despliega esta aplicación y el corpus en un host accesible por internet. La API local no crea por sí sola una URL pública.
