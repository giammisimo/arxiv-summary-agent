## arXiv Summary Agent

This agent system, based on LangGraph, generates a survey on a requested topic using the most relevant papers on arXiv.

A FastAPI implementation is also provided, which serves:
- `/agent`: an endpoint to the summary agent
- `/query`: an endpoint for semantic search

### Requirements
Both terminal use and Docker deployment require a Qdrant instance serving the following points:
| Field        | Description                                        |
|--------------|----------------------------------------------------|
| `id`         | Random uuid                                        |
| `vector`     | Embedding for the paper                            |
| `title`      | Title for the paper                                |
| `authors`    | Authors of the paper                               |
| `summary`    | Abstract for the paper                             |
| `categories` | arXiv categories for the paper                     |
| `published`  | Date of publication on arXiv (format `YYYY-MM-DD`) |
| `arxiv-id`   | arXiv ID for the paper                             |

All Qdrant and arXiv interactions are implemented in `qdrant_tool.py` and `arxiv_tool.py`.

Embeddings are implemented with `SentenceTransformer`, with the `all-mpnet-base-v2` model by default.

### Configuration
Most configuration is done through environment variables:
| Variable          | Description                               |
|-------------------|-------------------------------------------|
| `QDRANT_HOST`       | Url for the Qdrant instance               |
| `QDRANT_PORT`       | Port for the Qdrant instance              |
| `QDRANT_COLLECTION` | Name for the Qdrant collection to be used |
| `DEEPSEEK_API_KEY` | (Optional) If not provided, an Ollama instance is used |

### Deploy with Docker
Both `Dockerfile` and `docker-compose.yml` are provided.

### Command-Line use
To use the agent on the terminal, first install the python requirements (`requirements.txt`), then
```
python3 query.py
```
