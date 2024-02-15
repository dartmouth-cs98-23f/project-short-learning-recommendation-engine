# Discite RE Engine

## Introduction

This repo implements a simple recommendation engine for the Discite project.

### Tech Stack

| Technology | Why |
| ---------- | --- |
| [Python][python] | Python is easy to use. |
| [FastAPI][fast-api] | For developing quick-iteration API's to interact with out engine. |
| [Pinecone][pinecone] | For vector similarity search. |

### Architecture

### Installation

```bash
pip install -r requirements.txt

# or manually
pip install "fastapi[all]"
pip install pinecone-client
```

> [!IMPORTANT]
> The Pinecone API key is required to connect to the Pinecone service.
>
> Update `.env` with the API key.
> See [`.env.example`](.env.example) for an example.

### Structure

| File/Directory | Description |
| -------------- | ----------- |
| [demo.py](demo.py) | Demo routes that interact with Pinecone. |
| [main.py](main.py) | Main API routes. |
| [generation](generation) | Code relevant to candidate generation. |
| [ranking](ranking) | Code relevant to candidate ranking. |

&copy; 2023 [Discite][discite]

[discite]: https://discite.tech
[python]: https://www.python.org/
[fast-api]: https://fastapi.tiangolo.com/
[pinecone]: https://www.pinecone.io/
