# Semantic Manga RAG

## Overview

Manga Retrieval System is a tool that lets readers rediscover manga from the faintest memories—no title, author, or image required. By combining multimodal understanding of manga pages with Retrieval‑Augmented Generation (RAG), the system turns vague, natural‑language recollections (“the heroine time‑travels with a cat‑shaped robot…”) into accurate search results drawn from the user’s own manga library.

## Features

**Custom RAG database:** Indexes a user‑supplied manga library with vector embeddings in ChromaDB.

**Content understanding & summarization:** uses multimodal Gemini-2.0-flash to create rich textual summaries.

**Vague‑memory search:** Accepts natural‑language queries and surfaces the most semantically relevant manga.

**LLM re‑ranking & explanations:** Uses a large language model to justify why each returned title matches the memory fragment.

## How to start

**add api key**:
rename `.env_sample` to `.env` and update the API_KEY inside with yours.

```python
# start vir env
python -m venv test_env
source test_env/bin/activate

# install dependencies
pip install -r requirements.txt
```

**Prepare data:**
Before running the app, drag a folder named after your manga book name `book_name`into `manga_images` folder. The folder `book_name` should only contain images for all the pages in that manga book.

```
# run the app
python3.11 preprocessing.py
python3.11 vectorize.py
python3.11 query.py
```
