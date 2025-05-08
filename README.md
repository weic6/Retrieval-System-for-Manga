# Semantic Manga RAG

## Overview

Manga Retrieval System is a tool that lets readers rediscover manga from the faintest memories—no title, author, or image required. By combining multimodal understanding of manga pages with Retrieval‑Augmented Generation (RAG), the system turns vague, natural‑language recollections (“the heroine time‑travels with a cat‑shaped robot…”) into accurate search results drawn from the user’s own manga library.

## Features

**Custom RAG database:** Indexes a user‑supplied manga library with vector embeddings in ChromaDB.

**Content understanding & summarization:** uses multimodal Gemini-2.0-flash to create rich textual summaries.

For each manga, the following contents are extracted

**Vague‑memory search:** Accepts natural‑language queries and surfaces the most semantically relevant manga.

**LLM re‑ranking & explanations:** Uses a large language model to justify why each returned title matches the memory fragment.

## How to start

**Prepare data:**
Before running the app, user should create a folder `manga_images` in project root directory. Inside it, user can create folders called `manga_name_1`, `manga_name_2`, ..., `manga_name_n` for differnt manga books. Each `manga_name_n` folder contain images (should be in format .webp/.jpeg/.jpg/.png) for all the pages in that manga book.

I have upload one sample input `manga_images` for testing. You can download with this link: https://drive.google.com/drive/folders/1I0EOQyQLR32NNWqxWrcv7oKKc-OLzy29?usp=sharing

**add api key**:
rename `.env_sample` to `.env` and update the API_KEY inside with yours.

```python
# start vir env
python -m venv test_env
source test_env/bin/activate

# install dependencies
pip install -r requirements.txt
```

```
# run the app
python3.11 preprocessing.py
python3.11 vectorize.py
python3.11 query.py
```
