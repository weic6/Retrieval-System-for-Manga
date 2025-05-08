# Semantic Manga RAG

## Overview

Manga Retrieval System is a tool that lets readers rediscover manga from the faintest memories—no title, author, or image required. By combining multimodal understanding of manga pages with Retrieval‑Augmented Generation (RAG), the system turns vague, natural‑language recollections into accurate search results drawn from the user's own manga library.

## Features

**Custom RAG database:** Indexes a user‑supplied manga library with vector embeddings in ChromaDB.

**Content understanding & summarization:** uses multimodal Gemini-2.0-flash to create rich textual summaries.

**Vague‑memory search:** Accepts natural‑language queries and surfaces the most semantically relevant manga.

**LLM re‑ranking & explanations:** Uses a large language model to justify why each returned title matches the memory fragment.

## Setup

### Prepare data

Before running the app, create a folder `manga_images` in the project root directory. Inside it, create folders with manga titles (e.g., `manga_name_1`, `manga_name_2`, etc.) for different manga books. Each manga folder should contain images (in .webp, .jpeg, .jpg, or .png format) for all the pages in that manga book.

A sample dataset is available for testing: [Download Sample Manga Images](https://drive.google.com/drive/folders/1I0EOQyQLR32NNWqxWrcv7oKKc-OLzy29?usp=sharing)

### Add API key

Rename `.env_sample` to `.env` and update the API_KEY with your own.

### Installation

```bash
# Create and activate virtual environment
python -m venv test_env
source test_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the application

```bash
# Process manga images and generate summaries
python3.11 preprocessing.py

# Create vector embeddings for search
python3.11 vectorize.py

# Start the query interface
python3.11 query.py
```

## Example Usage

### Example preprocessing

Download the encyclopedia_girls manga in sample dataset.

The output of preprocessing in json format:

```json
{
  "manga_name": "encyclopedia_girls",
  "summary": "The Encyclopedia Girls introduce various topics and provide educational insights, often with humorous and dramatic interactions among the characters.",
  "pages": [
    {
      "page_number": 1,
      "image_path": "./manga_images/encyclopedia_girls/page1.png",
      "summary": "The Encyclopedia Girls introduce themselves and share some knowledge about mooring posts in harbors.",
      "panels": [
        {
          "panel_id": "1",
          "characters": [
            {
              "name": "Girl",
              "expression": "smiling",
              "pose": "pointing"
            }
          ],
          "setting": {
            "location": "outdoors",
            "background_elements": ["building"]
          },
          "narrative": {
            "actions": ["introduces a topic"],
            "dialogue": ["Mooring posts!"],
            "emotion": "cheerful"
          },
          "text_elements": [],
          "summary": "A girl introduces the topic of 'mooring posts'."
        },
        {
          "panel_id": "2",
          "characters": [
            {
              "name": "Girl",
              "expression": "neutral",
              "pose": "walking"
            },
            {
              "name": "Girl",
              "expression": "neutral",
              "pose": "walking"
            },
            {
              "name": "Girl",
              "expression": "neutral",
              "pose": "walking"
            }
          ],
          "setting": {
            "location": "street",
            "background_elements": ["buildings"]
          },
          "narrative": {
            "actions": ["walk down the street", "discussing something"],
            "dialogue": [
              "Like those steel mushroom-like objects that spring up at harbors.",
              "You know, there are often things that we all know but don't know the name of."
            ],
            "emotion": "informational"
          },
          "text_elements": ["!"],
          "summary": "Three girls are walking and one of them describes something they all know but might not know the name of."
        },
        {
          "panel_id": "3",
          "characters": [
            {
              "name": "Girl",
              "expression": "smiling",
              "pose": "explaining"
            },
            {
              "name": "Girl",
              "expression": "questioning",
              "pose": "looking"
            },
            {
              "name": "Girl",
              "expression": "neutral",
              "pose": "looking"
            }
          ],
          "setting": {
            "location": "street",
            "background_elements": []
          },
          "narrative": {
            "actions": ["explaining something"],
            "dialogue": [
              "...what?",
              "Those pillar-like things for mooring boats in harbors, right? They're mooring posts..."
            ],
            "emotion": "educational"
          },
          "text_elements": [],
          "summary": "One of the girls explains what they were talking about, stating that they are called 'mooring posts'."
        }
      ]
    },
    {
      "page_number": 2,
      "image_path": "./manga_images/encyclopedia_girls/page2.png",
      "summary": "A girl struggles to answer a question, while the others try to get her to answer, eventually resorting to physical prodding. Another girl reflects on whether she should have spoken up.",
      "panels": [
        {
          "panel_id": "1",
          "characters": [
            {
              "name": "Girl",
              "expression": "annoyed",
              "pose": "crossed arms"
            },
            {
              "name": "Girl",
              "expression": "worried",
              "pose": "standing"
            },
            {
              "name": "Girl",
              "expression": "distressed",
              "pose": "kneeling"
            }
          ],
          "setting": {
            "location": "outdoors",
            "background_elements": []
          },
          "narrative": {
            "actions": ["scolds", "kneels", "stands"],
            "dialogue": [
              "You can't take a hint for the life of you, huh?",
              "You did this not once but twice!"
            ],
            "emotion": "frustrated"
          },
          "text_elements": [],
          "summary": "Two girls confront a third, who is kneeling and appears distressed, for failing to answer a question."
        },
        {
          "panel_id": "2",
          "characters": [
            {
              "name": "Girl",
              "expression": "concerned",
              "pose": "standing"
            }
          ],
          "setting": {
            "location": "outdoors",
            "background_elements": []
          },
          "narrative": {
            "actions": ["thinks"],
            "dialogue": ["Oh no...", "If I answer now..."],
            "emotion": "worried"
          },
          "text_elements": [],
          "summary": "A girl looks on in concern, realizing the implications if she answers."
        },
        {
          "panel_id": "3",
          "characters": [
            {
              "name": "Girl",
              "expression": "concerned",
              "pose": "covering mouth"
            }
          ],
          "setting": {
            "location": "outdoors",
            "background_elements": []
          },
          "narrative": {
            "actions": ["reflects"],
            "dialogue": ["I know it's a little late to say this, but..."],
            "emotion": "thoughtful"
          },
          "text_elements": ["Hm ..."],
          "summary": "A girl considers speaking up, but hesitates, realizing it might be too late."
        },
        {
          "panel_id": "4",
          "characters": [
            {
              "name": "Girl",
              "expression": "determined",
              "pose": "shaking"
            },
            {
              "name": "Girl",
              "expression": "determined",
              "pose": "shaking"
            }
          ],
          "setting": {
            "location": "outdoors",
            "background_elements": []
          },
          "narrative": {
            "actions": ["shakes"],
            "dialogue": [],
            "emotion": "urgent"
          },
          "text_elements": ["Shake", "Shake"],
          "summary": "Two girls shake the third, trying to get her to answer."
        },
        {
          "panel_id": "5",
          "characters": [
            {
              "name": "Girl",
              "expression": "concerned",
              "pose": "watching"
            }
          ],
          "setting": {
            "location": "outdoors",
            "background_elements": []
          },
          "narrative": {
            "actions": ["observes"],
            "dialogue": [
              "You know the answer, don't you? It's okay, you can say it."
            ],
            "emotion": "encouraging"
          },
          "text_elements": [],
          "summary": "A girl encourages the distressed girl to answer the question."
        },
        {
          "panel_id": "6",
          "characters": [
            {
              "name": "Girl",
              "expression": "distressed",
              "pose": "being restrained"
            },
            {
              "name": "Girl",
              "expression": "determined",
              "pose": "restraining"
            },
            {
              "name": "Girl",
              "expression": "determined",
              "pose": "restraining"
            }
          ],
          "setting": {
            "location": "outdoors",
            "background_elements": []
          },
          "narrative": {
            "actions": ["restrains", "threatens"],
            "dialogue": [
              "Come on, I'll put you in a headlock if you don't spit it out."
            ],
            "emotion": "aggressive"
          },
          "text_elements": ["Shaaake"],
          "summary": "Two girls physically restrain the third, threatening her to answer the question."
        },
        {
          "panel_id": "7",
          "characters": [
            {
              "name": "Girl",
              "expression": "concerned",
              "pose": "watching"
            }
          ],
          "setting": {
            "location": "outdoors",
            "background_elements": []
          },
          "narrative": {
            "actions": ["observes"],
            "dialogue": [],
            "emotion": "concerned"
          },
          "text_elements": [],
          "summary": "A girl watches the other two girls pressuring the third."
        }
      ]
    }
  ]
}
```

After completing the setup and running the preprocessing and vectorization steps, you can use the query interface to search for manga based on your memories.

### Example Query

Using `query.py`, you can input natural language descriptions like:

```
> python3.11 query.py
shaaake
```

The system will return relevant matches from your manga library along with explanations:

```
===== Search Results =====

1. encyclopedia_girls - Page 2, Panel 6
   Relevance Score: 100
   Vector Similarity: -0.5601
   Match Type: text_elements
   Image Path: ./manga_images/encyclopedia_girls/page2.png
   This panel contains the exact text element 'Shaaake'. Given the user's query is 'shaaake', this is the most direct and relevant match. The panel-level detail is also appropriate given the specificity of the query.

2. encyclopedia_girls - Page 2
   Relevance Score: 60
   Vector Similarity: -0.6267
   Match Type: overall
   Image Path: ./manga_images/encyclopedia_girls/page2.png
   This page is relevant because it contains the panel with the text 'Shaaake'. While the page-level description doesn't explicitly mention the text, it provides context for the scene where the text appears. It's less relevant than the panel itself, but still important.

3. encyclopedia_girls - Page 1, Panel 1
   Relevance Score: 5
   Vector Similarity: -0.6225
   Match Type: narrative
   Image Path: ./manga_images/encyclopedia_girls/page1.png
   This panel is not relevant to the query 'shaaake'. It describes a girl introducing the topic of 'mooring posts' and contains no matching text or narrative elements.

4. encyclopedia_girls - Page 1
   Relevance Score: 5
   Vector Similarity: -0.6240
   Match Type: narrative
   Image Path: ./manga_images/encyclopedia_girls/page1.png
   This page is not relevant to the query 'shaaake'. It describes the Encyclopedia Girls introducing themselves and discussing mooring posts, with no matching text or narrative elements.

5. [Senukin] The Raise [complete]-1280x - Page 2, Panel 3
   Relevance Score: 1
   Vector Similarity: -0.6089
   Match Type: narrative
   Image Path: ./manga_images/[Senukin] The Raise [complete]-1280x/2_The_Raise_2.webp
   This panel is not relevant to the query 'shaaake'. It describes the boss showing pictures of a girl in a bunny suit and bikini, with no matching text or narrative elements.
```
