import os
import json
import chromadb
from chromadb.utils import embedding_functions
import uuid

def load_manga_schema_files(manga_root_folder="./manga_images"):
    """
    Load all manga schema files from the manga_analyses folder
    """
    analyses_dir = os.path.join(manga_root_folder, "..", "manga_analyses")
    
    # Get all files ending with _schema.json
    schema_files = [f for f in os.listdir(analyses_dir) if f.endswith("_schema.json")]
    print(f"Found {len(schema_files)} manga schema files to process")
    
    all_manga_data = []
    
    # Process each schema file
    for schema_file in schema_files:
        manga_file = os.path.join(analyses_dir, schema_file)
        manga_title = schema_file.replace("_schema.json", "")
        
        try:
            with open(manga_file, 'r', encoding='utf-8') as f:
                manga_data = json.load(f)
                all_manga_data.append({
                    "title": manga_title,
                    "data": manga_data
                })
                print(f"Loaded schema data for manga: {manga_title}")
        except FileNotFoundError:
            print(f"Warning: Schema file for {manga_title} not found")
            continue
    
    return all_manga_data

def create_documents_from_manga_schema(manga_data):
    """
    Create documents from manga schema data for vectorization at book, page, and panel levels
    """
    documents = []
    metadatas = []
    ids = []
    
    for manga in manga_data:
        manga_title = manga["title"]
        manga_obj = manga["data"]
        if manga_obj:
            # Create a book-level document
            book_summary = manga_obj.get("summary", "")
            if book_summary:
                book_doc = f"Manga: {manga_obj['manga_name']}. Summary: {book_summary}"
                documents.append(book_doc)
                metadatas.append({
                    "manga_title": manga_title,
                    "type": "book_summary",
                    "level": "book"
                })
                ids.append(f"{manga_title}_book")
            
            # Create page-level documents
            for page_index, page in enumerate(manga_obj.get("pages", [])):
                page_number = page.get("page_number", "unknown")
                page_summary = page.get("summary", "")
                
                # Create a descriptive page-level document
                page_doc = f"Manga: {manga_obj['manga_name']}. Page {page_number}. {page_summary}"
                documents.append(page_doc)
                metadatas.append({
                    "manga_title": manga_title,
                    "type": "page",
                    "page_number": page_number,
                    "level": "page",
                    "image_path": page.get("image_path", "")
                })
                ids.append(f"{manga_title}_page_{page_index+1}")
                
                # Create panel-level documents
                for panel_idx, panel in enumerate(page.get("panels", [])):
                    panel_id = panel.get("panel_id", f"{panel_idx+1}")
                    panel_summary = panel.get("summary", "")
                    
                    # Combine character information
                    characters_text = ""
                    for char in panel.get("characters", []):
                        char_name = char.get("name", "")
                        char_expr = char.get("expression", "")
                        char_pose = char.get("pose", "")
                        if char_name:
                            characters_text += f"{char_name} with {char_expr} expression, {char_pose}. "
                    
                    # Combine setting information
                    setting = panel.get("setting", {})
                    location = setting.get("location", "")
                    bg_elements = ", ".join(setting.get("background_elements", []))
                    setting_text = f"Location: {location}. Background elements: {bg_elements}. "
                    
                    # Combine narrative information
                    narrative = panel.get("narrative", {})
                    actions = ", ".join(narrative.get("actions", []))
                    dialogue = ". ".join(narrative.get("dialogue", []))
                    emotion = narrative.get("emotion", "")
                    narrative_text = f"Actions: {actions}. Dialogue: {dialogue}. Emotion: {emotion}. "
                    
                    # Combine text elements
                    text_elements = ", ".join(panel.get("text_elements", []))
                    text_elements_text = f"Text elements: {text_elements}. " if text_elements else ""
                    
                    # Create a descriptive panel-level document
                    panel_doc = f"Manga: {manga_obj['manga_name']}. Page {page_number}, Panel {panel_id}. {panel_summary} {characters_text}{setting_text}{narrative_text}{text_elements_text}"
                    documents.append(panel_doc)
                    metadatas.append({
                        "manga_title": manga_title,
                        "type": "panel",
                        "page_number": page_number,
                        "panel_id": panel_id,
                        "level": "panel",
                        "image_path": page.get("image_path", "")
                    })
                    ids.append(f"{manga_title}_page_{page_index+1}_panel_{panel_id}")
        if len(set(ids)) != len(ids):
            print(ids)
            duplicate_ids = [id for id in ids if ids.count(id) > 1]
            raise ValueError(f"Duplicate IDs found: {duplicate_ids[:5]}")
    return documents, metadatas, ids

def vectorize_manga_schemas(chroma_path="_chroma"):
    """
    Vectorize manga schema data and store in ChromaDB
    """
    # Load manga schema data
    manga_data = load_manga_schema_files()
    if not manga_data:
        print("No manga schema data found to vectorize")
        return
    
    # Create documents for vectorization
    documents, metadatas, ids = create_documents_from_manga_schema(manga_data)
    
    print(f"Created {len(documents)} documents for vectorization")
    for i in range(min(3, len(documents))):
        print(f"Sample document {i+1}: {documents[i][:100]}...")
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    
    # Create or get collection
    try:
        # Try to get existing collection
        collection = chroma_client.get_collection("manga_collection")
        print("Found existing manga_collection")
    except Exception:
        # Create new collection if it doesn't exist
        print("Creating new manga_collection")
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        collection = chroma_client.create_collection(
            name="manga_collection",
            embedding_function=default_ef,
            metadata={"description": "Manga summaries collection"}
        )
    
    # Add documents to collection
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} documents to ChromaDB collection")
    return collection

if __name__ == "__main__":
    # Create _chroma directory if it doesn't exist
    os.makedirs("_chroma", exist_ok=True)
    
    # Vectorize manga schemas
    vectorize_manga_schemas()
    print("Manga schema vectorization complete!")