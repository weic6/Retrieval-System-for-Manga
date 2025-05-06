import os
import json
import chromadb
from chromadb.utils import embedding_functions
import uuid

def load_manga_summaries(manga_root_folder="./manga_images"):
    """
    Load all manga summaries from the manga_analyses folder
    """
    analyses_dir = os.path.join(manga_root_folder, "manga_analyses")
    
    # Load the summary file to get list of manga titles
    summary_file = os.path.join(analyses_dir, "all_manga_summary.json")
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    manga_titles = summary_data["manga_titles"]
    print(f"Found {len(manga_titles)} manga titles to process")
    
    all_manga_data = []
    
    # Process each manga title
    for manga_title in manga_titles:
        manga_file = os.path.join(analyses_dir, f"{manga_title}_analysis.json")
        
        try:
            with open(manga_file, 'r', encoding='utf-8') as f:
                manga_data = json.load(f)
                all_manga_data.append({
                    "title": manga_title,
                    "data": manga_data
                })
                print(f"Loaded data for manga: {manga_title}")
        except FileNotFoundError:
            print(f"Warning: Analysis file for {manga_title} not found")
            continue
    
    return all_manga_data

def extract_clean_json_content(text):
    """
    Extract and clean JSON content from text that might contain markdown code blocks
    """
    if text.startswith("```json") and text.endswith("```"):
        # Extract content between the backticks
        json_text = text[7:-3]  # Remove ```json and ``` markers
    else:
        json_text = text
    
    print(text)
    try:
        # Try to parse as JSON to ensure it's valid
        parsed = json.loads(json_text)
        
        # Convert back to string - this effectively cleans the JSON
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        # If we can't parse it as JSON, return the original text
        print("Warning: Could not parse JSON content, using raw text")
        return text

def create_documents_from_manga(manga_data):
    """
    Create documents from manga data for vectorization
    """
    documents = []
    metadatas = []
    ids = []
    
    for manga in manga_data:
        manga_title = manga["title"]
        pages = manga["data"]
        
        # Create a book-level document
        if pages and "summary_book_level" in pages[0]:
            book_summary = pages[0]["summary_book_level"]
            documents.append(book_summary)
            metadatas.append({
                "manga_title": manga_title,
                "type": "book_summary",
                "page_number": -1,  # Use -1 instead of None for book summaries
                "json_data": json.dumps({"summary": book_summary})
            })
            ids.append(f"{manga_title}_book_summary")
        
        # Create page-level documents
        for page in pages:
            try:
                page_number = page["page_number"]
                page_summary_raw = page["summary_page_level"]
                
                # Clean up the page-level summary JSON
                cleaned_page_summary = extract_clean_json_content(page_summary_raw)
                
                # Add page-level document
                documents.append(cleaned_page_summary)
                metadatas.append({
                    "manga_title": manga_title,
                    "type": "page_summary",
                    "page_number": page_number,
                    "json_data": cleaned_page_summary
                })
                ids.append(f"{manga_title}_page_{page_number}")
            except Exception as e:
                print(f"Error processing page {page.get('page_number', 'unknown')} of {manga_title}: {e}")
    
    return documents, metadatas, ids

def vectorize_manga_summaries(chroma_path="_chroma"):
    """
    Vectorize manga summaries and store in ChromaDB
    """
    # Load manga data
    manga_data = load_manga_summaries()
    if not manga_data:
        print("No manga data found to vectorize")
        return
    
    # Create documents for vectorization
    documents, metadatas, ids = create_documents_from_manga(manga_data)
    
    print(f"Created {len(documents)} documents for vectorization")
    
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
        # Clean metadata by removing None values and the large json_data field
        cleaned_metadatas = []
        for m in metadatas:
            # Only keep non-None values and skip json_data
            cleaned_metadata = {k: v for k, v in m.items() if v is not None and k != 'json_data'}
            cleaned_metadatas.append(cleaned_metadata)
            
        collection.add(
            documents=documents,
            metadatas=cleaned_metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} documents to ChromaDB collection")
    return collection

if __name__ == "__main__":
    # Create _chroma directory if it doesn't exist
    os.makedirs("_chroma", exist_ok=True)
    
    # Vectorize manga summaries
    vectorize_manga_summaries()
    print("Manga summaries vectorization complete!")