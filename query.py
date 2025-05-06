import os
import json
from typing import List, Dict, Any
import chromadb
import google.generativeai as genai  # Changed import format
from google.generativeai import types  # Keep this import as is
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Initialize Google Generative AI
genai.configure(api_key=API_KEY)  # Changed initialization approach
model = "gemini-2.0-flash"

class MangaRetrieval:
    def __init__(self, chroma_db_path: str = "./_chroma"):
        """Initialize the manga retrieval system with a Chroma DB."""
        # Connect to Chroma
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_collection("manga_collection")
        
        # Configure LLM (similar to preprocessing.py but with updated API)
        self.generation_config = {
            "response_mime_type": "text/plain",
            "temperature": 0.2,
            "top_p": 0.95,
            "max_output_tokens": 4096,
        }
    
    def rerank_results(self, query: str, candidates: List[Dict[Any, Any]], n: int = 5) -> List[Dict[Any, Any]]:
      """Re-rank candidates using the LLM and provide explanations."""
      if not candidates:
          return []
      
      # Create prompt for LLM to rerank and explain
      prompt = f"""
  You are a manga search expert. I need you to re-rank and explain the relevance of manga search results.

  The user is looking for a manga with this description:
  "{query}"

  Here are the candidate results:
  {json.dumps([{
      "id": c["id"],
      "title": c["metadata"].get("manga_title", "Unknown"),
      "content": c["content"][:500] + "..." if c["content"] and len(c["content"]) > 500 else c["content"]
  } for c in candidates], indent=2)}

  For each candidate, evaluate how well it matches the user's query. Consider narrative elements, characters, themes, art style, and any other relevant factors.

  Return a JSON object with the following structure:
  {{
    "ranked_results": [
      {{
        "id": "manga_id",
        "title": "manga_title",
        "relevance_score": 0-100,
        "explanation": "detailed explanation of why this manga matches the query"
      }}
    ]
  }}

  Only include the top {n} most relevant results in descending order of relevance.
  Return ONLY valid JSON, no additional text. No backticks, no explanations outside the JSON.
  """
      
      try:
          # Create a generative model instance
          model_instance = genai.GenerativeModel(model_name=model)
          
          # Generate reranking and explanations using updated API
          response = model_instance.generate_content(
              contents=prompt,
              generation_config=self.generation_config
          )
          
          # Debug the response
          print(f"Response type: {type(response)}")
          response_text = response.text
          print(f"Response text (first 100 chars): {response_text[:100]}...")
          
          # Clean the response text - remove markdown code blocks if present
          if response_text.startswith("```json") and response_text.endswith("```"):
              response_text = response_text[7:-3]  # Remove ```json and ``` markers
          elif response_text.startswith("```") and response_text.endswith("```"):
              response_text = response_text[3:-3]  # Remove ``` markers
              
          # Try to parse as JSON
          result_json = json.loads(response_text.strip())
          
          # Verify the expected structure exists
          if "ranked_results" not in result_json:
              print("Warning: 'ranked_results' key not found in response. Response structure:")
              print(json.dumps(result_json, indent=2)[:300] + "...")
              raise KeyError("ranked_results key not found in response")
              
          return result_json["ranked_results"]
      except json.JSONDecodeError as e:
          print(f"JSON parsing error: {e}")
          print(f"Raw response text: {response_text}")
          # Fallback to original ranking
          return [{"id": c["id"], 
                  "title": c["metadata"].get("manga_title", "Unknown"),
                  "relevance_score": 100 - i*10,
                  "explanation": "Ranked by vector similarity."
                  } for i, c in enumerate(candidates[:n])]
      except Exception as e:
          print(f"Error during reranking: {e}")
          # Fallback to original ranking if LLM fails
          return [{"id": c["id"], 
                  "title": c["metadata"].get("manga_title", "Unknown"),
                  "relevance_score": 100 - i*10,
                  "explanation": "Ranked by vector similarity."
                  } for i, c in enumerate(candidates[:n])]

    def search(self, query: str, n_results: int = 5) -> List[Dict[Any, Any]]:
        """
        Search for manga based on a natural language query with LLM reranking.
        
        Args:
            query: User's natural language description
            n_results: Number of top results to return
            
        Returns:
            List of manga results with explanations
        """
        # First-stage retrieval: Get candidates from vector DB
        results = self.raw_search(query, n_results=n_results*2)  # Get more candidates for re-ranking
        
        if not results:
            return []
        
        # Second-stage re-ranking: Use LLM to rerank and explain
        ranked_results = self.rerank_results(query, results, n=n_results)
        
        return ranked_results
      
    def raw_search(self, query: str, n_results: int = 5):
      """
      Perform a direct vector search without LLM reranking.
      Return raw results with similarity scores.
      """
      try:
          # Check if the collection exists
          collection_names = [c.name for c in self.chroma_client.list_collections()]
          print(f"Available collections: {collection_names}")
          
          if "manga_collection" not in collection_names:
              print("Error: manga_collection does not exist in the database")
              return []
          
          print(f"Collection count: {self.collection.count()}")
          if self.collection.count() == 0:
              print("Warning: Collection is empty")
              return []
          
          # Query the collection - use include param instead of include_distances
          print(f"Querying with text: '{query}'")
          results = self.collection.query(
              query_texts=[query],
              n_results=min(n_results, self.collection.count()),  # Don't request more than we have
              include=["metadatas", "documents", "distances"]
          )
          
          # Debug the results structure
          print(f"Query returned keys: {list(results.keys())}")
          for key, value in results.items():
              if isinstance(value, list) and value:
                  print(f"  {key}: {len(value[0])} items")
          
          # Check if results contain data
          if not results or "ids" not in results or not results["ids"] or not results["ids"][0]:
              print("Query returned no results")
              return []
          
          # Process results
          candidates = []
          for i in range(len(results["ids"][0])):
              try:
                  manga_id = results["ids"][0][i]
                  metadata = results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] and i < len(results["metadatas"][0]) else {}
                  distance = results["distances"][0][i] if "distances" in results and results["distances"] and i < len(results["distances"][0]) else None
                  document = results["documents"][0][i] if "documents" in results and results["documents"] and i < len(results["documents"][0]) else None
                  
                  candidates.append({
                      "id": manga_id,
                      "metadata": metadata,
                      "content": document,
                      "similarity": distance
                  })
              except IndexError as e:
                  print(f"Index error at position {i}: {e}")
          
          return candidates
      except Exception as e:
          print(f"Error retrieving candidates: {e}")
          import traceback
          traceback.print_exc()
          return []

def test_shaaake():
    """Function specifically to test the SHAAAKE query."""
    print("Initializing manga retrieval system...")
    retriever = MangaRetrieval()
    
    # Test the specific query "SHAAAKE"
    test_query = "SHAAAKE"
    print(f"Searching for: '{test_query}'")
    results = retriever.raw_search(test_query)
    
    if not results:
        print("No results found.")
    else:
        print("\n=== Raw Vector Search Results ===\n")
        for i, result in enumerate(results):
            manga_title = result["metadata"].get("manga_title", "Unknown")
            page_num = result["metadata"].get("page_number", "Unknown")
            doc_type = result["metadata"].get("type", "Unknown")
            
            # Calculate cosine similarity from distance
            # In Chroma, similarity = 1 - distance
            cosine_similarity = 1 - result["similarity"] if result["similarity"] is not None else None
            
            print(f"{i+1}. {manga_title} ({doc_type}, Page: {page_num})")
            print(f"   Cosine Similarity: {cosine_similarity:.4f}")
            
            # Show excerpt of content where "SHAAAKE" might appear
            content = result["content"]
            if content and "SHAAAKE" in content:
                start = max(0, content.find("SHAAAKE")-50)
                end = min(len(content), content.find("SHAAAKE")+50)
                print(f"   Excerpt: ...{content[start:end]}...")
                print(f"   MATCH FOUND - 'SHAAAKE' is present in this document!")
            else:
                print(f"   Content: {content[:150]}..." if content and len(content) > 150 else content)
            print()

def main():
    """Main function to test both vector search and LLM reranking."""
    print("Initializing manga retrieval system...")
    retriever = MangaRetrieval()
    
    # Test specific query "SHAAAKE"
    test_query = "SHAAAKE"
    print(f"\nSearching for: '{test_query}'")
    
    # Step 1: Raw vector search results
    print("\n=== STEP 1: Raw Vector Search ===")
    raw_results = retriever.raw_search(test_query)
    
    if not raw_results:
        print("No vector search results found.")
    else:
        print("\n=== Raw Vector Search Results ===\n")
        for i, result in enumerate(raw_results):
            manga_title = result["metadata"].get("manga_title", "Unknown")
            page_num = result["metadata"].get("page_number", "Unknown")
            doc_type = result["metadata"].get("type", "Unknown")
            
            # Calculate cosine similarity from distance
            cosine_similarity = 1 - result["similarity"] if result["similarity"] is not None else None
            
            print(f"{i+1}. {manga_title} ({doc_type}, Page: {page_num})")
            print(f"   Cosine Similarity: {cosine_similarity:.4f}")
            
            # Show excerpt of content where "SHAAAKE" might appear
            content = result["content"]
            if content and "SHAAAKE" in content:
                start = max(0, content.find("SHAAAKE")-50)
                end = min(len(content), content.find("SHAAAKE")+50)
                print(f"   Excerpt: ...{content[start:end]}...")
                print(f"   MATCH FOUND - 'SHAAAKE' is present in this document!")
            else:
                print(f"   Content: {content[:100]}..." if content and len(content) > 100 else content)
            print()
    
    # Step 2: LLM reranking
    if raw_results:
        print("\n=== STEP 2: LLM Reranking ===")
        print("Reranking results with LLM...")
        
        ranked_results = retriever.rerank_results(test_query, raw_results)
        
        print("\n=== LLM-Reranked Results ===\n")
        for i, result in enumerate(ranked_results):
            print(f"{i+1}. {result['title']} (Score: {result['relevance_score']})")
            print(f"   {result['explanation']}")
            print()
    
    # Interactive search mode
    print("\nManga Retrieval System")
    print("Enter a description of the manga you're looking for (or press Enter to exit):")
    user_query = input("> ")
    
    if user_query:
        print("\nSearching for matching manga...")
        results = retriever.search(user_query)
        
        if not results:
            print("No results found.")
        else:
            print("\n=== Top Matches ===\n")
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']} (Score: {result['relevance_score']})")
                print(f"   {result['explanation']}")
                print()

if __name__ == "__main__":
    main()