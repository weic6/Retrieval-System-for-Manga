import os
import json
from typing import List, Dict, Any
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Initialize Google Generative AI
genai.configure(api_key=API_KEY)
model = "gemini-2.0-flash"

class MangaRetrieval:
    def __init__(self, chroma_db_path: str = "./_chroma"):
        """Initialize the manga retrieval system with a Chroma DB."""
        # Connect to Chroma
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_collection("manga_collection")
        
        # Configure LLM
        self.generation_config = {
            "response_mime_type": "text/plain",
            "temperature": 0.2,
            "top_p": 0.95,
            "max_output_tokens": 4096,
        }
    
    def clean_json_response(self, response_text):
        """Remove markdown code block formatting if present."""
        if response_text.startswith('```'):
            # Find the end of the opening markdown delimiter
            first_newline = response_text.find('\n')
            if first_newline > 0:
                # Find the closing markdown delimiter
                last_triple_backtick = response_text.rfind('```')
                if last_triple_backtick > first_newline:
                    # Extract just the JSON content
                    response_text = response_text[first_newline+1:last_triple_backtick].strip()
                else:
                    # Only remove the opening delimiter if no closing one is found
                    response_text = response_text[first_newline+1:].strip()
        return response_text
    
    def rerank_results(self, query: str, candidates: List[Dict[Any, Any]], n: int = 5) -> List[Dict[Any, Any]]:
        """Re-rank candidates using the LLM and provide explanations."""
        if not candidates:
            return []
        
        # Prepare candidate data with hierarchical information
        candidate_info = []
        for c in candidates:
            # Extract hierarchical information
            level = c["metadata"].get("level", "unknown")
            manga_title = c["metadata"].get("manga_title", "Unknown")
            page_number = c["metadata"].get("page_number", "Unknown")
            panel_id = c["metadata"].get("panel_id", "Unknown") if level == "panel" else ""
            
            # Build a hierarchical identifier
            identifier = f"{manga_title}"
            if level == "page":
                identifier += f" - Page {page_number}"
            elif level == "panel":
                identifier += f" - Page {page_number}, Panel {panel_id}"
            
            candidate_info.append({
                "id": c["id"],
                "title": manga_title,
                "level": level,
                "identifier": identifier,
                "content": c["content"][:500] + "..." if c["content"] and len(c["content"]) > 500 else c["content"],
                "metadata": {k: v for k, v in c["metadata"].items() if k != "image_path"},
                "vector_similarity": c.get("similarity", None)  # Include vector similarity for reference
            })
        
        # Create enhanced prompt for LLM to rerank with hierarchical understanding
        prompt = f"""
You are a manga search expert. I need you to re-rank and explain the relevance of manga search results.

The user is looking for a manga with this description:
"{query}"

Here are the candidate results:
{json.dumps(candidate_info, indent=2)}

For each candidate, evaluate how well it matches the user's query. 
Consider these aspects in your evaluation:
1. If the user is describing a specific panel or scene, prioritize panel-level matches
2. If the user is describing an overall page, prioritize page-level matches
3. If the user is describing the general story, prioritize book-level matches

Think about these elements:
- Character details (names, expressions, poses)
- Setting details (location, background elements)
- Narrative (actions, dialogue, emotions)
- Visual text elements and onomatopoeia

Return a JSON object with the following structure:
{{
  "ranked_results": [
    {{
      "id": "result_id",
      "level": "book|page|panel",
      "title": "manga_title",
      "identifier": "hierarchical identifier",
      "relevance_score": 0-100,
      "explanation": "detailed explanation of why this matches the query",
      "match_type": "character|setting|narrative|text_elements|overall"
    }}
  ]
}}

Only include the top {n} most relevant results in descending order of relevance.
Return ONLY valid JSON, no additional text.
"""
        
        try:
            # Create a generative model instance
            model_instance = genai.GenerativeModel(model_name=model)
            
            # Generate reranking and explanations
            response = model_instance.generate_content(
                contents=prompt,
                generation_config=self.generation_config
            )
            
            # Get response text
            response_text = response.text
            
            # Clean the response text
            cleaned_response = self.clean_json_response(response_text)
            
            # Parse as JSON
            result_json = json.loads(cleaned_response.strip())
            
            # Verify the expected structure exists
            if "ranked_results" not in result_json:
                print("Warning: 'ranked_results' key not found in response.")
                raise KeyError("ranked_results key not found in response")
                
            # Add image path and vector similarity back to results from original candidates
            id_to_info = {c["id"]: {"image_path": c["metadata"].get("image_path", ""), 
                                     "vector_similarity": c.get("similarity", None)} 
                          for c in candidates}
            
            for result in result_json["ranked_results"]:
                result_id = result["id"]
                if result_id in id_to_info:
                    result["image_path"] = id_to_info[result_id]["image_path"]
                    result["vector_similarity"] = id_to_info[result_id]["vector_similarity"]
            
            return result_json["ranked_results"]
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Cleaned response was: {cleaned_response[:500]}...")
            # Fallback to original ranking
            return self._fallback_ranking(candidates, n)
        except Exception as e:
            print(f"Error during reranking: {e}")
            # Fallback to original ranking if LLM fails
            return self._fallback_ranking(candidates, n)
    
    def _fallback_ranking(self, candidates, n):
        """Create a fallback ranking when LLM reranking fails"""
        return [{
            "id": c["id"], 
            "level": c["metadata"].get("level", "unknown"),
            "title": c["metadata"].get("manga_title", "Unknown"),
            "identifier": self._create_identifier(c["metadata"]),
            "relevance_score": 100 - i*10,
            "explanation": "Ranked by vector similarity.",
            "match_type": "overall",
            "image_path": c["metadata"].get("image_path", ""),
            "vector_similarity": c.get("similarity", None)  # Include vector similarity
        } for i, c in enumerate(candidates[:n])]
    
    def _create_identifier(self, metadata):
        """Create a hierarchical identifier from metadata"""
        level = metadata.get("level", "unknown")
        manga_title = metadata.get("manga_title", "Unknown")
        page_number = metadata.get("page_number", "Unknown")
        panel_id = metadata.get("panel_id", "Unknown") if level == "panel" else ""
        
        identifier = f"{manga_title}"
        if level == "page":
            identifier += f" - Page {page_number}"
        elif level == "panel":
            identifier += f" - Page {page_number}, Panel {panel_id}"
        
        return identifier

    def search(self, query: str, n_results: int = 5, filter_level: str = None) -> List[Dict[Any, Any]]:
        """
        Search for manga based on a natural language query with LLM reranking.
        
        Args:
            query: User's natural language description
            n_results: Number of top results to return
            filter_level: Optional filter for level (book, page, panel)
            
        Returns:
            List of manga results with explanations
        """
        # First-stage retrieval: Get candidates from vector DB
        results = self.raw_search(query, n_results=n_results*2, filter_level=filter_level)
        
        if not results:
            return []
        
        # Second-stage re-ranking: Use LLM to rerank and explain
        ranked_results = self.rerank_results(query, results, n=n_results)
        
        return ranked_results
      
    def raw_search(self, query: str, n_results: int = 5, filter_level: str = None):
        """
        Perform a direct vector search without LLM reranking.
        Return raw results with similarity scores.
        
        Args:
            query: User's natural language description
            n_results: Number of top results to return
            filter_level: Optional filter for level (book, page, panel)
        """
        try:
            # Check if the collection exists
            collection_names = [c.name for c in self.chroma_client.list_collections()]
            
            if "manga_collection" not in collection_names:
                print("Error: manga_collection does not exist in the database")
                return []
            
            if self.collection.count() == 0:
                print("Warning: Collection is empty")
                return []
            
            # Prepare filter if needed
            where_filter = {"level": filter_level} if filter_level else None
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where=where_filter,
                include=["metadatas", "documents", "distances"]
            )
            
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
                    
                    # Calculate cosine similarity from distance
                    similarity = 1 - distance if distance is not None else None
                    
                    candidates.append({
                        "id": manga_id,
                        "metadata": metadata,
                        "content": document,
                        "similarity": similarity,
                        "distance": distance
                    })
                except IndexError as e:
                    print(f"Index error at position {i}: {e}")
            
            return candidates
        except Exception as e:
            print(f"Error retrieving candidates: {e}")
            import traceback
            traceback.print_exc()
            return []

def main():
    """Main function to test manga retrieval system."""
    print("Initializing manga retrieval system...")
    retriever = MangaRetrieval()
    
    print("\n===== Manga Retrieval System =====")
    print("\nEnter a description of what you're looking for:")
    user_query = input("> ") or "shaaake"
    print(f"user_query: {user_query}")
    
    if user_query:
        # Optional level filter
        print("\nFilter by level (leave blank for all):")
        print("1. Book level")
        print("2. Page level")
        print("3. Panel level")
        filter_choice = input("Choose (1-3 or blank): ")
        
        filter_level = None
        if filter_choice == "1":
            filter_level = "book"
        elif filter_choice == "2":
            filter_level = "page"
        elif filter_choice == "3":
            filter_level = "panel"
        
        print("\nSearching for matching manga...")
        ranked_results = retriever.search(user_query, filter_level=filter_level)
        
        if ranked_results:
            print("\n===== Search Results =====\n")
            for i, result in enumerate(ranked_results):
                print(f"{i+1}. {result['identifier']}")
                print(f"   Relevance Score: {result['relevance_score']}")
                print(f"   Vector Similarity: {result.get('vector_similarity', 'N/A'):.4f}" if result.get('vector_similarity') is not None else f"   Vector Similarity: N/A")
                print(f"   Match Type: {result.get('match_type', 'overall')}")
                print(f"   Image Path: {result.get('image_path', 'N/A')}")
                print(f"   {result['explanation']}")
                print()
        else:
            print("No matching results found.")
        
        print("-" * 50)

if __name__ == "__main__":
    main()