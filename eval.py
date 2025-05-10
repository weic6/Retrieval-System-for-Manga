import json
import os
import query as RAG
from tqdm import tqdm

def read_query_data_from_json(filepath: str) -> list | None:
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Error: File {filepath} does not contain a valid JSON list.")
            return None


        print(f"成功从 {filepath} 读取数据。")
        return data

    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None

    except json.JSONDecodeError:
        print(f"Error: Not a valid JSON file: {filepath}")
        return None

    except Exception as e:
        print(f"Error: An unexpected error occurred while reading the file {filepath}: {e}")
        return None

def main():
    """Main function to test manga retrieval system."""
    print("Initializing manga retrieval system...")
    retriever = RAG.MangaRetrieval()
    
    # Load the query data
    json_file_path = "testset\\test_query.json"
    loaded_manga_data = read_query_data_from_json(json_file_path)

    TP = 0
    Total = len(loaded_manga_data)*5
    if loaded_manga_data is not None:
        for i, manga_entry in tqdm(enumerate(loaded_manga_data)):
            name = manga_entry.get('name')
            query_list = manga_entry.get('query_list')
            for j, query in enumerate(query_list):
                print(f"Manag Name: {name}, Query{j}: {query}")
                print("\n===== Manga Retrieval System =====")
                user_query = query
                print(f"user_query: {user_query}")
                
                if user_query:
                    filter_level = None
                    print("\nSearching for matching manga...")
                    ranked_results = retriever.search(user_query, filter_level=filter_level)
                    
                    T = False
                    if ranked_results:
                        print("\n===== Search Results =====\n")
                        for i, result in enumerate(ranked_results):
                            print(f"{i+1}. {result['identifier']}")
                            print(f"   Relevance Score: {result['relevance_score']}")
                            print(f"   Vector Similarity: {result.get('vector_similarity', 'N/A'):.4f}" if result.get('vector_similarity') is not None else f"   Vector Similarity: N/A")
                            print(f"   Match Type: {result.get('match_type', 'overall')}")
                            print(f"   Image Path: {result.get('image_path', 'N/A')}")
                            print(f"   {result['explanation']}")
                            if name in result['identifier']:
                                print(f"Match found")
                                T = True
                    else:
                        print("No matching results found.")
                print("-" * 50)
                if T:
                    TP += 1
        accuracy = TP / Total
        print(f"TP: {TP}, Total: {Total}, Accuracy: {accuracy:.4f}")

    else:
        print("Fail to load query data from JSON file.")


if __name__ == "__main__":
    main()
