import json
import os
import query as RAG
from tqdm import tqdm

def read_query_data_from_json(filepath: str) -> list | None:
    """
    从指定的 JSON 文件读取漫画数据。

    Args:
        filepath: JSON 文件的完整路径。

    Returns:
        如果文件成功读取且格式正确，返回一个包含漫画数据的列表。
        如果文件未找到、格式错误或发生其他读取错误，返回 None。
        返回的列表结构为：
        [
            {"name": "漫画名称A", "query_list": ["query1", "query2", ...]},
            {"name": "漫画名称B", "query_list": ["queryA", "queryB", ...]},
            ...
        ]
    """
    # 可选：先检查文件是否存在
    if not os.path.exists(filepath):
        print(f"错误：文件未找到，请检查路径是否正确: {filepath}")
        return None

    try:
        # 以读取模式 ('r') 打开文件，指定 UTF-8 编码
        with open(filepath, 'r', encoding='utf-8') as f:
            # 使用 json.load() 直接从文件对象中读取并解析 JSON
            data = json.load(f)

        # 可选：检查读取到的数据是否是预期的列表格式
        if not isinstance(data, list):
            print(f"错误：文件 {filepath} 的 JSON 内容不是预期的列表格式。")
            return None

        # 如果需要，你还可以进一步检查列表中的每个元素是否是包含 'name' 和 'query_list' 的字典

        print(f"成功从 {filepath} 读取数据。")
        return data # 返回解析后的 Python 列表

    except FileNotFoundError:
        # 实际上上面的 os.path.exists 已经检查了，但这里保留作为备用或清理
        print(f"错误：文件未找到: {filepath}")
        return None

    except json.JSONDecodeError:
        # 如果文件内容不是有效的 JSON 格式，会捕获这个错误
        print(f"错误：文件内容不是有效的 JSON 格式，请检查文件内容: {filepath}")
        return None

    except Exception as e:
        # 捕获其他可能的异常，例如权限问题等
        print(f"读取文件时发生意外错误 {filepath}: {e}")
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
                print(f"漫画名称: {name}, 查询键{j}: {query}")
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
        print("未能成功读取漫画数据。")


if __name__ == "__main__":
    main()
