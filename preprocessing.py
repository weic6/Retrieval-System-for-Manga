import os
import json
import datetime
import google.generativeai as genai  # Changed import pattern
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

prompt_page_level = """ 
You are analysing a single manga **page** that consists of multiple panels.
For **each panel**, output a JSON object that follows EXACTLY this schema
(note: omit comments):

{
  "panel_id": "string",           // e.g. "1", "2-A"
  "characters": [
    {
      "name": "string",           // use the canonical name **if** it appears in the dialogue;
                                  // otherwise fallback to role labels like "Boy", "Girl".
      "expression": "string",     // e.g. "smiling", "annoyed"
      "pose": "string"            // e.g. "crossed arms"; empty string if unclear
    }
  ],
  "setting": {
    "location": "string",         // e.g. "classroom", "outdoors"
    "background_elements": [ "string" ] // props or scenery
  },
  "narrative": {
    "actions":  [ "string" ],     // **MUST** include at least one verb
    "dialogue": [ "string" ],     // sentence-case, strip extra ALL-CAPS styling,
                                  // one line per bubble in reading order
    "emotion":  "string"          // overall scene tone, e.g. "tense"
  },
  "text_elements": [ "string" ],  // onomatopoeia, signage, UI text
  "summary": "string"             // 1–2 concise sentences
}

After you output every panel object in an array, add a top-level key
"page_summary" with a single-paragraph summary of the whole page.

Return **only** valid JSON – no markdown, no comments.
"""
    
prompt_book_level = """
Based on all the comic panels you've analyzed, please provide a paragraph, which covers the following aspects:
1. A complete summary of the comic's narrative
2. The main characters and their development
3. The overall theme or message
4. The artistic style throughout the comic
5. Any significant cultural or contextual elements
"""

def get_image_files(folder_path):
    """
    从指定文件夹中读取所有 .jpg/.jpeg/.png 文件，按文件名排序。
    返回完整路径的文件列表。
    """
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return image_files

def get_manga_folders(root_folder):
    """
    获取manga_images目录下所有子文件夹，每个子文件夹代表一本漫画。
    """
    manga_folders = []
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path):
            manga_folders.append(item_path)
    return sorted(manga_folders)

def generate_multiturn_comic_analysis(image_paths, manga_name):
    """
    分析单本漫画的所有页面
    
    Args:
        image_paths: 一本漫画的所有图片路径
        manga_name: 漫画名称（文件夹名）
    """
    if not image_paths:
        print(f"漫画 '{manga_name}' 没有找到任何图片文件")
        return

    # Configure the API
    genai.configure(api_key=API_KEY)

    print(f"漫画 '{manga_name}' 共找到 {len(image_paths)} 张图片：")
    for f in image_paths:
        print(" -", os.path.basename(f))

    # Initialize a list to store results for each image
    results = []

    # Create a generative model instance
    gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_output_tokens": 8192,
    }
    
    # Initialize content for conversation
    conversation = []
 
    for idx, image_path in enumerate(image_paths):
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Create image part and text part
        image_part = {"mime_type": "image/jpeg", "data": image_data}
        
        # Generate content with streaming for page analysis
        partial_response = ""
        for chunk in gemini_model.generate_content(
            [image_part, prompt_page_level],
            generation_config=generation_config,
            stream=True
        ):
            if hasattr(chunk, 'text'):
                partial_response += chunk.text

        print(f"\n {manga_name} 第 {idx + 1} 张图分析结果：\n{partial_response}\n")
        
        # Store image path and page-level summary
        results.append({
            "manga_name": manga_name,
            "image_path": image_paths[idx],
            "page_number": idx + 1,
            "summary_page_level": partial_response,
            "summary_book_level": ""  # Will be filled later
        })
        
        # Add model response to conversation for book-level analysis
        conversation.append({"role": "user", "parts": [image_part, prompt_page_level]})
        conversation.append({"role": "model", "parts": [{"text": partial_response}]})

    # Add book-level prompt
    conversation.append({"role": "user", "parts": [{"text": prompt_book_level}]})

    # Create a conversation for book-level analysis
    chat = gemini_model.start_chat(history=conversation)
    
    # Generate book-level summary
    final_response = chat.send_message(prompt_book_level)
    final_summary = final_response.text

    print(f"\n {manga_name} 总体分析结果:")
    print(final_summary)
    
    # Add the book-level summary to each result
    for result in results:
        result["summary_book_level"] = final_summary
    
    # Save results to a JSON file with manga name
    output_dir = os.path.join(os.path.dirname(os.path.dirname(image_paths[0])), "manga_analyses")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{os.path.basename(manga_name)}_analysis.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{manga_name} 分析结果已保存到 {output_file}")
    
    return results

def process_all_manga(manga_root_folder):
    """
    处理所有漫画文件夹
    """
    manga_folders = get_manga_folders(manga_root_folder)
    
    if not manga_folders:
        print(f"在 {manga_root_folder} 中没有找到任何漫画文件夹")
        return
    
    print(f"找到 {len(manga_folders)} 本漫画需要处理:")
    for folder in manga_folders:
        print(f" - {os.path.basename(folder)}")
    
    all_results = {}
    
    for manga_folder in manga_folders:
        manga_name = os.path.basename(manga_folder)
        print(f"\n开始处理漫画: {manga_name}")
        
        image_files = get_image_files(manga_folder)
        if image_files:
            manga_results = generate_multiturn_comic_analysis(image_files, manga_name)
            all_results[manga_name] = manga_results
        else:
            print(f"漫画 '{manga_name}' 文件夹中没有找到图片文件")
    
    # 保存所有漫画的汇总信息
    output_dir = os.path.join(manga_root_folder, "manga_analyses")
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "all_manga_summary.json")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_manga": len(all_results),
            "manga_titles": list(all_results.keys()),
            "creation_date": str(datetime.datetime.now())
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有漫画处理完成。汇总信息保存至 {summary_file}")


if __name__ == "__main__":
    manga_root_folder = "./manga_images"  # 包含多个漫画子文件夹的根目录
    process_all_manga(manga_root_folder)