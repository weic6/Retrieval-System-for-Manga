import os
import json
import datetime
import google.generativeai as genai  # Changed import pattern
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Helper function to clean JSON responses from markdown formatting
def clean_json_response(response_text):
    """
    Remove markdown code block formatting if present.
    """
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

# Updated prompt to match the specific schema requirements
prompt_page_level = """ 
You are analyzing a manga page image. Output a JSON object that follows EXACTLY this schema:

{
  "page_number": integer,     // The page number 
  "image_path": "string",     // Path to the image file
  "summary": "string",        // 1-2 sentence gist of this page
  "panels": [                 // Array of panels in reading order
    {
      "panel_id": "string",   // e.g. "1", "2-A"
      "characters": [
        {
          "name": "string",     // canonical name if it appears in dialogue; otherwise role labels like "Boy", "Girl"
          "expression": "string", // e.g. "smiling", "annoyed"
          "pose": "string"      // e.g. "crossed arms"
        }
      ],
      "setting": {
        "location": "string",   // e.g. "classroom", "outdoors"
        "background_elements": ["string"] // props or scenery
      },
      "narrative": {
        "actions": ["string"],  // MUST include at least one verb per entry
        "dialogue": ["string"], // One bubble per string, cleaned to sentence case
        "emotion": "string"     // overall scene tone, e.g. "tense"
      },
      "text_elements": ["string"], // onomatopoeia, signage, UI text
      "summary": "string"       // 1-2 concise sentences
    }
  ]
}

Return ONLY valid JSON – no markdown, no code blocks, no triple backticks, just the raw JSON object.
"""
    
prompt_book_level = """
Based on all the manga pages you've analyzed, create a complete manga book JSON object that follows EXACTLY this schema:

{
  "manga_name": "string",    // The name of the manga
  "summary": "string",       // Overall summary of the manga
  "pages": [                 // Array of all analyzed pages
    // Each page object you've already analyzed
  ]
}

Your response should be a single, complete JSON object that combines all the pages you've analyzed into a coherent manga book object.
Return ONLY valid JSON – no markdown, no code blocks, no triple backticks, just the raw JSON object.
"""

def get_image_files(folder_path):
    """
    从指定文件夹中读取所有 .jpg/.jpeg/.png/.webp 文件，按文件名排序。
    返回完整路径的文件列表。
    """
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])
    return image_files

def get_mime_type(file_path):
    """
    Determine the MIME type based on file extension
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.webp':
        return "image/webp"
    elif ext in ['.jpg', '.jpeg']:
        return "image/jpeg"
    elif ext == '.png':
        return "image/png"
    else:
        # Default to jpeg for unknown types
        return "image/jpeg"

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

def generate_schema_compliant_manga(image_paths, manga_name):
    """
    分析单本漫画的所有页面，并生成符合schema的JSON对象
    
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

    # Initialize a list to store page objects
    page_objects = []

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
        
        # Create image part with correct MIME type
        mime_type = get_mime_type(image_path)
        image_part = {"mime_type": mime_type, "data": image_data}
        
        # Modified prompt to include specific page information
        page_specific_prompt = prompt_page_level + f"\nThis is page {idx + 1} and the image path is '{image_path}'"
        
        # Generate content with streaming for page analysis
        partial_response = ""
        for chunk in gemini_model.generate_content(
            [image_part, page_specific_prompt],
            generation_config=generation_config,
            stream=True
        ):
            if hasattr(chunk, 'text'):
                partial_response += chunk.text

        print(f"\n{manga_name} 第 {idx + 1} 张图分析结果：\n{partial_response[:500]}...\n")
        
        try:
            # Clean the response and then parse the JSON
            cleaned_response = clean_json_response(partial_response)
            page_object = json.loads(cleaned_response)
            page_objects.append(page_object)
            
            # Add model response to conversation for book-level analysis
            conversation.append({"role": "user", "parts": [image_part, page_specific_prompt]})
            conversation.append({"role": "model", "parts": [{"text": partial_response}]})
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for page {idx + 1}: {e}")
            print("Response was:", partial_response[:1000])
            print("Cleaned response was:", cleaned_response[:1000])
    
    # Skip book-level processing if no valid pages were found
    if not page_objects:
        print(f"No valid page objects were parsed for manga '{manga_name}'. Skipping book-level analysis.")
        return None
        
    # Add book-level prompt to create the final manga object
    final_prompt = prompt_book_level + f"\nThe manga name is '{manga_name}' and it has {len(page_objects)} pages."
    conversation.append({"role": "user", "parts": [{"text": final_prompt}]})

    # Create a conversation for book-level analysis
    chat = gemini_model.start_chat(history=conversation)
    
    # Generate complete manga object
    final_response = chat.send_message(final_prompt)
    manga_json_text = final_response.text
    
    try:
        # Clean and parse the final manga object
        cleaned_manga_json = clean_json_response(manga_json_text)
        manga_object = json.loads(cleaned_manga_json)
        
        print(f"\n{manga_name} 生成完整manga对象成功")
        
        # Save results to a JSON file with manga name
        output_dir = os.path.join(os.path.dirname(os.path.dirname(image_paths[0])), '..',"manga_analyses")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{os.path.basename(manga_name)}_schema.json")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(manga_object, f, ensure_ascii=False, indent=2)
        
        print(f"\n{manga_name} schema对象已保存到 {output_file}")
        
        return manga_object
    except json.JSONDecodeError as e:
        print(f"Error parsing final manga JSON: {e}")
        print("Response was:", manga_json_text[:1000])
        print("Cleaned response was:", cleaned_manga_json[:1000])
        return None

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
    
    all_manga_objects = {}
    
    for manga_folder in manga_folders:
        manga_name = os.path.basename(manga_folder)
        print(f"\n开始处理漫画: {manga_name}")
        
        image_files = get_image_files(manga_folder)
        if image_files:
            manga_object = generate_schema_compliant_manga(image_files, manga_name)
            if manga_object:
                all_manga_objects[manga_name] = manga_object
        else:
            print(f"漫画 '{manga_name}' 文件夹中没有找到图片文件")
    
    # 保存所有漫画的汇总信息
    output_dir = os.path.join(manga_root_folder,"..", "manga_analyses")
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "all_manga_schema_summary.json")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_manga": len(all_manga_objects),
            "manga_titles": list(all_manga_objects.keys()),
            "creation_date": str(datetime.datetime.now())
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有漫画处理完成。汇总信息保存至 {summary_file}")


if __name__ == "__main__":
    manga_root_folder = "./manga_images"  # 包含多个漫画子文件夹的根目录
    process_all_manga(manga_root_folder)