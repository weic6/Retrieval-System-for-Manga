import os
from google import genai
from google.genai import types

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
   
# prompt_page_level = """
# For each panel in the picture, finish the following micro-tasks:
# 1. describe the characters present and their expressions
# 2. describe the setting and background elements
# 3. describe what appears to be happening in this scene
# 4. describe any text or dialogue visible
# 5. describe the emotion and tone of the scene
# 6. summary of the panel

# After finishing all the micro-tasks for each panel, finish the final task:
# provide a page-level summary, based on these micro-tasks, of all panels in the page.
# """
    
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


def generate_multiturn_comic_analysis(image_paths):
    if not image_paths:
        print("没有找到任何图片文件")
        return

    client = genai.Client(
        api_key="AIzaSyAszN7F6QDqJg7kbYshDIk-yrpUJ4ce3jU",  
    )

    print(f"共找到 {len(image_paths)} 张图片：")
    for f in image_paths:
        print(" -", os.path.basename(f))

    uploaded_files = [client.files.upload(file=img) for img in image_paths]
    contents = []

    model = "gemini-2.0-flash"
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=8192,
    )
 
    for idx, uploaded_file in enumerate(uploaded_files):
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    ),
                    types.Part.from_text(text=prompt_page_level),#描述每一张漫画的prompt
                ]
            )
        )

        partial_response = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            partial_response += chunk.text

        print(f"\n 第 {idx + 1} 张图分析结果：\n{partial_response}\n")
        contents.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text = partial_response)]
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text= prompt_book_level)#总结整个漫画的prompt
            ]
        )
    )

    final_summary = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        final_summary += chunk.text

    print("\n Final summary:")
    print(final_summary)


if __name__ == "__main__":
    image_folder = "./manga_images"  # ← 指定你的漫画图所在文件夹
    image_files = get_image_files(image_folder)
    generate_multiturn_comic_analysis(image_files)
    
