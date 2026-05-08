import json
import re
import os

def load_custom_txt(file_path):
    """
    读取自定义格式 txt: 
    <question>: ... <answer>: ...
    """
    data = []
    print(f"Loading custom QA data from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 简单的解析逻辑：根据 <answer>: 分割
            # 格式应该是: <question>: Q_CONTENT<answer>: A_CONTENT
            if "<answer>:" in line:
                parts = line.split("<answer>:")
                # parts[0] 是 "<question>: Q_CONTENT"
                # parts[1] 是 " A_CONTENT"
                
                # 去掉开头的 "<question>: "
                q_part = parts[0].replace("<question>: ", "").strip()
                a_part = parts[1].strip()
                
                data.append({
                    "question": q_part,
                    "answer": a_part
                })
            else:
                print(f"Skipping malformed line: {line[:50]}...")
    return data

def save_custom_txt(data, file_path):
    """
    保存为自定义格式:
    <question>: ...<answer>: ...
    """
    print(f"Saving to: {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        for sample in data:
            # 拼接回原始格式
            # 注意：根据你提供的样本，<answer>: 前面紧贴着 question 内容，或者有一个空格
            # 这里我们按照标准结构拼接
            line = f"<question>: {sample['question']}<answer>: {sample['answer']}\n"
            f.write(line)

def append_context_from_last_trajectory_poi(
    input_txt_path,
    aux_info_json_path,
    output_txt_path
):
    """
    核心逻辑：
    1. 解析自定义 txt。
    2. 找到 question 中最后一个 POI ID。
    3. 查表并在 "Given the data" 前插入提示语。
    4. 存回自定义 txt 格式。
    """

    # 1. 读取自定义格式 TXT
    qa_data = load_custom_txt(input_txt_path)

    # 2. 读取辅助信息 (这个还是 JSON)
    print(f"Loading Auxiliary info from: {aux_info_json_path}")
    with open(aux_info_json_path, "r", encoding="utf-8") as f:
        aux_data = json.load(f)

    processed_count = 0
    missing_info_count = 0
    target_phrase = "Given the data" # 定位锚点

    # 3. 逐样本处理
    for sample in qa_data:
        question_text = sample["question"]
        
        # 提取 question 中出现的所有 POI ID
        # 正则：匹配 "POI id" 后面的数字
        all_poi_ids = re.findall(r"POI id\s?(\d+)", question_text)
        
        if all_poi_ids:
            # 取最后一个，即 Trajectory 中的最后一次访问
            # 注意：all_poi_ids 包含 question 里所有的 ID，
            # 只要 "Given the data" 在最后，那么倒数第一个 ID 通常就是 Last Visit
            last_visit_poi_id = all_poi_ids[-1]
            
            # 检查 aux_data 是否包含该 ID
            if last_visit_poi_id in aux_data:
                info = aux_data[last_visit_poi_id]
                nearby_str = info.get("nearest", "none") 
                popular_str = info.get("popular", "none")
                
                # 构造符合要求的字符串
                context_str = (
                    f" For {last_visit_poi_id}, nearby POIs include: [{nearby_str}], "
                    f"For {last_visit_poi_id}, popular POIs include:[{popular_str}]."
                )
                
                # 【插入逻辑】
                if target_phrase in question_text:
                    # 在 "Given the data" 前面插入
                    # 结果变成: ... id 4186 ... [context_str] Given the data ...
                    new_question = question_text.replace(target_phrase, context_str + " " + target_phrase)
                    sample["question"] = new_question
                    processed_count += 1
                else:
                    # 如果没找到 Given the data，可以选择直接加在最后，或者跳过
                    pass
            else:
                missing_info_count += 1
        else:
            pass

    # 4. 保存为自定义格式 TXT
    save_custom_txt(qa_data, output_txt_path)

    print("-" * 30)
    print(f"✓ Processed {len(qa_data)} samples.")
    print(f"✓ Inserted context before '{target_phrase}' for {processed_count} samples.")
    if missing_info_count > 0:
        print(f"⚠  Auxiliary info missing for {missing_info_count} POIs.")
        
if __name__ == "__main__":
    # 输入文件路径
    original_txt_path = "../datasets/nyc/preprocessed/test_qa_pairs_kqt.txt"
    
    # 辅助信息文件 (JSON)
    aux_info_json_path = "../datasets/nyc/preprocessed/poi_auxiliary_info.json"
    
    # 输出文件路径
    output_txt_path = "../datasets/nyc/preprocessed/test_qa_pairs_can.txt"

    if not os.path.exists(aux_info_json_path):
        print(f"Error: Auxiliary file not found at {aux_info_json_path}")
    else:
        if not os.path.exists(original_txt_path):
             print(f"Error: Input file not found at {original_txt_path}")
        else:
            append_context_from_last_trajectory_poi(
                original_txt_path,
                aux_info_json_path,
                output_txt_path
            )