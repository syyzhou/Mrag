import json

# 文件路径
file1 = "../datasets/nyc/preprocessed/train_qa_pairs_kqt_point.json"  # TXT 文件
file2 = "../datasets/nyc/preprocessed/train_qa_pairs_kqt_pair.json"  # JSON 文件
output_file = "merged.json"

def load_txt(file_path):
    """读取 TXT 文件，每行可能是一个 QA 对"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    # 去掉空行
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def load_json(file_path):
    """读取 JSON 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def unify_format(data):
    """
    将不同格式的数据统一成 [{'question': ..., 'answer': ...}, ...] 的形式
    """
    unified = []
    for item in data:
        # 已经是字典形式
        if isinstance(item, dict) and "question" in item and "answer" in item:
            unified.append({
                "question": item["question"].strip(),
                "answer": item["answer"].strip()
            })
        # 文本形式 '<question>: ... <answer>: ...'
        elif isinstance(item, str):
            try:
                q_split = item.split("<answer>:")
                question = q_split[0].replace("<question>:", "").strip()
                answer = q_split[1].strip()
                unified.append({
                    "question": question,
                    "answer": answer
                })
            except IndexError:
                # 格式异常
                print("Warning: skipped malformed item:", item[:100])
    return unified

# 加载文件
data_txt = load_json(file1)  # TXT
data_json = load_json(file2)  # JSON

# 统一格式
data_txt_unified = unify_format(data_txt)
data_json_unified = unify_format(data_json)

# 合并
merged_data = data_json_unified + data_txt_unified

# 保存到新 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"Merged {len(merged_data)} records into {output_file}")
