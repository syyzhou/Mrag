import json
import re


# ===============================
# 1. 读取 predictions.json
# ===============================
def load_predictions(predictions_json_path):
    """
    从 predictions.json 中读取 sample_idx -> top_k_pois
    """
    with open(predictions_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pred_map = {}
    for item in data["predictions"]:
        pred_map[item["sample_idx"]] = item["top_k_pois"]

    print(f"✓ Loaded predictions for {len(pred_map)} samples")
    return pred_map


# ===============================
# 2. 处理 TXT QA（test）
# ===============================
def process_txt_with_predictions(
    original_txt_path,
    output_txt_path,
    pred_map,
):
    """
    对 TXT QA 文件：
    在 <answer>: 前插入 Candidate POIs
    """
    print(f"\n处理 TXT 文件: {original_txt_path}")

    with open(original_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    used = 0

    for idx, line in enumerate(lines):
        line = line.strip()

        # 非 QA 行 或 没有 prediction
        if "<answer>:" not in line or idx not in pred_map:
            new_lines.append(line)
            continue

        candidates = pred_map[idx]

        candidate_str = (
            f" Use the following candidate POIs as supplementary references "
            f"to refine your prediction. Candidate POIs:{candidates}."
        )

        question, answer = line.split("<answer>:", 1)
        new_line = f"{question}{candidate_str} <answer>:{answer}"
        new_lines.append(new_line)
        used += 1

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"✓ 成功插入候选样本数: {used}")
    print(f"✓ 输出保存到: {output_txt_path}")


# ===============================
# 3. 处理 JSON QA（train）
# ===============================
def process_json_with_predictions(
    original_json_path,
    output_json_path,
    pred_map,
):
    """
    对 JSON QA 文件：
    - 在 question 后追加 Candidate POIs
    - 额外保存 candidates 字段
    """
    print(f"\n处理 JSON 文件: {original_json_path}")

    with open(original_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []
    used = 0

    for idx, item in enumerate(data):
        # 没有对应 prediction
        if idx not in pred_map:
            new_data.append(item)
            continue

        candidates = pred_map[idx]

        candidate_str = (
            f" Use the following candidate POIs as supplementary references "
            f"to refine your prediction. Candidate POIs:{candidates}."
        )

        new_item = item.copy()
        new_item["question"] = item["question"] + candidate_str
        new_item["candidates"] = candidates

        new_data.append(new_item)
        used += 1

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 成功插入候选样本数: {used}")
    print(f"✓ 输出保存到: {output_json_path}")


# ===============================
# 4. main
# ===============================
def main():
    # ========= 路径配置 =========
    predictions_json = "../results/predictions.json"

    train_qa_json = "../datasets/nyc/preprocessed/train_qa_pairs_kqt.json"
    test_qa_txt = "../datasets/nyc/preprocessed/test_qa_pairs_kqt.txt"

    train_out_json = "../datasets/nyc/preprocessed/train_qa_pairs_kqt_candidates_me.json"
    test_out_txt = "../datasets/nyc/preprocessed/test_qa_pairs_kqt_candidates_me.txt"

    # ========= 加载 predictions =========
    pred_map = load_predictions(predictions_json)

    # ========= 处理训练集 =========
    process_json_with_predictions(
        train_qa_json,
        train_out_json,
        pred_map
    )

    # ========= 处理测试集 =========
    process_txt_with_predictions(
        test_qa_txt,
        test_out_txt,
        pred_map
    )

    print("\n✓ 全部处理完成")


if __name__ == "__main__":
    main()
