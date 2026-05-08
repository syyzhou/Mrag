# enrich_candidates.py

import json
import re
import os
import csv
import math
from collections import defaultdict
from typing import Dict, List, Optional


def load_txt_qa_format(path: str):
    """
    解析这种格式：
    <question>: ... Candidate POIs:[...]. <answer>: ...
    """
    import re

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = r"<question>:\s*(.*?)\s*<answer>:\s*(.*?)(?=<question>:|$)"
    matches = re.findall(pattern, text, re.DOTALL)

    data = []
    for q, a in matches:
        q = q.strip()
        a = a.strip()

        # 🔥 解析 candidates
        cand_match = re.search(r"Candidate POIs:\[(.*?)\]", q)
        if cand_match:
            candidates = list(map(int, cand_match.group(1).split(",")))
        else:
            candidates = []

        data.append({
            "question": q,
            "answer": a,
            "candidates": candidates
        })

    return data
def load_data(path: str):
    """自动识别json/txt格式加载"""
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    elif path.endswith('.txt'):
        # ❗改这里
        return load_txt_qa_format(path)

    else:
        # fallback
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return load_txt_qa_format(path)

def save_data(data, path: str):
    """根据后缀选择保存格式"""
    if path.endswith('.json'):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif path.endswith('.txt'):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(f"<question>: {item['question']}\n")
                f.write(f"<answer>: {item['answer']}\n")
    else:
        # 默认jsonl
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def build_poi_info_from_csv(csv_path: str) -> Dict:
    poi_coords = defaultdict(lambda: {'lats': [], 'lngs': [], 'category': '', 'cat_id': -1})
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poi_id = int(row['PoiId'])
            lat = float(row['Latitude'])
            lng = float(row['Longitude'])
            cat_name = row['PoiCategoryName'].strip()
            cat_id = int(row['PoiCategoryId'])
            
            poi_coords[poi_id]['lats'].append(lat)
            poi_coords[poi_id]['lngs'].append(lng)
            poi_coords[poi_id]['category'] = cat_name
            poi_coords[poi_id]['cat_id'] = cat_id
    
    poi_info = {}
    for poi_id, data in poi_coords.items():
        poi_info[poi_id] = {
            'category': data['category'],
            'cat_id': data['cat_id'],
            'lat': sum(data['lats']) / len(data['lats']),
            'lng': sum(data['lngs']) / len(data['lngs']),
        }
    
    print(f"Loaded {len(poi_info)} POIs from CSV")
    return poi_info


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def format_distance(dist_km: float) -> str:
    if dist_km < 1.0:
        return f"{dist_km * 1000:.0f}m"
    elif dist_km < 10.0:
        return f"{dist_km:.1f}km"
    else:
        return f"{dist_km:.0f}km"


def parse_last_poi_in_current_trajectory(question: str) -> Optional[int]:
    current_match = re.search(
        r"\[Current trajectory's check-in sequence\]:(.*?)(?:\[Historical|Given the data,)",
        question, re.DOTALL
    )
    current_text = current_match.group(1) if current_match else question
    matches = re.findall(r'visited POI id (\d+)', current_text)
    return int(matches[-1]) if matches else None


def build_candidate_text(
    candidates: List[int],
    last_poi_id: Optional[int],
    poi_info: Dict,
) -> str:
    last_lat, last_lng = None, None
    last_poi_name = ""
    if last_poi_id is not None and last_poi_id in poi_info:
        last_lat = poi_info[last_poi_id]['lat']
        last_lng = poi_info[last_poi_id]['lng']
        last_poi_name = f"POI {last_poi_id}"
    
    parts = []
    for cand_id in candidates:
        if cand_id in poi_info:
            cat = poi_info[cand_id]['category']
            if last_lat is not None:
                dist = haversine_km(last_lat, last_lng,
                                    poi_info[cand_id]['lat'], poi_info[cand_id]['lng'])
                parts.append(f"POI {cand_id}({cat}, {format_distance(dist)})")
            else:
                parts.append(f"POI {cand_id}({cat})")
        else:
            parts.append(f"POI {cand_id}(Unknown)")
    
    if last_lat is not None:
        instruction = (
            "Use the following candidate POIs as supplementary references "
            "to refine your prediction. Each candidate is shown as "
            "POI id(category, distance), where category is the POI's "
            f"business type and distance is measured from the last visited "
            f"{last_poi_name} in the current trajectory. "
        )
    else:
        instruction = (
            "Use the following candidate POIs as supplementary references "
            "to refine your prediction. Each candidate is shown as "
            "POI id(category), where category is the POI's business type. "
        )
    
    return instruction + "Candidate POIs:[" + ", ".join(parts) + "]."


def replace_candidate_in_question(question: str, new_text: str) -> str:
    pattern = (
        r'Use the following candidate POIs as supplementary references '
        r'to refine your prediction\.\s*'
        r'Candidate POIs:\[[^\]]*\]\.'
    )
    match = re.search(pattern, question)
    if match:
        return question[:match.start()] + new_text + question[match.end():]
    
    pattern2 = r'Candidate POIs:\[[^\]]*\]\.'
    match2 = re.search(pattern2, question)
    if match2:
        return question[:match2.start()] + new_text + question[match2.end():]
    
    print(f"  WARNING: Could not find candidate text to replace")
    return question


def enrich_dataset(input_path: str, output_path: str, csv_path: str):
    print(f"Loading POI info from {csv_path}...")
    poi_info = build_poi_info_from_csv(csv_path)
    
    print(f"Loading dataset from {input_path}...")
    dataset = load_data(input_path)
    print(f"Total samples: {len(dataset)}")
    print(f"Input format: {os.path.splitext(input_path)[1]}")
    print(f"Output format: {os.path.splitext(output_path)[1]}")
    
    enriched = []
    stats = {'total': 0, 'replaced': 0, 'no_match': 0, 'no_candidates': 0}
    
    for idx, item in enumerate(dataset):
        question = item['question']
        candidates = item.get('candidates', [])
        stats['total'] += 1
        
        if not candidates:
            stats['no_candidates'] += 1
            enriched.append(item)
            continue
        
        last_poi = parse_last_poi_in_current_trajectory(question)
        new_cand_text = build_candidate_text(candidates, last_poi, poi_info)
        new_question = replace_candidate_in_question(question, new_cand_text)
        
        if new_question != question:
            stats['replaced'] += 1
        else:
            stats['no_match'] += 1
        
        result = {'question': new_question, 'answer': item['answer']}
        if 'candidates' in item:
            result['candidates'] = item['candidates']
        enriched.append(result)
        
        if idx < 2:
            print(f"\n{'='*80}")
            print(f"Sample {idx}")
            print(f"{'='*80}")
            print(f"Last POI in current trajectory: {last_poi}", end="")
            if last_poi and last_poi in poi_info:
                p = poi_info[last_poi]
                print(f" -> {p['category']} at ({p['lat']:.4f}, {p['lng']:.4f})")
            else:
                print()
            
            print(f"\n[BEFORE]:")
            old_match = re.search(
                r'Use the following.*?Candidate POIs:\[[^\]]*\]\.',
                question
            )
            if old_match:
                print(f"  {old_match.group()[:200]}...")
            
            print(f"\n[AFTER]:")
            print(f"  {new_cand_text[:300]}...")
            print(f"{'='*80}")
        
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx+1}/{len(dataset)}")
    
    print(f"\n{'='*60}")
    print(f"Total: {stats['total']}")
    print(f"Replaced: {stats['replaced']}")
    print(f"No match: {stats['no_match']}")
    print(f"No candidates: {stats['no_candidates']}")
    print(f"{'='*60}")
    
    save_data(enriched, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()
    
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_enriched{ext}"
    
    enrich_dataset(args.input, args.output, args.csv)