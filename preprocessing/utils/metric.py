import numpy as np
def recall(recommended_list, ground_truth, k=10):
    """ 计算 Recall@K """
    hits = sum(1 for item in recommended_list[:k] if item in ground_truth)
    return hits / len(ground_truth) if ground_truth else 0

def dcg(recommended_list, ground_truth, k=10):
    """ 计算 DCG@K """
    return sum((1 / np.log2(idx + 2)) for idx, item in enumerate(recommended_list[:k]) if item in ground_truth)

def ndcg(recommended_list, ground_truth, k=10):
    """ 计算 NDCG@K """
    ideal_dcg = dcg(sorted(ground_truth, reverse=True), ground_truth, k)
    return dcg(recommended_list, ground_truth, k) / ideal_dcg if ideal_dcg > 0 else 0

def average_precision(recommended_list, ground_truth, k=10):
    """ 计算 MAP@K """
    hits = 0
    sum_precisions = 0
    for i, item in enumerate(recommended_list[:k]):
        if item in ground_truth:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / min(len(ground_truth), k) if ground_truth else 0

def mrr(recommended_list, ground_truth, k=10):
    """ 计算 MRR@K """
    for i, item in enumerate(recommended_list[:k]):
        if item in ground_truth:
            return 1 / (i + 1)
    return 0
