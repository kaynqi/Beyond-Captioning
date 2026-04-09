import json
import os
from typing import List, Dict


# ==========================================
# 1. 配置与辅助函数
# ==========================================
def load_json(file_path: str) -> dict:
    if not os.path.exists(file_path):
        print(f"[-] 文件不存在: {file_path}")
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: any, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[+] 偏好对齐数据集已保存至: {file_path}")


def safe_float(value, default=0.0):
    """安全转换为浮点数，处理 'Err' 或 None 的情况"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ==========================================
# 2. 合并与排序逻辑
# ==========================================
def merge_and_rank(stage1_path: str, stage2_path: str, stage3_path: str) -> List[dict]:
    print("[*] 正在加载各阶段评估结果...")
    vlm_results = load_json(stage1_path)
    lmm_results = load_json(stage2_path)
    rule_results = load_json(stage3_path)

    if not vlm_results or not lmm_results or not rule_results:
        print("[-] 错误：存在未生成的评估结果，请检查前面的步骤是否完成。")
        return []

    # 1. 收集所有的模型名称和任务 ID
    models = list(vlm_results.keys())
    all_tasks = set()
    for model in models:
        all_tasks.update(vlm_results[model].keys())
        all_tasks.update(lmm_results.get(model, {}).keys())
        all_tasks.update(rule_results.get(model, {}).keys())

    preference_dataset = []

    # 2. 按任务进行遍历并合并
    for task_id in sorted(list(all_tasks)):
        task_candidates = []

        for model in models:
            s1_data = vlm_results.get(model, {}).get(task_id, {}).get('judge_1_eval', {})
            s2_data = lmm_results.get(model, {}).get(task_id, {}).get('judge_2_eval', {})
            s3_data = rule_results.get(model, {}).get(task_id, {}).get('judge_3_eval', {})

            # 提取各项分数 (若发生错误或缺失，默认给 0 分作为惩罚)
            step1_score = safe_float(s1_data.get('step_1_score', 0))
            step2_score = safe_float(s1_data.get('step_2_score', 0))
            step3_score = safe_float(s2_data.get('step_3_score', 0))
            step4_logic_score = safe_float(s2_data.get('step_4_logic_score', 0))
            step4_rule_score = safe_float(s3_data.get('step_4_rule_score', 0))
            step5_score = safe_float(s2_data.get('step_5_score', 0))
            step6_score = safe_float(s3_data.get('step_6_score', 0))

            # 计算综合过程分 (Process Score)，最高 10 分
            # 您可以根据重点调节这里的权重，这里采用算数平均
            process_score = (step1_score + step2_score + step3_score +
                             step4_logic_score + step4_rule_score +
                             step5_score + step6_score) / 7.0

            # 提取最终结果分 (Hard Score) 与正样本标记
            answer_score = safe_float(s3_data.get('answer_match_score', 0))
            is_positive_sample = s3_data.get('is_positive_sample', False)

            task_candidates.append({
                "model_name": model,
                "answer_score": answer_score,
                "process_score": round(process_score, 2),
                "is_positive_sample": is_positive_sample,
                "detailed_scores": {
                    "step_1": step1_score,
                    "step_2": step2_score,
                    "step_3": step3_score,
                    "step_4_logic": step4_logic_score,
                    "step_4_rule": step4_rule_score,
                    "step_5": step5_score,
                    "step_6": step6_score
                },
                "reasoning_feedback": {
                    "vlm_feedback": s1_data,
                    "lmm_feedback": s2_data,
                    "rule_feedback": s3_data
                }
            })

        # 3. 双层排序逻辑
        # 第一层：最终答案对错 (answer_score 降序)
        # 第二层：推理过程严谨度 (process_score 降序)
        task_candidates.sort(key=lambda x: (x['answer_score'], x['process_score']), reverse=True)

        # 4. 构建 DPO/RLAIF 偏好对 (Chosen & Rejected)
        if not task_candidates:
            continue

        best_candidate = task_candidates[0]

        # 严格把控 Chosen 的质量：只有答案正确且过程分尚可（如 > 6.0 分），才配作为 Chosen
        # 否则这个 task 被认为是极其困难的（所有模型全军覆没），可以打上标记
        has_valid_chosen = (best_candidate['answer_score'] >= 10 and best_candidate['process_score'] > 6.0)

        rejected_candidates = task_candidates[1:]

        preference_dataset.append({
            "task_id": task_id,
            "has_valid_chosen": has_valid_chosen,
            "chosen": best_candidate if has_valid_chosen else None,
            "rejected": rejected_candidates,
            # 将排序后的完整列表也保留下来，方便如果您想做多候选偏好排序 (Plackett-Luce) 或 ORM/PRM 训练
            "ranked_candidates": task_candidates
        })

    return preference_dataset


# ==========================================
# 3. 主执行流程
# ==========================================
if __name__ == "__main__":
    # ------------------ 用户配置区 ------------------
    STAGE1_PATH = "judge/judge-output-change/judge-1-vlm-batch-results-gemini-change.json"
    STAGE2_PATH = "judge/judge-output-change/judge-2-lmm-batch-results-async-deepseek.json"
    STAGE3_PATH = "judge/judge-output-change/judge-3-rule-batch-results-async-gpt4o.json"

    # 输出的 DPO/RLAIF 对齐训练数据集路径
    OUTPUT_PREFERENCE_DATASET = "judge/judge-output-change/preference-alignment-dataset.json"
    # ------------------------------------------------

    print("[*] 开始合并数据并生成偏好排序...")
    dataset = merge_and_rank(STAGE1_PATH, STAGE2_PATH, STAGE3_PATH)

    # 统计信息
    total_tasks = len(dataset)
    valid_chosen_count = sum(1 for item in dataset if item["has_valid_chosen"])

    print("\n========== 合并与统计报告 ==========")
    print(f"总计评估任务 (Tasks): {total_tasks}")
    print(f"拥有高质量 Chosen 样本的任务数: {valid_chosen_count} (占比: {valid_chosen_count / total_tasks * 100:.1f}%)")

    if total_tasks > 0:
        # 打印第一个任务的排名示例
        sample = dataset[0]
        print(f"\n[示例] Task ID: {sample['task_id']}")
        for rank, cand in enumerate(sample['ranked_candidates'], 1):
            tag = "(Chosen)  " if rank == 1 and sample['has_valid_chosen'] else "(Rejected)"
            print(
                f"  Rank {rank} {tag} - {cand['model_name']} | 结果分: {cand['answer_score']} | 过程分: {cand['process_score']}")

    # 落盘
    save_json(dataset, OUTPUT_PREFERENCE_DATASET)