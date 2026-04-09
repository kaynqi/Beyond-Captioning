import json

def process_final_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果数据是单条对象，先转成列表处理
    if isinstance(data, dict):
        data = [data]
        
    ranking_decision = {}

    for entry in data:
        task_id = entry.get("task_id")
        candidates = entry.get("ranked_candidates", [])
        
        # 重新计算并排序（确保万无一失）
        scored_list = []
        for model in candidates:
            a_score = model.get("answer_score", 0.0)
            p_score = model.get("process_score", 0.0)
            scored_list.append({
                "model_name": model.get("model_name"),
                "total_score": round(a_score + p_score, 2),
                "answer_score": a_score,
                "process_score": p_score,
                "is_positive": model.get("is_positive_sample", False)
            })
        
        # 排序规则：总分降序 -> 结果分降序
        scored_list.sort(key=lambda x: (x['total_score'], x['answer_score']), reverse=True)
        
        ranking_decision[task_id] = scored_list

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ranking_decision, f, indent=4, ensure_ascii=False)

# 使用方法
process_final_json('/home/hanmz/yq/workspace/deepseek/deepseek/judge/judge-output-change/preference-alignment-dataset.json', '/home/hanmz/yq/workspace/deepseek/deepseek/judge/judge-output-change/final_ranking_decision.json')