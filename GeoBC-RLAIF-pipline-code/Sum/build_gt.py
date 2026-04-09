import json
import os

def build_final_gt(ranking_file, model_files):
    # 1. 加载排名决策数据
    with open(ranking_file, 'r', encoding='utf-8') as f:
        rankings = json.load(f)

    # 2. 将五个模型的所有内容加载到内存中
    model_data_cache = {}
    
    for m_file in model_files:
        # 【修正点】：使用 os.path.basename 只取文件名，再去掉 .jsonl
        # 这样不管路径多深，得到的都是 gemini_answer_nochange
        base_name = os.path.basename(m_file)
        model_name = base_name.replace(".jsonl", "") 
        
        print(f"正在索引模型 [{model_name}] 的文件: {base_name} ...")
        
        if not os.path.exists(m_file):
            print(f"错误：找不到文件 {m_file}")
            continue

        with open(m_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        t_id = entry.get("task_id")
                        # 缓存 Key: 模型名_任务ID
                        model_data_cache[f"{model_name}_{t_id}"] = entry
                    except json.JSONDecodeError:
                        continue

    # 3. 根据排名第一的模型提取数据
    final_gt_list = []
    missing_count = 0

    print("-" * 30)
    print("开始合成最终版本...")
    for task_id, candidates in rankings.items():
        if not candidates:
            continue
        
        # 获取排名第一的模型名称
        best_model = candidates[0]["model_name"]
        cache_key = f"{best_model}_{task_id}"
        
        if cache_key in model_data_cache:
            # 提取原始数据
            final_entry = model_data_cache[cache_key]
            
            # 把决策信息也存进去（可选）
            final_entry["selection_info"] = {
                "source_model": best_model,
                "score": candidates[0].get("total_score")
            }
            final_gt_list.append(final_entry)
        else:
            print(f"警告: 找不到任务 {task_id} 在模型 {best_model} 中的原始数据 (Key: {cache_key})")
            missing_count += 1

    # 4. 保存最终结果
    output_filename = "/home/hanmz/yq/workspace/deepseek/deepseek/judge/judge-output/final_gt.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_gt_list, f, indent=4, ensure_ascii=False)
    
    print("-" * 30)
    print(f"完成！最终文件已保存为: {output_filename}")
    print(f"成功合成: {len(final_gt_list)} 条")
    if missing_count > 0:
        print(f"缺失条目: {missing_count} 条 (请检查模型名称是否在 JSONL 和 排名JSON 中一致)")

# --- 配置部分 ---
model_files_list = [
    "judge/nochange-answer/gemini_answer_nochange.jsonl",
    "judge/nochange-answer/gpt41_answer_nochange.jsonl",
    "judge/nochange-answer/llava_answer_nochange.jsonl",
    "judge/nochange-answer/qvq_answer_nochange.jsonl",
    "judge/nochange-answer/qwen_answer_nochange.jsonl"
]

ranking_json_path = "judge/judge-output/final_ranking_decision.json"

build_final_gt(ranking_json_path, model_files_list)