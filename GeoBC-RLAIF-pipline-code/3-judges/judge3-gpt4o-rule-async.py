import json
import os
import httpx
import asyncio
import time
from openai import AsyncOpenAI

# ==========================================
# 1. 配置与初始化
# ==========================================
# 使用 AsyncOpenAI 客户端，并挂载本地代理
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    # 建议后续改为 os.getenv("OPENROUTER_API_KEY")
    http_client=httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(
            proxy="http://127.0.0.1:7897"
        ),
        timeout=120
    )
)


class Config:
    RULE_MODEL = "openai/gpt-4o-2024-05-13"
    # 【并发控制】GPT-4o 并发能力较强，可设置为 10-20
    MAX_CONCURRENT_REQUESTS = 20


# ==========================================
# 2. 辅助函数
# ==========================================
def load_text_file(file_path: str) -> str:
    """读取文本知识库文件 (如 markdown)"""
    if not os.path.exists(file_path):
        print(f"[-] 知识库文件不存在: {file_path}")
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_json(file_path: str) -> dict:
    """读取 Ground Truth JSON 或历史断点文件"""
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model_data(file_path: str) -> dict:
    """兼容读取 JSONL 格式的模型输出数据"""
    if not os.path.exists(file_path):
        print(f"[-] 文件不存在: {file_path}")
        return {}

    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    task_id = item.get("task_id")
                    if task_id:
                        data_dict[str(task_id)] = item.get("model_response", {})
                except Exception as e:
                    print(f"[-] JSONL 解析错误: {e}")
        else:
            raw_data = json.load(f)
            for k, v in raw_data.items():
                task_id = os.path.splitext(k)[0]
                data_dict[str(task_id)] = v

    return data_dict


def save_json(data: dict, file_path: str):
    """保存 JSON 结果"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        # 【重点】加入 sort_keys=True，保证断点续跑、乱序并发后的结果依然是按 task_id 排序的
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_existing_results(file_path: str) -> dict:
    """加载已存在的结果文件，用于断点续跑"""
    if not os.path.exists(file_path):
        print(f"[*] 未找到历史结果文件：{file_path}，将从头开始执行")
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[+] 找到历史结果文件，共包含 {sum(len(v) for v in data.values())} 条结果")

        failed_count = 0
        failed_tasks = []
        for model_name, model_results in data.items():
            for task_id, result in model_results.items():
                if not result or 'error' in result:
                    failed_count += 1
                    failed_tasks.append(f"{model_name}/{task_id}")

        if failed_count > 0:
            print(f"[!] 发现 {failed_count} 条失败记录，将重新执行这些任务")
            if failed_count <= 10:
                print(f"    失败任务列表：{', '.join(failed_tasks)}")
        else:
            print("[*] 所有历史记录均为成功结果。")

        return data
    except Exception as e:
        print(f"[-] 加载历史结果失败：{e}，将从头开始执行")
        return {}


def extract_json_from_response(text: str) -> dict:
    """清理大模型输出的 Markdown 符号并解析 JSON"""
    try:
        text = text.strip()
        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0].strip()
        elif text.startswith("```"):
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception as e:
        return {"error": f"JSON 解析失败: {str(e)}", "raw": text}


def format_time(seconds: float) -> str:
    """格式化秒数为 HH:MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}时{m:02d}分{s:02d}秒"
    return f"{m:02d}分{s:02d}秒"


# ==========================================
# 3. 异步 Rule 评委核心调用逻辑
# ==========================================
async def run_rule_judge_async(gt_answer: str, step_4_text: str, step_6_json: dict, model_answer: str,
                               land_use_rules: str, semaphore: asyncio.Semaphore) -> dict:
    prompt = f"""
[System]
You are a Strict Rule-Based Compliance AI Judge for Remote Sensing Classification. You evaluate based on rigid Land-Use Classification Standards (Level 1 and Level 2). Your role is an inflexible, dogmatic AI prosecutor.

[Task]
Evaluate the model's adherence to classification rules in `step_4`, the academic rigor in `step_6`, and the absolute correctness of the final `answer` compared to the Ground Truth (GT).

[Evaluation Criteria]
1. Step 4 (Rule Compliance): Does the change involve an effective Level 2 semantic shift according to the taxonomy? Heavily penalize if the model considers an intra-class Level 2 change (e.g., A to B, but both belong to the same Level 2 category) as a valid semantic change.
2. Step 6 (Confidence & Rigor): Does the model exhibit academic rigor? Does it correctly identify classification ambiguities lacking multi-source data? Are the limitations and alternative candidates reasonable?
3. Final Answer Match: Compare the model's final answer with the Ground Truth Short Answer. This is a HARD METRIC (0 or 10).
4. Positive Sample Flag: Based on the above, determine if this entire response qualifies as a high-quality "Positive Sample" for preference alignment training.

[Input Context]
- Land-Use Taxonomy Rules: 
\"\"\"
{land_use_rules}
\"\"\"
- GT Short Answer: {gt_answer}

[Model's Predictions]
- Model Step 4 (Reasoning CoT): {step_4_text}
- Model Step 6 (Confidence & Rigor): {json.dumps(step_6_json, ensure_ascii=False)}
- Model Final Answer: {model_answer}

[Output Format]
Provide your evaluation in strict JSON format exactly like this:
{{
  "judge_3_eval": {{
    "step_4_rule_score": [Score 0-10],
    "step_4_rule_reasoning": "[1-2 sentences checking strict adherence to Level 1/Level 2 taxonomy rules]",
    "step_6_score": [Score 0-10],
    "step_6_reasoning": "[1-2 sentences evaluating scientific rigor, ambiguity handling, and valid limitations]",
    "answer_match_score": [0 or 10],
    "answer_reasoning": "[1 sentence: 0 for incorrect, 10 for strict semantic match with GT]",
    "is_positive_sample": [true or false]
  }}
}}
"""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=Config.RULE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                # GPT-4o 完美支持强制 JSON 模式
                response_format={"type": "json_object"}
            )
            result_text = response.choices[0].message.content
            return extract_json_from_response(result_text)
        except Exception as e:
            return {"error": str(e)}


async def process_single_task(model_name, clean_task_id, current_gt_answer, step_4_text, step_6_json,
                              model_final_answer, land_use_rules_text, semaphore, rule_evaluations, counter_info,
                              output_save_path):
    # 发起评估请求
    eval_result = await run_rule_judge_async(
        gt_answer=current_gt_answer,
        step_4_text=step_4_text,
        step_6_json=step_6_json,
        model_answer=model_final_answer,
        land_use_rules=land_use_rules_text,
        semaphore=semaphore
    )

    # 写入字典
    rule_evaluations[model_name][clean_task_id] = eval_result

    # 时间与进度计算逻辑
    counter_info['processed'] += 1
    processed = counter_info['processed']
    total = counter_info['total']
    start_time = counter_info['start_time']

    elapsed = time.time() - start_time
    avg_time_per_task = elapsed / processed
    remaining = avg_time_per_task * (total - processed)
    est_total = elapsed + remaining

    s4_rule = eval_result.get('judge_3_eval', {}).get('step_4_rule_score', 'Err')
    s6 = eval_result.get('judge_3_eval', {}).get('step_6_score', 'Err')
    ans_score = eval_result.get('judge_3_eval', {}).get('answer_match_score', 'Err')
    is_pos = eval_result.get('judge_3_eval', {}).get('is_positive_sample', False)

    print(
        f"\n[+] 进度: [{processed}/{total}] | 模型: {model_name} | 任务: {clean_task_id} 评估完成 -> Step4(规则): {s4_rule}, Step6(置信): {s6}, Answer: {ans_score}, 正样本: {is_pos}")
    print(
        f"    ⏳ 耗时统计: [本次已耗时: {format_time(elapsed)}] | [预计剩余: {format_time(remaining)}] | [预计本次总需: {format_time(est_total)}]")

    # 每 20 个请求保存一次中间结果
    if processed % 20 == 0:
        save_json(rule_evaluations, output_save_path)
        print(f"    💾 [系统提示] 已自动保存中间结果...")


# ==========================================
# 4. 异步主执行流程
# ==========================================
async def main_async():
    # ------------------ 用户配置区 ------------------
    LAND_USE_MD_PATH = r"land-use.md"
    # GT_JSON_PATH = r"nochange_regions_merged_test.json"
    GT_JSON_PATH = r"E:\AAA-VLM\MYdataset\vlm\answer\change_regions_merged-train.json"

    # MODEL_JSON_PATHS = [
    #     "judge/gemini_answer_nochange.jsonl",
    #     "judge/gpt41_answer_nochange.jsonl",
    #     "judge/llava_answer_nochange.jsonl",
    #     "judge/qvq_answer_nochange.jsonl",
    #     "judge/qwen_answer_nochange.jsonl"
    # ]
    MODEL_JSON_PATHS = [
        "judge/gemini_answer_change.jsonl",
        "judge/gpt41_answer_change.jsonl",
        "judge/llava_answer_change.jsonl",
        "judge/qvq_answer_change.jsonl",
        "judge/qwen_answer_change.jsonl"
    ]

    OUTPUT_SAVE_PATH = "judge/judge-output/judge_3_rule_batch_results_async_gpt4o_nochange-val.json"
    # ------------------------------------------------

    print("[*] 开始执行 Rule 规则匹配评委 (Stage 3) 批量打分阶段 (异步断点续传模式)...")

    land_use_rules_text = load_text_file(LAND_USE_MD_PATH)
    if not land_use_rules_text:
        print("[-] 警告: 土地利用标准知识库未加载，可能会严重影响评判质量！")

    gt_data_all = load_json(GT_JSON_PATH)

    models_data = {}
    for path in MODEL_JSON_PATHS:
        model_name = os.path.splitext(os.path.basename(path))[0]
        models_data[model_name] = load_model_data(path)
        print(f"[+] 成功加载模型结果: {model_name} (共包含 {len(models_data[model_name])} 条任务)")

    # ================= 【断点续传核心逻辑】 =================
    rule_evaluations = load_existing_results(OUTPUT_SAVE_PATH)
    for name in models_data.keys():
        if name not in rule_evaluations:
            rule_evaluations[name] = {}
    # ========================================================

    all_task_ids = list(gt_data_all.keys())
    print(f"[*] 共在 GT 数据中找到 {len(all_task_ids)} 个任务待评估。")

    task_params_list = []
    already_completed_count = 0

    for task_id in all_task_ids:
        # 注意此处的 task_id 截取逻辑保持一致，若需5位可改为 [:5]
        clean_task_id = str(task_id).split('.')[0][:6]
        current_gt_data = gt_data_all.get(task_id, {})
        current_gt_answer = current_gt_data.get("answer", "")

        for model_name, model_output_all in models_data.items():

            # --- 【检查断点】 ---
            existing_result = rule_evaluations.get(model_name, {}).get(clean_task_id)
            if existing_result and "error" not in existing_result:
                already_completed_count += 1
                continue
            # --------------------

            current_model_data = model_output_all.get(clean_task_id)
            if not current_model_data:
                continue

            thought_process = current_model_data.get("thought_process", {})
            step_4_text = thought_process.get("step_4_reasoning", "")
            step_6_json = thought_process.get("step_6_confidence", {})
            model_final_answer = current_model_data.get("answer", "")

            if not step_4_text and not step_6_json and not model_final_answer:
                continue

            task_params_list.append({
                "model_name": model_name,
                "clean_task_id": clean_task_id,
                "current_gt_answer": current_gt_answer,
                "step_4_text": step_4_text,
                "step_6_json": step_6_json,
                "model_final_answer": model_final_answer
            })

    total_tasks = len(task_params_list)
    print(f"[*] 任务检查完毕！已跳过 {already_completed_count} 个已成功完成的任务。")
    print(f"[*] 🚀 本次实际需要投递 {total_tasks} 个 API 请求...")

    if total_tasks == 0:
        print("[*] 🎉 所有任务均已评估完成，无需执行任何请求！")
        return

    semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
    counter_info = {
        'processed': 0,
        'total': total_tasks,
        'start_time': time.time()
    }

    async_tasks = []
    for params in task_params_list:
        task = asyncio.create_task(process_single_task(
            model_name=params['model_name'],
            clean_task_id=params['clean_task_id'],
            current_gt_answer=params['current_gt_answer'],
            step_4_text=params['step_4_text'],
            step_6_json=params['step_6_json'],
            model_final_answer=params['model_final_answer'],
            land_use_rules_text=land_use_rules_text,
            semaphore=semaphore,
            rule_evaluations=rule_evaluations,
            counter_info=counter_info,
            output_save_path=OUTPUT_SAVE_PATH
        ))
        async_tasks.append(task)

    # 启动全量并发
    await asyncio.gather(*async_tasks)

    # 收尾
    total_time_cost = time.time() - counter_info['start_time']
    save_json(rule_evaluations, OUTPUT_SAVE_PATH)
    print(f"\n[*] 🎉 阶段三 (Rule & Answer 异步批量评估) 脚本执行完毕！")
    print(f"[*] 🏆 本次运行成功补充了 {total_tasks} 个评估，实际总耗时: {format_time(total_time_cost)}")


if __name__ == "__main__":
    asyncio.run(main_async())