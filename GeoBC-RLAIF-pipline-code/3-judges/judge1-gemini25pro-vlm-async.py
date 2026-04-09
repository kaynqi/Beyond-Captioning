import json
import base64
import os
import httpx
import asyncio
import time
from openai import AsyncOpenAI

# ==========================================
# 1. 配置与初始化
# ==========================================
# 使用 AsyncOpenAI 替代 OpenAI，并设置异步的 httpx 客户端
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key= os.getenv("OPENROUTER_API_KEY"),
    http_client=httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(
            proxy="http://127.0.0.1:7890"
        ),
        timeout=120
    )
)


class Config:
    # 评委模型配置
    VLM_MODEL =  "google/gemini-3.1-pro-preview" # 或者 "openai/gpt-5.4-vision" 视您的实际模型定
    # 并发控制：由于图片处理较大，建议并发数控制在 5-10 之间
    MAX_CONCURRENT_REQUESTS = 10


# ==========================================
# 2. 辅助函数
# ==========================================
def encode_image_to_base64(image_path: str) -> str:
    """将图像转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_json(file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model_data(file_path: str) -> dict:
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
                except Exception:
                    pass
        else:
            raw_data = json.load(f)
            for k, v in raw_data.items():
                task_id = os.path.splitext(k)[0]
                data_dict[str(task_id)] = v
    return data_dict


def save_json(data: dict, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        # 强制按 task_id 排序输出
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def extract_json_from_response(text: str) -> dict:
    try:
        text = text.strip()
        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0].strip()
        elif text.startswith("```"):
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception as e:
        return {"error": "JSON parse failed", "raw": text}


def format_time(seconds: float) -> str:
    """格式化时间为 HH:MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}时{m:02d}分{s:02d}秒"
    return f"{m:02d}分{s:02d}秒"


# ==========================================
# 3. 异步 VLM 评委核心调用逻辑
# ==========================================
async def run_vlm_judge_async(pre_img_b64: str, post_img_b64: str, gt_json: dict, step_1_text: str, step_2_text: str,
                              semaphore: asyncio.Semaphore) -> dict:
    prompt = f"""
[System]
You are an expert Vision-Language AI Judge specializing in Remote Sensing (RS) and Multi-Modal Object Grounding. 
Your task is to evaluate the visual perception accuracy of a model's response describing changes between two bi-temporal remote sensing images.

[Task]
Compare the provided Pre-image, Post-image, and the Ground Truth (GT) Semantic JSON against the Model's Output for `step_1_global_perception` and `step_2_instance_visual`.

[Evaluation Criteria]
1. Step 1 (Global Perception): Does the model accurately describe the general scene and correctly couple visual attributes with semantic land-use definitions based on the GT? 
2. Step 2 (Instance Visual): Are the descriptive attributes (color, texture, shape, size) accurate and strictly grounded to the visible instances in the images? 
Penalty: Heavily penalize any "visual hallucinations" (describing objects or changes that do not exist in the images or GT).

[Input]
- GT Semantic JSON: {json.dumps(gt_json, ensure_ascii=False)}
- Model Step 1: {step_1_text}
- Model Step 2: {step_2_text}

[Output Format]
Provide your evaluation in strict JSON format exactly like this:
{{
  "judge_1_eval": {{
    "step_1_score": [Score 0-10],
    "step_1_reasoning": "[1-2 sentences explaining the score]",
    "step_2_score": [Score 0-10],
    "step_2_reasoning": "[1-2 sentences explaining the score]",
    "hallucination_flag": [true/false]
  }}
}}
"""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=Config.VLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pre_img_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{post_img_b64}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
            result_text = response.choices[0].message.content
            return extract_json_from_response(result_text)
        except Exception as e:
            return {"error": str(e)}


async def process_single_task(model_name, task_id, pre_path, post_path, gt_semantics, step_1_text, step_2_text,
                              semaphore, vlm_evaluations, counter_info, output_save_path):
    # 动态加载图像（避免一次性全部读取导致内存爆满）
    try:
        pre_b64 = encode_image_to_base64(pre_path)
        post_b64 = encode_image_to_base64(post_path)
    except Exception as e:
        print(f"[-] 图像读取失败: {e}")
        return

    eval_result = await run_vlm_judge_async(
        pre_img_b64=pre_b64,
        post_img_b64=post_b64,
        gt_json=gt_semantics,
        step_1_text=step_1_text,
        step_2_text=step_2_text,
        semaphore=semaphore
    )

    # 保存至字典
    vlm_evaluations[model_name][task_id] = eval_result

    # 进度与时间推算
    counter_info['processed'] += 1
    processed = counter_info['processed']
    total = counter_info['total']
    start_time = counter_info['start_time']

    elapsed = time.time() - start_time
    avg_time = elapsed / processed
    remaining = avg_time * (total - processed)
    est_total = elapsed + remaining

    if isinstance(eval_result, list):
        s1 = eval_result[0].get('judge_1_eval', {}).get('step_1_score', 'Err') if eval_result else 'Err'
        s2 = eval_result[0].get('judge_1_eval', {}).get('step_2_score', 'Err') if eval_result else 'Err'
    elif isinstance(eval_result, dict):
        s1 = eval_result.get('judge_1_eval', {}).get('step_1_score', 'Err')
        s2 = eval_result.get('judge_1_eval', {}).get('step_2_score', 'Err')
    else:
        s1 = 'Err'
        s2 = 'Err'

    print(
        f"\n[+] 进度：[{processed}/{total}] | 模型：{model_name} | 任务：{task_id} 评估完成 -> Step1:{s1}, Step2:{s2}")
    print(
        f"    ⏳ 耗时统计: [已运行: {format_time(elapsed)}] | [预计剩余: {format_time(remaining)}] | [预计本次总需: {format_time(est_total)}]")

    # 自动保存中间结果
    if processed % 15 == 0:
        save_json(vlm_evaluations, output_save_path)
        print(f"    💾 [系统提示] 已自动保存中间结果...")


# ==========================================
# 4. 异步主执行流程
# ==========================================
async def main_async():
    # ------------------ 用户配置区 ------------------

    PRE_IMG_DIR = r"G:\ct\our_dataset\train\change\2024_RGB"
    POST_IMG_DIR = r"G:\ct\our_dataset\train\change\2025_RGB"
    GT_JSON_PATH = r"F:\AAA-VLM\MYdataset\vlm\answer\change_regions_merged-train.json"

    MODEL_JSON_PATHS = [
        r"output\pro-train\benchmark_gemini_answer_change-train.jsonl",
        r"output\pro-train\benchmark_gpt41_answer_change-async-train.jsonl",
        r"output\pro-train\benchmark_llava_answer_change_train.jsonl",
        r"output\pro-train\benchmark_qvq_answer_change_train.jsonl",
        r"output\pro-train\benchmark_qwen_answer_change_train.jsonl"
    ]
    # 评委 1 (VLM) 打分结果的保存路径
    OUTPUT_SAVE_PATH = "judge/judge-output/judge_1_vlm_batch_results-gemini-change.json"
    # ------------------------------------------------

    print("[*] 开始执行 VLM 评委批量打分阶段 (异步并发与断点续传模式)...")

    gt_data_all = load_json(GT_JSON_PATH)

    models_data = {}
    for path in MODEL_JSON_PATHS:
        model_name = os.path.splitext(os.path.basename(path))[0]
        models_data[model_name] = load_model_data(path)
        print(f"[+] 成功加载模型结果: {model_name} (共包含 {len(models_data[model_name])} 条任务)")

    # ================= 【断点续传核心逻辑开始】 =================
    if os.path.exists(OUTPUT_SAVE_PATH):
        print(f"[*] 发现已存在的输出文件 {OUTPUT_SAVE_PATH}，正在读取断点数据...")
        vlm_evaluations = load_json(OUTPUT_SAVE_PATH)
        for name in models_data.keys():
            if name not in vlm_evaluations:
                vlm_evaluations[name] = {}
    else:
        print(f"[*] 未发现历史输出文件，将从头开始建立全新评估。")
        vlm_evaluations = {name: {} for name in models_data.keys()}
    # ================= 【断点续传核心逻辑结束】 =================

    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    if not os.path.exists(PRE_IMG_DIR):
        raise FileNotFoundError(f"前时相文件夹不存在: {PRE_IMG_DIR}")

    image_filenames = [f for f in os.listdir(PRE_IMG_DIR) if f.lower().endswith(valid_extensions)]

    task_params_list = []
    already_completed_count = 0

    for img_filename in image_filenames:
        # 注意：这里根据您实际的文件名提取长度可能为 [:5] 或者是 [:6]
        # 请确保与您的 GT 里的 key 长度一致
        task_id = os.path.splitext(img_filename)[0][:6]

        pre_path = os.path.join(PRE_IMG_DIR, img_filename)
        # 根据您之前的代码逻辑替换年份
        post_filename = img_filename.replace('_2024_', '_2025_')
        post_path = os.path.join(POST_IMG_DIR, post_filename)

        if not os.path.exists(post_path):
            continue

        current_gt_data = gt_data_all.get(task_id) or gt_data_all.get(img_filename, {})
        current_gt_semantics = current_gt_data.get("semantics", {})

        for model_name, model_output_all in models_data.items():
            # --- 【检查该任务是否已经完成】 ---
            existing_result = vlm_evaluations.get(model_name, {}).get(task_id)
            if existing_result and "error" not in existing_result:
                already_completed_count += 1
                continue
            # -----------------------------------

            current_model_data = model_output_all.get(task_id)
            if not current_model_data:
                continue

            thought_process = current_model_data.get("thought_process", {})
            step_1_text = thought_process.get("step_1_global_perception", "")
            step_2_text = thought_process.get("step_2_instance_visual", "")

            # 如果步骤文本全为空，跳过
            if not step_1_text and not step_2_text:
                continue

            task_params_list.append({
                "model_name": model_name,
                "task_id": task_id,
                "pre_path": pre_path,
                "post_path": post_path,
                "gt_semantics": current_gt_semantics,
                "step_1_text": step_1_text,
                "step_2_text": step_2_text
            })

    total_tasks = len(task_params_list)
    print(f"[*] 任务检查完毕！已跳过 {already_completed_count} 个已完成的任务。")
    print(f"[*] 🚀 本次实际需要投递 {total_tasks} 个 API 请求...")

    if total_tasks == 0:
        print("[*] 🎉 所有视觉评估任务均已完成，无需执行任何请求！")
        return

    # 初始化并发控制器与共享计数器
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
            task_id=params['task_id'],
            pre_path=params['pre_path'],
            post_path=params['post_path'],
            gt_semantics=params['gt_semantics'],
            step_1_text=params['step_1_text'],
            step_2_text=params['step_2_text'],
            semaphore=semaphore,
            vlm_evaluations=vlm_evaluations,
            counter_info=counter_info,
            output_save_path=OUTPUT_SAVE_PATH
        ))
        async_tasks.append(task)

    # 执行全部异步任务
    await asyncio.gather(*async_tasks)

    # 完成后计算总耗时并作最后一次保存
    total_time_cost = time.time() - counter_info['start_time']
    save_json(vlm_evaluations, OUTPUT_SAVE_PATH)

    print(f"\n[*] 🎉 阶段一 (VLM 异步批量评估) 脚本执行完毕！")
    print(f"[*] 🏆 本次运行成功补充了 {total_tasks} 个评估，实际总耗时: {format_time(total_time_cost)}")


if __name__ == "__main__":
    # 使用 asyncio 启动
    asyncio.run(main_async())