import os
import json
import base64
import re
import time
import asyncio
import httpx
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================
# 1. API 客户端配置 (OpenRouter)
# ============================
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    # 明确配置异步 HTTP 客户端，挂载本地代理
    http_client=httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(proxy="http://127.0.0.1:7897"),
        timeout=180.0
    )
)

class Config:
    MODEL_NAME = "openai/gpt-4.1"
    # 【并发控制】
    MAX_CONCURRENT_REQUESTS = 20


# ============================
# 路径配置
# ============================
FOLDER_2024 = r"E:\nochangeall\last_2024_RGB"
FOLDER_2025 = r"E:\nochangeall\last_2025_RGB"
KNOWLEDGE_MD = r"E:\AAA-VLM\MYdataset\vlm\land-use.md"
OUTPUT_JSONL = r"output\gpt41_answer_nochangeall.jsonl"


# ============================
# 2. 辅助工具函数
# ============================
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def load_knowledge(md_path):
    if not os.path.exists(md_path):
        return "Standard Land Classification Table"
    with open(md_path, "r", encoding="utf8") as f:
        return f.read()

def extract_json(text):
    if not text:
        return None
    try:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                content = text[start:end + 1]
            else:
                return None
        return json.loads(content)
    except Exception:
        return None

def format_time(seconds: float) -> str:
    """格式化秒数为 HH:MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}时{m:02d}分{s:02d}秒"
    return f"{m:02d}分{s:02d}秒"

def build_prompt(task_id, knowledge_text):
    prompt = f"""
### Reference Knowledge

{knowledge_text}

### Task ID

{task_id}

You are a remote sensing expert. Given two co-registered bi-temporal images (T1, T2), explain why this region does **not** constitute an INTER_L2_CHANGE.

All observed differences must be explained as either:

1. **Pseudo-change**: caused by illumination, shadows, phenology, seasonality, or imaging conditions.
2. **NO_CHANGE**: changes within the same secondary land-use class (L2) without functional land-use transition.

Your goal is to prove that no INTER_L2_CHANGE exists.

Return **only** valid JSON in this format:

{{
  "thought_process": {{
    "step_1_global_perception": "",
    "step_2_instance_visual": "",
    "step_3_relational_model": "",
    "step_4_reasoning": "",
    "step_5_future_inference": "",
    "step_6_confidence": {{
      "score": 0,
      "justification": "",
      "limitations": "",
      "alternative_l2_candidates": []
    }}
  }},
  "answer": ""
}}

### Output Rules

- Avoid redundant and meaningless content; only include 1–2 sentences of key details.
- Focus only on evidence needed to justify **pseudo-change / NO_CHANGE / no INTER_L2_CHANGE**.

### Field Instructions

- **step_1_global_perception**: Briefly summarize the overall scene type and whether land-use function remains stable from T1 to T2.
- **step_2_instance_visual**: Identify only the main apparently changed regions and briefly state the key visual evidence and likely L2 class.
- **step_3_relational_model**: Analyze whether T1 and T2 preserve the same land-use logic from four aspects:
  1. spatial relations,
  2. physical interactions,
  3. functional relations,
  4. usage relations.
- **step_4_reasoning**: State why the observed differences are pseudo-changes or within-class NO_CHANGE, and why they do not support INTER_L2_CHANGE.
- **step_5_future_inference**: Use one short sentence to infer the likely short-term continuation of land use or function.
- **step_6_confidence**: 
- Give a confidence score (0-10) and one brief reason.
- Base difficulty:
  - Easy: construction land, roads, rivers/lakes/ponds/reservoirs → 8–9
  - Medium: paddy field, general forest land, rural construction land → 7–8
  - Hard: dryland vs. irrigated land, forest subtypes, orchard, grassland subtypes → 5–6
  - Return:

- `score`: integer 0–10
- `justification`: key reasons
- `limitations`: major uncertainty sources
- `alternative_l2_candidates`: plausible alternative L2 classes, or `[]`

### Answer

Write **one short professional paragraph** summarizing:

- the main observed differences,
- why no INTER_L2_CHANGE exists,
- and the final confidence.

### Constraints

- The final conclusion must clearly state that the region contains no INTER_L2_CHANGE.
- Output must be strictly valid JSON.
  """
    return prompt.strip()


# ============================
# 3. 异步并发核心调用 (带有防断流与重试)
# ============================
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True
)
async def call_model_with_retry_async(messages):
    """带重试机制的异步 API 调用，使用流式接收防 OpenRouter 网关超时"""
    response = await client.chat.completions.create(
        model=Config.MODEL_NAME,
        messages=messages,
        stream=True,    # 【核心】保留流式输出防静默断联
        temperature=0,  # 事实推理任务设为0
        timeout=180
    )

    full_content = ""
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content

    return full_content


async def process_single_task(prefix, img_pre_path, img_post_path, prompt_content, semaphore, write_lock, counter_info, output_jsonl_abs):
    """处理单一图像对的异步任务"""
    record = {"task_id": prefix, "success": False, "model_response": None}

    async with semaphore:
        try:
            # 获取并发锁后读取图片防止 OOM
            img1_b64 = encode_image(img_pre_path)
            img2_b64 = encode_image(img_post_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_content},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}}
                    ]
                }
            ]

            full_content = await call_model_with_retry_async(messages)
            parsed_json = extract_json(full_content)

            if parsed_json:
                record["success"] = True
                record["model_response"] = parsed_json
            else:
                record["raw_output"] = full_content

        except Exception as e:
            record["raw_output"] = f"Error: {str(e)}"

    # ================= 写入落盘与进度计算 =================
    async with write_lock:
        with open(output_jsonl_abs, "a", encoding="utf8") as out_f:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    counter_info['processed'] += 1
    processed = counter_info['processed']
    total = counter_info['total']
    start_time = counter_info['start_time']

    elapsed = time.time() - start_time
    avg_time_per_task = elapsed / processed
    remaining = avg_time_per_task * (total - processed)
    est_total = elapsed + remaining

    status_tag = "✅ 成功" if record["success"] else "❌ 失败/无效JSON"
    print(f"\n[+] 进度: [{processed}/{total}] | 任务: {prefix} | 状态: {status_tag}")

    if not record["success"]:
        error_detail = record.get("raw_output", "未知错误")
        print(f"    ⚠️ 错误详情: {error_detail[:200]}...")

    print(f"    ⏳ 耗时统计: [已耗时: {format_time(elapsed)}] | [预计剩余: {format_time(remaining)}] | [预计总共: {format_time(est_total)}]")


# ============================
# 4. 异步主程序
# ============================
async def main_async():
    print(f"[*] 开始执行 {Config.MODEL_NAME} (No-Change) 数据标注任务 (异步并发断点续传模式)...")

    knowledge_text = load_knowledge(KNOWLEDGE_MD)

    if not os.path.exists(FOLDER_2024):
        print(f"[-] 错误：找不到前时相文件夹 {FOLDER_2024}")
        return

    files_2024 = sorted([f for f in os.listdir(FOLDER_2024) if "_2024_RGB.png" in f])
    print(f">>> 扫描到 {len(files_2024)} 个 T1 图像文件。")

    # 【防缺失设计】预索引 T2 文件夹
    print(">>> 正在为 T2 文件夹建立索引防缺失...")
    all_post_files = os.listdir(FOLDER_2025)
    post_map = {}
    for f in all_post_files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            p = f.split('_')[0]
            post_map[p] = f

    output_dir = os.path.dirname(OUTPUT_JSONL)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    OUTPUT_JSONL_ABS = os.path.abspath(OUTPUT_JSONL)

    # ================= 【断点续传核心逻辑】 =================
    processed_ids = set()
    existing_records = {}

    if os.path.exists(OUTPUT_JSONL_ABS):
        with open(OUTPUT_JSONL_ABS, "r", encoding="utf8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    task_id = data.get("task_id")
                    if task_id:
                        existing_records[task_id] = data
                        if data.get("success"):
                            processed_ids.add(task_id)
                except:
                    continue
        print(f"[*] 发现历史记录：共找到 {len(processed_ids)} 个已成功的任务。")

        # 覆写文件，清除失败留下的垃圾数据
        records_to_write = [existing_records[tid] for tid in processed_ids]
        with open(OUTPUT_JSONL_ABS, "w", encoding="utf8") as out_f:
            for record in records_to_write:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[*] 已清理输出文件，移除了历史失败任务的脏数据，准备重新发起。")
    # ========================================================

    task_params_list = []

    for f_name in files_2024:
        prefix = f_name.split("_2024_RGB")[0]

        # 仅跳过成功的任务
        if prefix in processed_ids:
            continue

        img_pre_path = os.path.join(FOLDER_2024, f_name)
        matched_post_name = post_map.get(prefix)

        if not matched_post_name:
            print(f"    [-] 缺失后时相图像 {prefix}，跳过该任务。")
            continue

        img_post_path = os.path.join(FOLDER_2025, matched_post_name)

        # No-Change 任务不需要 task_metadata
        prompt_content = build_prompt(prefix, knowledge_text)

        task_params_list.append({
            "prefix": prefix,
            "img_pre_path": img_pre_path,
            "img_post_path": img_post_path,
            "prompt_content": prompt_content
        })

    total_tasks = len(task_params_list)
    print(f"[*] 🚀 本次实际需要投递 {total_tasks} 个图像解析请求...")

    if total_tasks == 0:
        print("[*] 🎉 所有任务均已成功标注完成！")
        return

    # 初始化并发控制器和全局锁
    semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
    write_lock = asyncio.Lock()

    counter_info = {
        'processed': 0,
        'total': total_tasks,
        'start_time': time.time()
    }

    async_tasks = []
    for params in task_params_list:
        task = asyncio.create_task(process_single_task(
            prefix=params['prefix'],
            img_pre_path=params['img_pre_path'],
            img_post_path=params['img_post_path'],
            prompt_content=params['prompt_content'],
            semaphore=semaphore,
            write_lock=write_lock,
            counter_info=counter_info,
            output_jsonl_abs=OUTPUT_JSONL_ABS
        ))
        async_tasks.append(task)

    # 启动全量并发
    await asyncio.gather(*async_tasks)

    total_time_cost = time.time() - counter_info['start_time']
    print(f"\n[*] 🎉 {Config.MODEL_NAME} (No-Change) 异步批量数据标注脚本执行完毕！")
    print(f"[*] 🏆 本次运行共完成 {total_tasks} 个请求，实际总耗时: {format_time(total_time_cost)}")
    print(f"[*] 💾 结果已全部保存至：{OUTPUT_JSONL_ABS}")


if __name__ == "__main__":
    asyncio.run(main_async())