import os
import json
import base64
import re
import time
import asyncio
import httpx
from openai import AsyncOpenAI  # 使用异步客户端
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================
# 1. API 客户端配置
# ============================
client = AsyncOpenAI(
    # 强烈建议：运行完成后去控制台重置此 Key，并改用环境变量读取
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # 明确配置异步 HTTP 客户端，禁用代理防拦截，并设置长超时
    http_client=httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(proxy=None),
        timeout=180.0
    )
)


class Config:
    MODEL_NAME = "qwen2.5-vl-72b-instruct"
    # 【并发控制】QvQ 系列作为高负载的推理模型，建议并发维持在 3，避免触发阿里限流 (429)
    MAX_CONCURRENT_REQUESTS = 8


# 保留第一个脚本的 Change 目录和输出路径
FOLDER_2024 = r"E:\ct\our_dataset\train\change\2024_RGB"
FOLDER_2025 = r"E:\ct\our_dataset\train\change\post"
ANSWER_JSON = r"C:\Users\conti\Desktop\k\change_regions_merged.json"
KNOWLEDGE_MD = r"C:\Users\conti\Desktop\k\land-use(1).md"
OUTPUT_JSONL = r"C:\Users\conti\Desktop\k\output\benchmark_qwen_answer_change_train.jsonl"


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


def load_answer_json(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf8") as f:
        return json.load(f)


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


def build_prompt(task_id, knowledge_text, task_metadata):
    # 完全保留第一个脚本针对 Change 的原始 Prompt
    prompt = f"""
        ### Reference Knowledge
        {knowledge_text}

        ### Input Metadata
        {json.dumps(task_metadata, ensure_ascii=False, indent=2)}

        ### Task ID
        {task_id}

        ### Task
        You are a remote sensing expert in land-use change analysis. Analyze two precisely aligned images from T1 and T2 using both:
        1. the reference knowledge (land_use.md), and
        2. the target-region information in the input JSON.

        Important:
        - The marked areas only help locate possible change regions.
        - Do not mention any visual cue symbols or markings in the output.
        - First identify all real changes.
        - Then keep only true cross-L2 changes as INTER_L2_CHANGE.
        - Exclude same-L2 changes as NO_CHANGE.

        ---

        ### Output Format
        Return valid JSON only. Do not add any extra text.

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

        ---

        ### Field Requirements

        #### step_1_global_perception
        Briefly identify the overall geographic setting and land-use context, such as rural area, urban area, peri-urban zone, industrial area, transport corridor, or ecological patch.
        Summarize the main environmental features, spatial pattern, and dominant land use in T1 and T2.

        #### step_2_instance_visual
        Using the input JSON, identify each candidate change area and describe its location and clues for L2 classification.
        For each instance, briefly analyze:
        - shape
        - texture
        - structure
        - tone/spectral appearance
        - relation to nearby features

        Then infer the most likely L2 categories and explain why.

        #### step_3_relational_model
        Analyze the relationship between change instances, nearby features, and the overall scene from four aspects:
        1. spatial position
        2. physical interaction
        3. functional relationship
        4. usage relationship

        Also explain the possible local effects on ecology, social function, or land use.

        #### step_4_reasoning
        Use a strict two-stage process:

        [Stage 1: Forward Reasoning]
        Compare T1 and T2 and identify all real changes first.
        Exclude pseudo-change caused by:
        - illumination differences
        - shadows
        - vegetation seasonality
        - imaging condition differences
        - other visual-only differences without real land-use change

        Only confirmed real changes can enter the next stage.

        [Stage 2: Backward Reasoning]
        For each real change, decide whether it should be excluded or retained:
        - If the area stays in the same L2 class and land-use function does not really change, label it NO_CHANGE.
        - If the area changes across different L2 classes and land-use attributes, spatial pattern, or function truly change, label it INTER_L2_CHANGE.

        For each real change, explain:
        1. why it is not pseudo-change;
        2. why it is NO_CHANGE or INTER_L2_CHANGE;
        3. if INTER_L2_CHANGE, what evidence supports the cross-L2 transition.

        Your logic must be:
        - identify all real changes first;
        - remove within-class changes as NO_CHANGE;
        - retain only true cross-class changes as INTER_L2_CHANGE.

        #### step_5_future_inference
        In one sentence, predict the likely future ecological, social, or land-use trend of the area.

        #### step_6_confidence
        Give a calibrated confidence score based on class difficulty, visual evidence, temporal consistency, and ambiguity.

        Base score:
        - Easy: construction, roads, rivers/lakes/ponds/reservoirs → 8-9
        - Medium: paddy field, general forest, rural construction land → 7-8
        - Hard: dryland vs irrigated land, forest subtypes, orchard, grassland subtypes → 5-6

        Adjust upward for:
        - clear object extent
        - strong surrounding context
        - consistent T1/T2 evidence
        - metadata support

        Adjust downward for:
        - shadows
        - seasonal effects
        - low resolution
        - fragmented objects
        - confusion with similar classes

        Return:
        - score: integer 0-10
        - justification: main reasons
        - limitations: key uncertainties
        - alternative_l2_candidates: possible alternative L2 classes, or []

        If confidence is low, recommend field checking or higher-resolution imagery.

        ---

        ### answer
        Write one concise, professional paragraph that naturally combines:
        - overall scene context
        - instance visual evidence
        - relationship analysis
        - change reasoning
        - future implication
        - confidence

        Requirements:
        1. do not repeat step titles;
        2. keep the writing formal and geographically grounded;
        3. clearly separate:
           - excluded within-class changes (NO_CHANGE)
           - retained cross-class changes (INTER_L2_CHANGE)
        4. end with an explicit confidence statement.

        Return valid JSON only.
"""
    return prompt.strip()


# ============================
# 3. 异步并发核心调用 (完美防超时)
# ============================
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True
)
async def call_qvq_with_retry_async(messages):
    """带重试机制的异步 API 调用，使用流式接收防网关静默超时"""
    response = await client.chat.completions.create(
        model=Config.MODEL_NAME,
        messages=messages,
        stream=True,  # 【核心】保留 stream=True
        temperature=0,
        timeout=180
    )

    full_content = ""
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content

    return full_content


async def process_single_task(prefix, img_pre_path, img_post_path, prompt_content, semaphore, write_lock, counter_info,
                              output_jsonl_abs):
    """处理单一图像对的异步任务"""
    record = {"task_id": prefix, "success": False, "model_response": None}

    async with semaphore:
        try:
            # 获取锁后读取图片，防止内存溢出
            img1_b64 = encode_image(img_pre_path)
            img2_b64 = encode_image(img_post_path)

            # 图片在前，文本在后，有利于 QvQ 解析
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}},
                        {"type": "text", "text": prompt_content}
                    ]
                }
            ]

            full_content = await call_qvq_with_retry_async(messages)
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

    print(
        f"    ⏳ 耗时统计: [已耗时: {format_time(elapsed)}] | [预计剩余: {format_time(remaining)}] | [预计总共: {format_time(est_total)}]")


# ============================
# 4. 异步主程序
# ============================
async def main_async():
    print("[*] 开始执行 QvQ-72b (Change) 数据标注任务 (智能断点续传模式)...")

    knowledge_text = load_knowledge(KNOWLEDGE_MD)
    all_metadata = load_answer_json(ANSWER_JSON)

    if isinstance(all_metadata, list):
        metadata_map = {item["task_id"]: item for item in all_metadata if "task_id" in item}
    elif isinstance(all_metadata, dict):
        metadata_map = all_metadata
    else:
        metadata_map = {}

    if not os.path.exists(FOLDER_2024):
        print(f"[-] 错误：找不到前时相文件夹 {FOLDER_2024}")
        return

    files_2024 = sorted([f for f in os.listdir(FOLDER_2024) if "_2024_RGB.png" in f])
    print(f">>> 扫描到 {len(files_2024)} 个 T1 图像文件。")

    output_dir = os.path.dirname(OUTPUT_JSONL)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    OUTPUT_JSONL_ABS = os.path.abspath(OUTPUT_JSONL)

    # ================= 【优化核心：智能断点续传与过滤】 =================
    processed_ids = set()
    existing_records = {}

    if os.path.exists(OUTPUT_JSONL_ABS):
        with open(OUTPUT_JSONL_ABS, "r", encoding="utf8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    task_id = data.get("task_id")
                    if not task_id:
                        continue

                    existing_records[task_id] = data

                    if data.get("success"):
                        # 1. 完全成功的JSON，加入跳过列表
                        processed_ids.add(task_id)
                    else:
                        # 2. 失败的情况分类处理：
                        raw_output = data.get("raw_output", "")

                        # 使用正则检测是否为计费/限流/网络断开等纯API报错
                        # "429" 限流/欠费, "403" 权限/体验额度耗尽, "Error:" 代码抛出异常
                        if isinstance(raw_output, str) and re.search(r"Error:|429|403|exceeded", raw_output,
                                                                     re.IGNORECASE):
                            # 这是 API 报错，模型没进行推理！不加入 processed_ids，让它重跑
                            pass
                        else:
                            # 模型确实跑了（消耗了Token），只是输出格式坏了（比如 "Alright..." 开头）
                            # 为了省钱，把这种也当做 "已处理完"，保留它的文本！
                            processed_ids.add(task_id)
                except:
                    continue

        print(f"[*] 发现历史记录：共需要保留 {len(processed_ids)} 个已处理任务（含格式错误但已推理的任务，为您节省Token）。")

        # 覆写文件，仅清除真正因为 API Error 失败的垃圾数据
        records_to_write = [existing_records[tid] for tid in processed_ids]
        with open(OUTPUT_JSONL_ABS, "w", encoding="utf8") as out_f:
            for record in records_to_write:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[*] 已清理输出文件中的 API 报错脏数据，准备继续推理未完成的任务。")
    # ========================================================

    task_params_list = []

    for f_name in files_2024:
        prefix = f_name.split("_2024_RGB")[0]

        # 仅跳过成功的任务
        if prefix in processed_ids:
            continue

        img_pre_path = os.path.join(FOLDER_2024, f_name)
        # 匹配原本 Change 脚本中的命名规则：f"{prefix}_extra.png"
        img_post_path = os.path.join(FOLDER_2025, f"{prefix}_extra.png")

        if not os.path.exists(img_post_path):
            print(f"    [-] 缺失后时相图像 {prefix}_extra.png，跳过该任务。")
            continue

        task_metadata = metadata_map.get(prefix, {})
        prompt_content = build_prompt(prefix, knowledge_text, task_metadata)

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
    print(f"\n[*] 🎉 QvQ-72b (Change) 异步批量数据标注脚本执行完毕！")
    print(f"[*] 🏆 本次运行共完成 {total_tasks} 个请求，实际总耗时: {format_time(total_time_cost)}")
    print(f"[*] 💾 结果已全部保存至：{OUTPUT_JSONL_ABS}")


if __name__ == "__main__":
    asyncio.run(main_async())