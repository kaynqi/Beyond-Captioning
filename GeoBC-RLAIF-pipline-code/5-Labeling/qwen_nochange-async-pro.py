import os
import json
import base64
import re
import time
import asyncio
import httpx
from tqdm import tqdm
from openai import AsyncOpenAI  # 【修改1】使用异步客户端
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================
# 1. API 客户端配置
# ============================
# 【修改2】初始化 AsyncOpenAI 客户端
client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # 明确配置异步 HTTP 客户端，禁用代理，加入长超时
    http_client=httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(proxy=None),
        timeout=180.0
    )
)


class Config:
    MODEL_NAME = "qvq-max"
    # 【并发控制】QvQ-Max 是极高负载的慢思考模型，阿里对其并发限制通常非常严苛。
    # 强烈建议将并发数设置在 2~4 之间，过高必定疯狂报 429 错误。
    MAX_CONCURRENT_REQUESTS = 3


FOLDER_2024 = r"G:\ct\our_dataset\train\change\2024_RGB"
FOLDER_2025 = r"G:\ct\our_dataset\train\change\post"
ANSWER_JSON = r"G:\ct\our_dataset\train\change_regions.json"
KNOWLEDGE_MD = r"G:\ct\our_dataset\test\land-use.md"
OUTPUT_JSONL = r"output\benchmark_qvq_answer_change_train.jsonl"


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
        # 匹配 markdown json 块
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            # 或者找最外层大括号
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
    prompt = f"""
### Reference Knowledge
{knowledge_text}

### Input Metadata (Target regions for analysis)
{json.dumps(task_metadata, ensure_ascii=False, indent=2)}

### Task ID
{task_id}

### Task Description
You are a senior remote sensing analysis expert specializing in land-use dynamics and social geography. You are required to conduct in-depth interpretation and evolutionary reasoning on two precisely co-registered bi-temporal remote sensing images.

The known conditions are as follows:
1. The input contains remote sensing imagery from two time phases: T1 and T2.
2. The red-marked areas are only visual cues to help locate potential change regions.
3. In the output description, you must not mention "boundaries," "red boxes," "red markings," or any visual cue symbols themselves.
4. You must conduct an integrated analysis by combining the Reference Knowledge(land_use.md) and the target-region information provided in the input JSON.
5. Your task is not merely to determine whether change exists, but to:
   - first identify all real changes that actually occurred;
   - then filter out the instances that truly belong to cross-category change at the secondary class level (INTER_L2_CHANGE);
   - and exclude within-class changes that do not belong to the target change category (NO_CHANGE).

---

### Output Requirements
You must strictly output according to the following JSON structure. Do not add any explanatory text outside the JSON, do not omit any fields, and do not output invalid JSON:

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

### Writing Requirements for Each Field

#### step_1_global_perception
Start from the overall environment and determine the macro geographic unit and land-use background of the study area, such as rural area, urban area, peri-urban transition zone, industrial zone, transportation corridor, ecological patch, etc.
Summarize the overall environmental characteristics, spatial organization patterns, and major land-use patterns in T1 and T2 to provide contextual constraints for subsequent instance-level interpretation.

#### step_2_instance_visual
Refer to the main change indicators in the input JSON, identify candidate change instances one by one, and specify their precise locations and initial clues for secondary-category classification.
For each instance, provide a detailed analysis of its visual characteristics, including but not limited to:
- geometric shape
- texture characteristics
- structural characteristics
- spectral / tonal appearance
- configurational relationship with surrounding features

Based on this visual evidence, connect each instance to possible secondary land-use categories and explain the basis for the judgment.

#### step_3_relational_model
For multiple changing entities in complex scenes, perform relationship modeling and relationship completion.
You must analyze the mutual relationships of instances with themselves, their surrounding environment, and the overall environment from the following four dimensions:
1. spatial positional relationships
2. physical interaction relationships
3. functional relationships
4. basic usage relationships

Combine the specific content within the imagery to further elaborate on the specific impacts that local changes may exert on regional ecological processes, social functional organization, or land use.
#### step_4_reasoning
You must complete the change analysis strictly through a two-stage process of "forward reasoning chain + backward reasoning chain." Do not skip steps, and do not merge "real change identification" and "target change filtering" into a single step.

[Stage 1: Forward Reasoning Chain]
Conduct a comprehensive comparison between T1 and T2, and first identify all instances and locations where real change actually occurred.
Note: Regardless of whether these changes ultimately belong to the target changes to be retained, they must all be identified first, forming the "complete set of real changes."

At this stage, you must be highly sensitive to and exclude the following sources of pseudo-change:
- illumination differences
- shadow variation
- vegetation phenology change
- imaging condition differences
- other factors that cause only apparent visual differences but do not lead to real land-use change

Only after excluding the above interferences can an instance be included as a "real change."

[Stage 2: Backward Reasoning Chain]
Based on the cues in the input JSON and in combination with the reference knowledge, analyze the evolutionary logic of each real change instance and filter the "complete set of real changes" one by one.

Filtering rules are as follows:
- If the area remains within the same secondary class (L2) before and after the change, and there is no evidence that the land-use function has changed, classify it as NO_CHANGE.
  Such changes are "within-class changes." Although apparent changes may exist, they are not target changes to be retained in this task.
- If the area belongs to different secondary classes (L2) before and after the change, and there is sufficient evidence that the land-use attributes, spatial organization pattern, or functional use have undergone a real cross-class transformation, classify it as INTER_L2_CHANGE.

For each change instance, you must explicitly explain:
1. why it is not a pseudo-change;
2. why it should be excluded as NO_CHANGE, or why it should be retained as INTER_L2_CHANGE;
3. if it is judged as INTER_L2_CHANGE, what specific evidence and evolutionary logic support the cross-class transition.

The final reasoning must demonstrate the following closed logical loop:
- first identify all real changes through the forward reasoning chain;
- then filter out unnecessary within-class changes (NO_CHANGE) through the backward reasoning chain;
- finally retain only the required cross-class changes (INTER_L2_CHANGE).

#### step_5_future_inference
Based on the previous relational modeling and change reasoning, use one sentence to predict the future ecological impact, social impact, or land-use evolution direction of the area.
The language should be concise, but it must reflect geographic and socio-spatial evolutionary logic.

#### step_6_confidence
Provide a calibrated confidence assessment based on class difficulty, visual evidence, temporal consistency, and ambiguity.

Base score by class difficulty:
- Easy classes (construction, roads, rivers/lakes/ponds/reservoirs): 8-9
- Medium classes (paddy field, general forest land, rural construction land): 7-8
- Hard classes (dryland vs irrigated land, forest subtypes, orchard, grassland subtypes): 5-6

Adjust the score by:
- adding points for clear boundaries, strong surrounding context, consistent T1-T2 evidence, and metadata support;
- subtracting points for shadows, seasonal differences, low resolution, fragmented objects, or confusion with similar classes.


Return:
- score: integer from 0 to 10
- justification: main reasons for the score
- limitations: main uncertainty sources
- alternative_l2_candidates: possible alternative secondary classes, or [] if none
When the confidence is low (less than ), you should explicitly recommend field verification or confirmation with higher-resolution data.

---

### Requirements for the answer Field
Integrate the above layered reasoning into a fluent, professional, rigorous, and geographically grounded interpretation paragraph.

Requirements:
1. Do not mechanically repeat the step titles;
2. The narrative must naturally integrate the overall environment, instance-level visual features, relationship modeling, change reasoning, future implications, and confidence judgment;
3. The writing style should resemble a professional geography report or remote sensing interpretation report;
4. Use standard, formal, and academic transitions and connective expressions;
5. The ending must explicitly include a confidence statement, for example:
   "Overall, this area is determined with X confidence to have undergone …"
6. In the answer, you must clearly distinguish:
   - the within-class changes that are excluded (NO_CHANGE)
   - the cross-class changes that are retained (INTER_L2_CHANGE)

Please strictly return valid JSON only.
"""
    return prompt.strip()


# ============================
# 3. 异步并发核心调用 (完美解决阿里网关超时问题)
# ============================
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True
)
async def call_qvq_with_retry_async(messages):
    """带重试机制的异步 API 调用，使用流式接收防止网关因为等待长思考而静默断开"""
    response = await client.chat.completions.create(
        model=Config.MODEL_NAME,
        messages=messages,
        stream=True,  # 【核心】务必保留 stream=True
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
            # 获取锁后读取图片
            img1_b64 = encode_image(img_pre_path)
            img2_b64 = encode_image(img_post_path)

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

    # 打印失败错误详情以供排查
    if not record["success"]:
        error_detail = record.get("raw_output", "未知错误")
        print(f"    ⚠️ 错误详情: {error_detail[:200]}...")

    print(
        f"    ⏳ 耗时统计: [已耗时: {format_time(elapsed)}] | [预计剩余: {format_time(remaining)}] | [预计总共: {format_time(est_total)}]")


# ============================
# 4. 异步主程序
# ============================
async def main_async():
    print("[*] 开始执行 QVQ-Max (Change) 数据标注任务 (异步并发断点续传模式)...")

    knowledge_text = load_knowledge(KNOWLEDGE_MD)
    all_metadata = load_answer_json(ANSWER_JSON)

    # 解析元数据
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

        # 仅跳过已成功的任务
        if prefix in processed_ids:
            continue

        img_pre_path = os.path.join(FOLDER_2024, f_name)
        img_post_path = os.path.join(FOLDER_2025, f"{prefix}_extra.png")

        if not os.path.exists(img_post_path):
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
    print(f"\n[*] 🎉 QVQ-Max (Change) 异步批量数据标注脚本执行完毕！")
    print(f"[*] 🏆 本次运行共完成 {total_tasks} 个请求，实际总耗时: {format_time(total_time_cost)}")
    print(f"[*] 💾 结果已全部保存至：{OUTPUT_JSONL_ABS}")


if __name__ == "__main__":
    asyncio.run(main_async())