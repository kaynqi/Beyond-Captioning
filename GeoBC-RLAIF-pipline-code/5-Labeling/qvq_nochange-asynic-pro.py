import os
import json
import base64
import re
import time
import asyncio
import httpx
from tqdm import tqdm
from openai import AsyncOpenAI  # 使用异步客户端
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================
# 1. API 客户端配置
# ============================
client = AsyncOpenAI(
    api_key= os.getenv("OPENROUTER_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # 明确配置异步 HTTP 客户端，禁用代理防拦截，并设置长超时
    http_client=httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(proxy=None),
        timeout=180.0
    )
)


class Config:
    # 针对代码中的配置使用该模型
    MODEL_NAME = "qvq-72b-preview"
    # 【并发控制】QvQ 系列作为高负载的推理模型，建议并发维持在 3，避免触发阿里限流 (429)
    MAX_CONCURRENT_REQUESTS = 3


FOLDER_2024 = r"G:\ct\our_dataset\train\nochange\2024_RGB"
FOLDER_2025 = r"G:\ct\our_dataset\train\nochange\post"
ANSWER_JSON = r"G:\ct\our_dataset\train\change\change_regions.json"
KNOWLEDGE_MD = r"G:\ct\our_dataset\test\land-use.md"
OUTPUT_JSONL = r"output\benchmark_qvq_answer_nochange_train.jsonl"


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
    # 保持您原有的 Prompt 文本不变
    prompt = f"""
### Reference Knowledge
{knowledge_text}

### Task Description
You are a senior remote sensing analysis expert specializing in land-use dynamics and social geography. You are required to conduct in-depth interpretation and evolutionary reasoning on two co-registered bi-temporal remote sensing images.

The known conditions are as follows:
1. The input contains remote sensing imagery from two time phases: T1 and T2.
2. You need to prove that: the region contains at most two types of situations:
   - Pseudo-changes: Apparent differences caused by factors such as lighting, shadows, vegetation phenology, and imaging condition differences.
   - Within-class changes (NO_CHANGE): Instances where the region remains within the same secondary class (L2), and there is no evidence of a functional land-use change.
3. Your task is not to search for INTER_L2_CHANGE, but rather to provide a systematic analysis explaining why this region does not constitute INTER_L2_CHANGE.

---

### Output Requirements
You must strictly output the following JSON structure. Do not add any explanation outside the JSON, do not omit any fields, and do not output invalid JSON:

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
Begin by analyzing the overall environment, determining the macro-geographical unit and land-use background of the study area, such as rural, urban, peri-urban transition zones, industrial zones, transportation corridors, ecological patches, etc.
Summarize the overall environmental features, spatial organization patterns, and major land-use patterns in T1 and T2, explaining whether the region has maintained high land-use continuity and functional stability overall. This will provide background support for the judgment of "no target change."

#### step_2_instance_visual
Based on the main cue information in the input JSON, identify each instance in the region that appears to have changed or could easily be misjudged as a change, and specify its precise location and initial secondary classification clues.
For each instance, analyze its visual characteristics in detail, including but not limited to:
- Geometric shape
- Texture characteristics
- Structural characteristics
- Spectral/tonal expression
- Configuration relationship with surrounding features

Clearly explain whether these differences are:
- Pseudo-changes;
- Or apparent adjustments, status changes, or structural refinements within the same secondary class.

Based on this visual evidence, link each instance to potential secondary land-use categories and explain why it remains within the same class or function.

#### step_3_relational_model
Analyze the relationships among instances from the following four dimensions:
1. Spatial positional relationships
2. Physical interaction relationships
3. Functional relationships
4. Basic usage relationships

Analyze whether these relationships remain consistent between T1 and T2, and whether they indicate that the land-use system in the region maintains the same functional logic, social use, or ecological role.
Further explain why local apparent differences are insufficient to support the judgment of "cross-class change at the secondary class level."

#### step_4_reasoning
Strictly follow a two-stage process of "positive screening + negative exclusion."
Do not skip steps or directly conclude "no change."

【Stage 1: Positive Screening】
Comprehensively compare T1 and T2, first identifying all instances in the region that appear to have changed, could be misjudged as changes, or exhibit local differences.
Note: In this stage, you must fully check all potential sources of differences, and cannot skip the analysis of local differences just because it is known that "no target change" is present.

At this stage, prioritize identifying and explaining the following sources of pseudo-changes:
- Illumination differences
- Shadow variations
- Vegetation phenology changes
- Imaging condition differences
- Seasonal humidity, bare soil exposure, local coverage status changes
- Other factors causing apparent differences but not leading to real land-use functional changes

If certain areas show real change traces, further determine whether these changes are simply state changes, density changes, minor adjustments in form, or structural updates within the same secondary class, rather than cross-class changes.

【Stage 2: Negative Exclusion】
Based on the input JSON cues and reference knowledge, analyze the evolutionary logic of each difference instance identified in Stage 1 and exclude the possibility of INTER_L2_CHANGE.

The judgment rules are as follows:
- If differences are caused only by imaging, seasonal, shadow, phenology, or similar factors, classify them as pseudo-changes, not real land-use changes.
- If differences remain within the same secondary class (L2), and there is no evidence that the land-use function has changed, classify them as NO_CHANGE.
- Only if the differences are between different secondary classes and there is sufficient evidence of a cross-class transition in land-use attributes, spatial organization, or functional use, can they be classified as INTER_L2_CHANGE.

However, in this task, your goal is to provide rigorous analysis and prove that no INTER_L2_CHANGE exists in this region.

For each difference instance, clearly explain:
1. Why it is a pseudo-change or why it belongs to within-class change (NO_CHANGE);
2. Why these differences are insufficient to support a secondary class cross-class transition;
3. Why it cannot be classified as INTER_L2_CHANGE.

The final analysis must follow this logical loop:
- First, fully identify all potential differences;
- Then, distinguish between pseudo-changes and within-class changes (NO_CHANGE);
- Finally, prove that there are no required cross-class changes (INTER_L2_CHANGE).

#### step_5_future_inference
Based on the relationship modeling and stability analysis, use one sentence to predict the possible future ecological function, social use, or land-use continuation direction of the region.
The language should be concise but must reflect geographic and socio-spatial evolutionary logic.

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
When the confidence is low, you should explicitly recommend field verification or confirmation with higher-resolution data.

---
Specifically:
- score: An integer between 0 and 10 indicating the overall confidence in the "no target change" judgment;
- justification: Explain the main basis for this confidence score, such as strong continuity of regional functions, consistency of key visual evidence, stable relationship structures, and no evidence of cross-class evolution;
- limitations: Clearly state any limitations of this analysis, such as insufficient spatial resolution, occlusion effects, seasonal interference, or ambiguous category boundaries;
- alternative_l2_candidates: If there is slight uncertainty for some instances, list their possible alternative secondary categories; if no obvious alternatives, return an empty array [].

If confidence is low, explicitly recommend combining higher-resolution imagery, more temporal data, or field verification for further confirmation.

---

### Answer Field Requirements
Integrate the above layered reasoning into a fluent, professional, rigorous, and geographically grounded interpretation paragraph.

Requirements:
1. Do not mechanically repeat step titles;
2. The narrative must naturally integrate the overall environment, instance visual features, relationship modeling, difference screening, the reasoning for "no target change," future impacts, and confidence judgment;
3. The writing style should resemble a professional geography report or remote sensing interpretation report;
4. Use standard, formal, academic transitions and expressions;
5. The conclusion must explicitly include a confidence statement, for example:
   "Overall, it is determined with X confidence that the region has not undergone any secondary class cross-class changes..."
6. The answer must clearly state:
   - Which differences are pseudo-changes;
   - Which differences belong to within-class changes (NO_CHANGE);
   - Why no INTER_L2_CHANGE exists.

---

### Additional Execution Constraints
1. Do not skip identifying and analyzing local differences just because the region is known to have "no target change."
2. Do not directly equate "minor visual differences" with "no change," a clear evidence chain is required.
3. Do not equate "apparent changes" with "land-use changes."
4. Do not misclassify within-class changes as INTER_L2_CHANGE.
5. Do not mention any visual cue markers in the answer.
6. If real change traces exist in the region, prioritize determining whether they are NO_CHANGE rather than exaggerating them as cross-class changes.
7. The final conclusion must clearly state that the region does not contain any INTER_L2_CHANGE that needs to be retained.
8. The output must be strictly valid JSON.
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
        temperature=0,  # 原本配置中的 temperature=0
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

            # 注意内容列表中：图片在前，文本在后，有利于 QvQ 解析
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
    print("[*] 开始执行 QvQ-72b (No-Change) 数据标注任务 (异步并发断点续传模式)...")

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
            # 缺失对应的 T2 图像，直接跳过并提示
            print(f"    [-] 缺失后时相图像 {prefix}，跳过该任务。")
            continue

        img_post_path = os.path.join(FOLDER_2025, matched_post_name)

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
    print(f"\n[*] 🎉 QvQ-72b (No-Change) 异步批量数据标注脚本执行完毕！")
    print(f"[*] 🏆 本次运行共完成 {total_tasks} 个请求，实际总耗时: {format_time(total_time_cost)}")
    print(f"[*] 💾 结果已全部保存至：{OUTPUT_JSONL_ABS}")


if __name__ == "__main__":
    asyncio.run(main_async())