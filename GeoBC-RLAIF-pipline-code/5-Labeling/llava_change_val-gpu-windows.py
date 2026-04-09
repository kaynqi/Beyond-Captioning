#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import time
import argparse
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

#单卡
# $env:HF_HUB_OFFLINE = "1"
# python llava_nochange_val_optimized222-linux.py --gpus 0 --max-new-tokens 320
#双卡
# $env:HF_HUB_OFFLINE = "1"
# python llava_nochange_val_optimized222-linux.py --gpus 0,1 --max-new-tokens 320


# ============================
# 1. 路径配置
# ============================
MODEL_PATH = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

FOLDER_2024 = r"E:\srk\llava\val\change\2024_RGB"
FOLDER_2025 = r"E:\srk\llava\val\change\post"
ANSWER_JSON = r"E:\srk\llava\dataset_code\change_regions_merged-val.json"
KNOWLEDGE_MD = r"E:\srk\llava\dataset_code\land-use.md"
OUTPUT_JSONL = r"E:\srk\llava\dataset_code\llava_answer_changenew.jsonl"


# ============================
# 2. 基础工具
# ============================
def setup_torch() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def maybe_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_knowledge(md_path: str) -> str:
    if not os.path.exists(md_path):
        return "Standard Land Classification Table"
    with open(md_path, "r", encoding="utf8") as f:
        return f.read()


def load_answer_json(json_path: str):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf8") as f:
        return json.load(f)


def extract_json(text: str):
    try:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        content = match.group(1).strip() if match else None
        if not content:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            content = match.group() if match else None
        return json.loads(content) if content else None
    except Exception:
        return None


def shard_output_path(base_output: str, shard_id: int, num_shards: int) -> str:
    root, ext = os.path.splitext(base_output)
    return f"{root}.shard{shard_id}of{num_shards}{ext}"


def discover_shard_paths(base_output: str, num_shards: int) -> List[str]:
    return [shard_output_path(base_output, i, num_shards) for i in range(num_shards)]


def read_existing_records(paths: List[str]) -> Dict[str, dict]:
    """
    读取已有结果：
    - 同一 task_id 如果有 success=True，优先保留 success=True
    - 否则保留最后一次记录
    """
    result_map: Dict[str, dict] = {}

    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                task_id = data.get("task_id")
                if not task_id:
                    continue

                old = result_map.get(task_id)
                if old is None:
                    result_map[task_id] = data
                else:
                    old_success = bool(old.get("success"))
                    new_success = bool(data.get("success"))
                    if (not old_success and new_success) or (old_success == new_success):
                        result_map[task_id] = data

    return result_map


def merge_shards(base_output: str, num_shards: int) -> None:
    shard_paths = discover_shard_paths(base_output, num_shards)
    all_paths = [base_output] + shard_paths
    merged_map = read_existing_records(all_paths)

    tmp_output = base_output + ".tmp"
    with open(tmp_output, "w", encoding="utf8") as f:
        for task_id in sorted(merged_map.keys()):
            f.write(json.dumps(merged_map[task_id], ensure_ascii=False) + "\n")

    os.replace(tmp_output, base_output)
    print(f">>> Merged {len(merged_map)} unique records into: {base_output}")


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def estimate_finish_time(remaining_seconds: float) -> str:
    ts = time.time() + max(0, remaining_seconds)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ============================
# 3. prompt 构造
# ============================
def build_prompt(task_id, knowledge_text, task_metadata):
    return f"""
You are a remote sensing land-use change analyst.

Reference knowledge:
{knowledge_text}

Input metadata:
{json.dumps(task_metadata, ensure_ascii=False, separators=(",", ":"))}

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

Task ID: {task_id}

Return valid JSON only.
""".strip()


# ============================
# 4. 数据准备
# ============================
def build_task_list(folder_2024: str, folder_2025: str) -> List[Tuple[str, str, str]]:
    files_2024 = sorted([f for f in os.listdir(folder_2024) if "_2024_RGB.png" in f])
    tasks = []

    for f_name in files_2024:
        prefix = f_name.split("_2024_RGB")[0]
        img_pre_path = os.path.join(folder_2024, f_name)
        img_post_path = os.path.join(folder_2025, f"{prefix}_extra.png")
        if os.path.exists(img_post_path):
            tasks.append((prefix, img_pre_path, img_post_path))

    return tasks


def split_tasks(tasks: List[Tuple[str, str, str]], shard_id: int, num_shards: int):
    return [x for i, x in enumerate(tasks) if i % num_shards == shard_id]


# ============================
# 5. 推理
# ============================
def generate_once(
    model,
    processor,
    device: str,
    task_id: str,
    img_pre_path: str,
    img_post_path: str,
    knowledge_text: str,
    task_metadata,
    max_new_tokens: int,
) -> str:
    # Windows / Linux 都更稳：直接用 PIL 打开图片
    img_pre = Image.open(img_pre_path).convert("RGB")
    img_post = Image.open(img_post_path).convert("RGB")

    prompt = build_prompt(task_id, knowledge_text, task_metadata)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[img_pre, img_post],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            num_beams=1,
        )

    input_len = inputs["input_ids"].shape[1]
    raw_output = processor.decode(
        generated_ids[0][input_len:],
        skip_special_tokens=True,
    )

    del inputs
    del generated_ids
    img_pre.close()
    img_post.close()
    return raw_output


def run_worker(args):
    setup_torch()

    device = "cuda:0"
    torch.cuda.set_device(0)

    shard_id = args.shard_id
    num_shards = args.num_shards
    shard_out = shard_output_path(args.output_jsonl, shard_id, num_shards)

    print(f">>> Worker shard={shard_id}/{num_shards}")
    print(f">>> Visible GPU = {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f">>> Device = {device}")
    print(f">>> Loading model: {args.model_path}")
    print(f">>> use_fast={args.use_fast}, max_new_tokens={args.max_new_tokens}, attn_impl={args.attn_impl}")
    print(f">>> shard output: {shard_out}")

    existing_map = read_existing_records([args.output_jsonl, shard_out])
    processed_success = {k for k, v in existing_map.items() if v.get("success")}
    processed_all = set(existing_map.keys())

    knowledge_text = load_knowledge(args.knowledge_md)
    all_metadata = load_answer_json(args.answer_json)
    metadata_map = (
        {item["task_id"]: item for item in all_metadata}
        if isinstance(all_metadata, list)
        else all_metadata
    )

    all_tasks = build_task_list(args.folder_2024, args.folder_2025)
    shard_tasks = split_tasks(all_tasks, shard_id, num_shards)

    if args.skip_failed:
        todo_tasks = [x for x in shard_tasks if x[0] not in processed_all]
    else:
        todo_tasks = [x for x in shard_tasks if x[0] not in processed_success]

    print(
        f">>> total tasks={len(all_tasks)}, shard tasks={len(shard_tasks)}, "
        f"todo={len(todo_tasks)}, skipped={len(shard_tasks) - len(todo_tasks)}"
    )

    if len(todo_tasks) == 0:
        print(">>> Nothing to do for this shard.")
        return

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=args.use_fast,
    )

    model_kwargs = dict(
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs,
    ).eval()

    maybe_empty_cache()

    start_ts = time.time()
    done = 0

    with open(shard_out, "a", encoding="utf8") as out_f:
        pbar = tqdm(todo_tasks, desc=f"LLaVA shard {shard_id}/{num_shards}")
        for prefix, img_pre_path, img_post_path in pbar:
            item_start = time.time()

            try:
                try:
                    raw_output = generate_once(
                        model=model,
                        processor=processor,
                        device=device,
                        task_id=prefix,
                        img_pre_path=img_pre_path,
                        img_post_path=img_post_path,
                        knowledge_text=knowledge_text,
                        task_metadata=metadata_map.get(prefix, {}),
                        max_new_tokens=args.max_new_tokens,
                    )
                except torch.OutOfMemoryError:
                    maybe_empty_cache()
                    fallback_tokens = max(args.oom_retry_tokens, args.max_new_tokens // 2)
                    print(f"\n[OOM-Retry] Task {prefix}: retry with max_new_tokens={fallback_tokens}")
                    raw_output = generate_once(
                        model=model,
                        processor=processor,
                        device=device,
                        task_id=prefix,
                        img_pre_path=img_pre_path,
                        img_post_path=img_post_path,
                        knowledge_text=knowledge_text,
                        task_metadata=metadata_map.get(prefix, {}),
                        max_new_tokens=fallback_tokens,
                    )

                parsed_json = extract_json(raw_output)

                if parsed_json is None:
                    print(f"\n[DEBUG] Task {prefix} raw_output:\n{raw_output[:800]}\n")

                record = {
                    "task_id": prefix,
                    "success": parsed_json is not None,
                    "model_response": parsed_json,
                    "raw_output": raw_output.strip(),
                    "shard_id": shard_id,
                    "num_shards": num_shards,
                    "elapsed_sec": round(time.time() - item_start, 3),
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

            except Exception as e:
                maybe_empty_cache()
                print(f"\n[ERROR] Task {prefix}: {e}")
                record = {
                    "task_id": prefix,
                    "success": False,
                    "model_response": None,
                    "raw_output": f"Error: {str(e)}",
                    "shard_id": shard_id,
                    "num_shards": num_shards,
                    "elapsed_sec": round(time.time() - item_start, 3),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

            done += 1
            if done % args.empty_cache_every == 0:
                maybe_empty_cache()

            avg_sec = (time.time() - start_ts) / max(done, 1)
            remain = len(todo_tasks) - done
            eta_sec = remain * avg_sec

            pbar.set_postfix(
                avg=f"{avg_sec:.2f}s",
                eta=format_seconds(eta_sec),
                finish=estimate_finish_time(eta_sec),
            )

    maybe_empty_cache()
    print(f">>> Shard {shard_id}/{num_shards} done: {shard_out}")


# ============================
# 6. 多卡调度
# ============================
def launch_multi_gpu(args):
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if len(gpus) == 0:
        raise ValueError("No GPUs provided.")

    print(f">>> Launching workers on GPUs: {gpus}")

    procs = []
    for shard_id, gpu_id in enumerate(gpus):
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--shard-id", str(shard_id),
            "--num-shards", str(len(gpus)),
            "--model-path", args.model_path,
            "--knowledge-md", args.knowledge_md,
            "--answer-json", args.answer_json,
            "--output-jsonl", args.output_jsonl,
            "--folder-2024", args.folder_2024,
            "--folder-2025", args.folder_2025,
            "--max-new-tokens", str(args.max_new_tokens),
            "--oom-retry-tokens", str(args.oom_retry_tokens),
            "--empty-cache-every", str(args.empty_cache_every),
        ]

        if args.use_fast:
            cmd.append("--use-fast")
        if args.skip_failed:
            cmd.append("--skip-failed")
        if args.attn_impl:
            cmd.extend(["--attn-impl", args.attn_impl])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("HF_HUB_OFFLINE", "1")

        print(">>> Spawn:", " ".join(cmd), f"(CUDA_VISIBLE_DEVICES={gpu_id})")
        p = subprocess.Popen(cmd, env=env)
        procs.append((gpu_id, p))

        if args.stagger_seconds > 0 and shard_id < len(gpus) - 1:
            print(f">>> Sleep {args.stagger_seconds}s before launching next worker ...")
            time.sleep(args.stagger_seconds)

    failed = False
    for gpu_id, p in procs:
        ret = p.wait()
        if ret != 0:
            failed = True
            print(f"[ERROR] Worker on GPU {gpu_id} exited with code {ret}")

    merge_shards(args.output_jsonl, len(gpus))

    if failed:
        raise SystemExit(1)

    print(">>> All workers completed successfully.")


# ============================
# 7. CLI
# ============================
def parse_args():
    parser = argparse.ArgumentParser(description="Windows/Linux compatible resumable LLaVA batch runner")

    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--knowledge-md", default=KNOWLEDGE_MD)
    parser.add_argument("--answer-json", default=ANSWER_JSON)
    parser.add_argument("--output-jsonl", default=OUTPUT_JSONL)
    parser.add_argument("--folder-2024", default=FOLDER_2024)
    parser.add_argument("--folder-2025", default=FOLDER_2025)

    parser.add_argument("--gpus", default="0,1", help="Example: 0,1")
    parser.add_argument("--use-fast", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")

    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--oom-retry-tokens", type=int, default=192)
    parser.add_argument("--empty-cache-every", type=int, default=20)
    parser.add_argument("--stagger-seconds", type=int, default=20)
    parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])

    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.merge_only:
        num_shards = len([x for x in args.gpus.split(",") if x.strip()])
        merge_shards(args.output_jsonl, num_shards)
        return

    if args.worker:
        run_worker(args)
        return

    launch_multi_gpu(args)


if __name__ == "__main__":
    main()