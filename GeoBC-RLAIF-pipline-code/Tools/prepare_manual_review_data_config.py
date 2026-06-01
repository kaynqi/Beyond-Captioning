# -*- coding: utf-8 -*-
"""
prepare_manual_review_data_config.py

用途：
1. 从 2024 前时相图片目录随机抽取 10% 样本
2. 根据图片文件名前 6 位 task_id，去 5 个模型结果文件里找 answer
3. 去最终 best 文件里找 selection_info.source_model
4. 导出盲审数据和对照表，供后续人工选择页面使用
5. 将抽样得到的前后时相图片分别复制到：
   - check/check-test-change/pre_sampled_images
   - check/check-test-change/post_sampled_images

运行方式：
python prepare_manual_review_data_config.py
"""

import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# =========================
# 在这里填写你的路径和参数
# =========================
CONFIG = {
    # 前后时相图片目录
    "pre_image_dir": r"G:\ct\our_dataset\train\nochange\2024_RGB",
    "post_image_dir": r"G:\ct\our_dataset\train\nochange\2025_RGB",

    # 5 个模型结果文件
    # val change
    # "model_files": {
    #     "gemini_answer_nochange": r"output/test/benchmark_gemini_answer_nochange.jsonl",
    #     "gpt41_answer_nochange": r"output/test/benchmark_gpt41_answer_nochange.jsonl",
    #     "qvq_answer_nochange": r"output/test/benchmark_llava_answer_nochange.jsonl",
    #     "qwen_answer_nochange": r"output/test/benchmark_qvq_answer_nochange.jsonl",
    #     "llava_answer_nochange": r"output/test/benchmark_qwen_answer_nochange.jsonl",
    # },
    # nochange train
    "model_files": {
        "benchmark_gemini_answer_nochange-train": r"output/train/benchmark_gemini_answer_nochange-train.jsonl",
        "benchmark_gpt41_answer_nochange-train": r"output/train/benchmark_gpt41_answer_nochange-train.jsonl",
        "benchmark_qvq_answer_nochange_train": r"output/train/benchmark_qvq_answer_nochange_train.jsonl",
        "benchmark_qwen_answer_nochange_train": r"output/train/benchmark_qwen_answer_nochange_train.jsonl",
        "benchmark_llava_answer_nochange_train": r"output/train/benchmark_llava_answer_nochange_train.jsonl",
    },
    # # change-train
    # "model_files": {
    #     "benchmark_gemini_answer_change-train": r"output/train/benchmark_gemini_answer_change-train.jsonl",
    #     "benchmark_gpt41_answer_change-train": r"output/train/benchmark_gpt41_answer_change-train.jsonl",
    #     "benchmark_qvq_answer_change_train": r"output/train/benchmark_llava_answer_change_train.jsonl",
    #     "benchmark_qwen_answer_change_train": r"output/train/benchmark_qvq_answer_change_train.jsonl",
    #     "benchmark_llava_answer_change_train": r"output/train/benchmark_qwen_answer_change_train.jsonl",
    # },
    # 最终 best 文件
    "auto_best_file": r"best-answer/final_gt-nochange-train.json",

    # 输出目录
    "output_dir": r"check/check-nochange-train",

    # 抽样比例
    "sample_ratio": 0.1,

    # 随机种子
    "seed": 42,

    # 是否复制抽样图片
    "copy_images": True,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def first6_from_filename(filename: str) -> str:
    return Path(filename).stem[:6]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_records(file_path: Path) -> Iterable[Dict[str, Any]]:
    """
    支持：
    - .jsonl: 每行一个 json object
    - .json : 顶层是 list[dict] 或 dict
    """
    suffix = file_path.suffix.lower()

    if suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"{file_path} 第 {line_no} 行 JSON 解析失败: {e}") from e
                if isinstance(obj, dict):
                    yield obj
        return

    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return

        if isinstance(data, dict):
            if "task_id" in data:
                yield data
            else:
                for _, v in data.items():
                    if isinstance(v, dict):
                        yield v
            return

    raise ValueError(f"不支持的文件格式: {file_path}")


def extract_task_id(record: Dict[str, Any]) -> str:
    return normalize_text(record.get("task_id"))[:6]


def extract_model_answer(record: Dict[str, Any]) -> str:
    model_response = record.get("model_response")
    if not isinstance(model_response, dict):
        return ""
    return normalize_text(model_response.get("answer"))


def extract_auto_best_model(record: Dict[str, Any]) -> str:
    selection_info = record.get("selection_info")
    if not isinstance(selection_info, dict):
        return ""
    return normalize_text(selection_info.get("source_model"))


def extract_auto_best_score(record: Dict[str, Any]) -> Any:
    selection_info = record.get("selection_info")
    if not isinstance(selection_info, dict):
        return ""
    return selection_info.get("score", "")


def build_model_index(file_path: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for record in iter_records(file_path):
        task_id = extract_task_id(record)
        if not task_id:
            continue
        result[task_id] = {
            "task_id": task_id,
            "success": bool(record.get("success")),
            "answer": extract_model_answer(record),
        }
    return result


def build_auto_best_index(file_path: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for record in iter_records(file_path):
        task_id = extract_task_id(record)
        if not task_id:
            continue
        result[task_id] = {
            "task_id": task_id,
            "success": bool(record.get("success")),
            "auto_best_model": extract_auto_best_model(record),
            "auto_best_score": extract_auto_best_score(record),
        }
    return result


def list_images(image_dir: Path) -> List[Path]:
    files = []
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            files.append(p)
    return files


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    """
    按文件名前 6 位建立索引。
    若同一个 task_id 对应多个文件，保留字典序最小的那个。
    """
    index: Dict[str, Path] = {}
    for p in sorted(image_dir.iterdir(), key=lambda x: x.name):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            task_id = first6_from_filename(p.name)
            if task_id and task_id not in index:
                index[task_id] = p
    return index


def random_sample(items: List[Path], ratio: float, seed: int) -> List[Path]:
    if not items:
        return []
    n = max(1, math.ceil(len(items) * ratio))
    rng = random.Random(seed)
    sampled = rng.sample(items, min(n, len(items)))
    return sorted(sampled, key=lambda x: x.name)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_sampled_pairs(
    sampled_pre_images: List[Path],
    post_index: Dict[str, Path],
    pre_output_dir: Path,
    post_output_dir: Path,
) -> List[str]:
    """
    复制抽样得到的前后时相图片子集。
    返回复制过程中的 warnings。
    """
    ensure_dir(pre_output_dir)
    ensure_dir(post_output_dir)

    warnings: List[str] = []

    for pre_img in sampled_pre_images:
        task_id = first6_from_filename(pre_img.name)
        shutil.copy2(pre_img, pre_output_dir / pre_img.name)

        post_img = post_index.get(task_id)
        if post_img is None:
            warnings.append(f"{task_id}: 未找到对应的后时相图片，未复制到 post_sampled_images")
            continue

        shutil.copy2(post_img, post_output_dir / post_img.name)

    return warnings


def validate_config(cfg: Dict[str, Any]) -> None:
    required_keys = [
        "pre_image_dir",
        "post_image_dir",
        "model_files",
        "auto_best_file",
        "output_dir",
        "sample_ratio",
        "seed",
        "copy_images",
    ]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"CONFIG 缺少字段: {key}")

    pre_image_dir = Path(cfg["pre_image_dir"])
    post_image_dir = Path(cfg["post_image_dir"])
    if not pre_image_dir.exists():
        raise FileNotFoundError(f"前时相图片目录不存在: {pre_image_dir}")
    if not post_image_dir.exists():
        raise FileNotFoundError(f"后时相图片目录不存在: {post_image_dir}")

    auto_best_file = Path(cfg["auto_best_file"])
    if not auto_best_file.exists():
        raise FileNotFoundError(f"最终 best 文件不存在: {auto_best_file}")

    model_files = cfg["model_files"]
    if not isinstance(model_files, dict) or not model_files:
        raise ValueError("CONFIG['model_files'] 必须是非空字典")

    for alias, path_str in model_files.items():
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"模型文件不存在: {alias} -> {p}")

    ratio = cfg["sample_ratio"]
    if not (0 < ratio <= 1):
        raise ValueError("sample_ratio 必须在 (0, 1] 范围内")


def main():
    validate_config(CONFIG)

    pre_image_dir = Path(CONFIG["pre_image_dir"])
    post_image_dir = Path(CONFIG["post_image_dir"])
    auto_best_file = Path(CONFIG["auto_best_file"])
    output_dir = Path(CONFIG["output_dir"])
    ensure_dir(output_dir)

    model_files = {k: Path(v) for k, v in CONFIG["model_files"].items()}
    sample_ratio = float(CONFIG["sample_ratio"])
    seed = int(CONFIG["seed"])
    copy_images = bool(CONFIG["copy_images"])

    # 1) 前时相图片抽样
    all_pre_images = list_images(pre_image_dir)
    sampled_pre_images = random_sample(all_pre_images, ratio=sample_ratio, seed=seed)

    # 后时相建立索引
    post_image_index = build_image_index(post_image_dir)

    # 2) 建索引
    print("正在读取 5 个模型结果文件...")
    model_indexes = {}
    for alias, file_path in model_files.items():
        print(f"  - {alias}: {file_path}")
        model_indexes[alias] = build_model_index(file_path)

    print("正在读取最终 best 文件...")
    auto_best_index = build_auto_best_index(auto_best_file)

    # 3) 汇总
    blind_rows: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    print("正在生成抽样汇总...")
    for pre_img_path in sampled_pre_images:
        task_id = first6_from_filename(pre_img_path.name)
        post_img_path = post_image_index.get(task_id)

        model_answers: Dict[str, str] = {}
        missing_models: List[str] = []

        for alias, index in model_indexes.items():
            row = index.get(task_id)
            if not row:
                missing_models.append(alias)
                warnings.append(f"{task_id}: 未找到模型 {alias} 的记录")
                continue

            answer = normalize_text(row.get("answer"))
            if not answer:
                missing_models.append(alias)
                warnings.append(f"{task_id}: 模型 {alias} 的 answer 为空")
                continue

            model_answers[alias] = answer

        auto_row = auto_best_index.get(task_id, {})
        auto_best_model = normalize_text(auto_row.get("auto_best_model"))
        auto_best_score = auto_row.get("auto_best_score", "")

        if not auto_best_model:
            warnings.append(f"{task_id}: 未找到自动 best source_model")

        if post_img_path is None:
            warnings.append(f"{task_id}: 未找到对应的后时相图片")

        # 盲审选项打乱
        option_items = [{"model_name": k, "answer": v} for k, v in model_answers.items()]
        rng = random.Random(f"{seed}-{task_id}")
        rng.shuffle(option_items)

        slots = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        blind_options = []
        option_mapping = {}
        for i, item in enumerate(option_items):
            slot = slots[i]
            blind_options.append({
                "slot": slot,
                "answer": item["answer"],
            })
            option_mapping[slot] = item["model_name"]

        blind_rows.append({
            "task_id": task_id,
            "image_filename": pre_img_path.name,
            "pre_image_filename": pre_img_path.name,
            "pre_image_path": str(pre_img_path.resolve()),
            "post_image_filename": post_img_path.name if post_img_path else "",
            "post_image_path": str(post_img_path.resolve()) if post_img_path else "",
            "missing_models": missing_models,
            "blind_options": blind_options,
        })

        table_row = {
            "task_id": task_id,
            "pre_image_filename": pre_img_path.name,
            "pre_image_path": str(pre_img_path.resolve()),
            "post_image_filename": post_img_path.name if post_img_path else "",
            "post_image_path": str(post_img_path.resolve()) if post_img_path else "",
            "missing_models": "|".join(missing_models),
        }
        for alias in model_files.keys():
            table_row[f"{alias}_answer"] = model_answers.get(alias, "")
        table_rows.append(table_row)

        key_row = {
            "task_id": task_id,
            "image_filename": pre_img_path.name,
            "pre_image_filename": pre_img_path.name,
            "post_image_filename": post_img_path.name if post_img_path else "",
            "auto_best_model": auto_best_model,
            "auto_best_score": auto_best_score,
            "missing_models": "|".join(missing_models),
        }
        for slot, model_name in option_mapping.items():
            key_row[f"option_{slot}"] = model_name
        key_rows.append(key_row)

    # 4) 导出
    blind_jsonl_path = output_dir / "blind_review.jsonl"
    review_table_csv_path = output_dir / "sampled_review_table.csv"
    answer_key_csv_path = output_dir / "answer_key.csv"
    meta_json_path = output_dir / "prepare_meta.json"

    write_jsonl(blind_jsonl_path, blind_rows)

    table_fields = [
        "task_id",
        "pre_image_filename",
        "pre_image_path",
        "post_image_filename",
        "post_image_path",
        "missing_models",
    ]
    for alias in model_files.keys():
        table_fields.append(f"{alias}_answer")
    write_csv(review_table_csv_path, table_rows, table_fields)

    key_fields = [
        "task_id",
        "image_filename",
        "pre_image_filename",
        "post_image_filename",
        "auto_best_model",
        "auto_best_score",
        "missing_models",
    ]
    max_options = max((len([k for k in row.keys() if k.startswith("option_")]) for row in key_rows), default=0)
    slots = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(max_options):
        key_fields.append(f"option_{slots[i]}")
    write_csv(answer_key_csv_path, key_rows, key_fields)

    copy_warnings: List[str] = []
    if copy_images:
        pre_sampled_dir = output_dir / "pre_sampled_images"
        post_sampled_dir = output_dir / "post_sampled_images"
        copy_warnings = copy_sampled_pairs(
            sampled_pre_images=sampled_pre_images,
            post_index=post_image_index,
            pre_output_dir=pre_sampled_dir,
            post_output_dir=post_sampled_dir,
        )
        warnings.extend(copy_warnings)

    meta = {
        "pre_image_dir": str(pre_image_dir),
        "post_image_dir": str(post_image_dir),
        "auto_best_file": str(auto_best_file),
        "model_files": {k: str(v) for k, v in model_files.items()},
        "total_pre_images": len(all_pre_images),
        "sampled_images": len(sampled_pre_images),
        "sample_ratio": sample_ratio,
        "seed": seed,
        "warnings_count": len(warnings),
        "warnings": warnings,
        "outputs": {
            "blind_review_jsonl": str(blind_jsonl_path),
            "sampled_review_table_csv": str(review_table_csv_path),
            "answer_key_csv": str(answer_key_csv_path),
            "pre_sampled_images_dir": str(output_dir / "pre_sampled_images"),
            "post_sampled_images_dir": str(output_dir / "post_sampled_images"),
        }
    }
    write_json(meta_json_path, meta)

    print(f"[OK] 抽样图片数: {len(sampled_pre_images)} / {len(all_pre_images)}")
    print(f"[OK] blind_review.jsonl: {blind_jsonl_path}")
    print(f"[OK] sampled_review_table.csv: {review_table_csv_path}")
    print(f"[OK] answer_key.csv: {answer_key_csv_path}")
    print(f"[OK] prepare_meta.json: {meta_json_path}")
    if copy_images:
        print(f"[OK] 前时相子集已保存到: {output_dir / 'pre_sampled_images'}")
        print(f"[OK] 后时相子集已保存到: {output_dir / 'post_sampled_images'}")
    if warnings:
        print(f"[WARN] 发现 {len(warnings)} 条警告，请查看 prepare_meta.json")


if __name__ == "__main__":
    main()