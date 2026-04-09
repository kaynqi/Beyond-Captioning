import os
import json
import cv2
import numpy as np
from math import sqrt
from collections import Counter
from skimage.measure import label, regionprops
from tqdm import tqdm

# ==========================
# Class ID → Land-cover Name
# ==========================
label_map = {
    11: "Paddy Field",
    12: "Irrigated Cropland",
    13: "Rainfed Cropland",
    21: "Orchard",
    22: "Tea Plantation",
    23: "Other Plantations",
    31: "Forest Land",
    32: "Shrubland",
    33: "Other Woodland",
    41: "Natural Grassland",
    42: "Artificial Grassland",
    43: "Other Grassland",
    51: "Urban Built-up Land",
    52: "Rural Residential Land",
    53: "Artificial Excavation Land",
    54: "Other Built-up Land",
    61: "Rural Roads",
    62: "Other Transportation Land",
    71: "Rivers and Reservoirs",
    72: "Wetlands",
    73: "Permanent Ice and Snow",
    81: "Saline Land",
    82: "Sandy Land",
    83: "Bare Soil Land",
    84: "Rocky Bare Land"
}


# ==========================
# Utils
# ==========================
def load_label_image(image_path):
    """
    使用 OpenCV 加载单通道标签图
    IMREAD_UNCHANGED 确保不丢失原始类别 ID (0-80)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 如果是多通道，只取第一个通道
    if len(img.shape) == 3:
        img = img[:, :, 0]
    return img


def pos_to_relative_location(cx, cy, H, W):
    """映射质心坐标到 3×3 空间区域"""
    xr = cx / W
    yr = cy / H

    if yr < 1 / 3:
        vert = "upper"
    elif yr > 2 / 3:
        vert = "lower"
    else:
        vert = "central"

    if xr < 1 / 3:
        hori = "left"
    elif xr > 2 / 3:
        hori = "right"
    else:
        hori = "middle"

    return f"{vert}-{hori}"


# ======================================
# Change-Object Aggregation (Key Module)
# ======================================
MERGE_DIST_THRESHOLD = 40  # 像素距离阈值，可调


def merge_regions(regions):
    """
    聚合破碎的变化斑块：
    - 相同的变化前类型
    - 相同的变化后类型
    - 相同的空间位置 (3x3 grid)
    - 质心距离 < 阈值
    """
    merged = []

    for r in regions:
        assigned = False
        for m in merged:
            # 语义一致性 + 空间位置一致性检查
            if (
                    r["before_name"] == m["before_name"]
                    and r["after_name"] == m["after_name"]
                    and r["position"] == m["position"]
            ):
                # 计算欧几里得质心距离
                d = sqrt((r["cx"] - m["cx"]) ** 2 + (r["cy"] - m["cy"]) ** 2)

                if d < MERGE_DIST_THRESHOLD:
                    # 更新质心 (加权平均)
                    total_sub = m["num_subregions"] + 1
                    m["cx"] = (m["cx"] * m["num_subregions"] + r["cx"]) / total_sub
                    m["cy"] = (m["cy"] * m["num_subregions"] + r["cy"]) / total_sub
                    m["num_subregions"] += 1
                    assigned = True
                    break

        if not assigned:
            r["num_subregions"] = 1
            merged.append(r)

    return merged


# ==========================
# Extract Change Regions
# ==========================
def extract_change_regions(img24_path, img25_path):
    A = load_label_image(img24_path)
    B = load_label_image(img25_path)

    H, W = A.shape

    # 仅提取发生变化的区域 (且排除背景值 0)
    change_mask = (A != B) & (A != 0) & (B != 0)

    # 连通域分析
    cc = label(change_mask, connectivity=2)
    raw_regions = []

    for r in regionprops(cc):
        ys, xs = r.coords[:, 0], r.coords[:, 1]

        # 统计该区域内最主要的类别
        before_main = int(Counter(A[ys, xs]).most_common(1)[0][0])
        after_main = int(Counter(B[ys, xs]).most_common(1)[0][0])

        cy, cx = r.centroid
        loc = pos_to_relative_location(cx, cy, H, W)

        raw_regions.append({
            "before_name": label_map.get(before_main, f"Label_{before_main}"),
            "after_name": label_map.get(after_main, f"Label_{after_main}"),
            "position": loc,
            "cx": float(cx),
            "cy": float(cy)
        })

    # 对象级聚合步骤
    merged_regions = merge_regions(raw_regions)

    # 整理最终输出格式
    clean_regions = [
        {
            "before_name": r["before_name"],
            "after_name": r["after_name"],
            "position": r["position"],
            "num_subregions": r["num_subregions"]
        }
        for r in merged_regions
    ]

    return clean_regions


# ==========================
# Batch Processing
# ==========================
def batch_generate_change_regions(path_2024, path_2025, output_json):
    results = {}

    # 过滤获取所有 2024 的 PNG 文件
    files_2024 = [f for f in os.listdir(path_2024) if f.endswith(".png")]
    print(f"--- 开始处理: 共有 {len(files_2024)} 组文件 ---")

    # 按照 ID 排序，保证 JSON 输出有序
    files_2024.sort()

    for fname_24 in tqdm(files_2024, desc="分析进度"):
        # 1. 提取核心 ID (从 "00001_2024_label.png" 提取 "00001")
        sample_id = fname_24.split("_")[0]

        # 2. 构造对应的 2025 文件名
        fname_25 = f"{sample_id}_2025_label.png"

        img2024_path = os.path.join(path_2024, fname_24)
        img2025_path = os.path.join(path_2025, fname_25)

        # 3. 检查配对是否存在
        if not os.path.exists(img2025_path):
            continue

        # 4. 提取并分析
        try:
            regions = extract_change_regions(img2024_path, img2025_path)

            # 记录结果（包含变化数量和详细信息）
            results[sample_id] = {
                "num_change_regions": len(regions),
                "regions": regions
            }
        except Exception as e:
            print(f"\n❌ 处理 ID {sample_id} 时发生错误: {e}")

    # 5. 保存结果
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n--- 🎉 处理完成！---")
    print(f"JSON 结果保存位置: {output_json}")


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    # 配置你的路径
    # BASE_DIR = "/home/xutao/jq/nyq"
    #
    # PATH_24 = os.path.join(BASE_DIR, "2024_label")
    # PATH_25 = os.path.join(BASE_DIR, "2025_label")
    # OUTPUT = os.path.join(BASE_DIR, "change_regions_merged.json")
    #
    # # 执行
    # batch_generate_change_regions(PATH_24, PATH_25, OUTPUT)

# if __name__ == "__main__":
#
    batch_generate_change_regions(
        path_2024=r"G:\ct\our_dataset\val\nochange\2024_label",
        path_2025=r"G:\ct\our_dataset\val\nochange\2025_label",
        output_json=r"G:\ct\our_dataset\val\nochange_regions_merged.json"
    )

    # batch_generate_change_regions(
    #     path_2024=r"G:\ct\our_dataset\test\nochange\2024_label",
    #     path_2025=r"G:\ct\our_dataset\test\nochange\2025_label",
    #     output_json="nochange_regions_merged_test.json"
    # )
