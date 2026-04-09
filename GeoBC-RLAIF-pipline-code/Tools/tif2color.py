import os
import cv2
import numpy as np

# 输入 / 输出目录
input_folder = "E:/3705/370502/2025_label"
output_folder = "E:/3705/370502/2025_label_color"
os.makedirs(output_folder, exist_ok=True)

# BGR
palette = {
    0:  (255,255,255),   # 背景

    # --- 耕地 ---
    11: (102,194,165),   # 水田
    12: (141,211,199),   # 水浇地
    13: (230,171,2),     # 旱地

    # --- 园地 ---
    21: (253,180,98),    # 果园
    22: (179,222,105),   # 茶园
    23: (255,255,179),   # 其他园地

    # --- 林地 ---
    31: (27,120,55),     # 有林地
    32: (90,174,97),     # 灌木林地
    33: (166,219,160),   # 其他林地

    # --- 草地 ---
    41: (217,240,163),   # 天然牧草地
    42: (247,252,185),   # 人工牧草地
    43: (255,255,204),   # 其他草地

    # --- 建设用地 ---
    51: (178,24,43),     # 城镇建设用地
    52: (239,138,98),    # 农村建设用地
    53: (128,115,172),   # 人为坑塘
    54: (244,165,130),   # 其他建设用地

    # --- 交通用地 ---
    61: (153,153,153),   # 农村道路
    62: (102,102,102),   # 其他交通用地

    # --- 水域与湿地 ---
    71: (77,163,255),    # 河湖库塘
    72: (127,205,187),   # 沼泽地
    73: (255,255,255),   # 冰川及积雪

    # --- 未利用地 ---
    81: (217,217,217),   # 盐碱地
    82: (238,207,139),   # 沙地
    83: (194,154,107),   # 裸土地
    84: (140,109,72),    # 裸岩石砾地
}


# 遍历整个文件夹
for file in os.listdir(input_folder):
    if file.lower().endswith((".png", ".jpg", ".tif", ".tiff")):

        in_path = os.path.join(input_folder, file)

        # 读取灰度标签
        gray = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)

        if gray is None:
            print("无法读取", file)
            continue

        # 创建空 RGB 图
        h, w = gray.shape
        color = np.zeros((h, w, 3), np.uint8)

        # 灰度映射到 RGB
        for cls, rgb in palette.items():
            color[gray == cls] = rgb

        # 统一保存为 PNG
        filename = os.path.splitext(file)[0] + ".png"
        out_path = os.path.join(output_folder, filename)

        cv2.imwrite(out_path, color)
        print("已保存:", out_path)

print("\n全部处理完成！")