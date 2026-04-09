import os


def clean_redundant_images(ref_dir, target_dir):
    """
    根据 ref_dir 中文件的前 6 位字符，清理 target_dir 中多余的图片。
    """
    # 检查文件夹是否存在
    if not os.path.exists(ref_dir):
        print(f"错误: 供参考的第一个文件夹不存在 -> {ref_dir}")
        return
    if not os.path.exists(target_dir):
        print(f"错误: 需要清理的第二个文件夹不存在 -> {target_dir}")
        return

    # 1. 获取第一个文件夹中所有文件的前 6 位，存入集合 (set) 中以加快查找速度
    valid_prefixes = set()
    for filename in os.listdir(ref_dir):
        ref_file_path = os.path.join(ref_dir, filename)
        # 确保是文件而不是子文件夹
        if os.path.isfile(ref_file_path):
            prefix = filename[:6]
            valid_prefixes.add(prefix)

    print(f"在第一个文件夹中提取了 {len(valid_prefixes)} 个唯一的前缀。")

    # 2. 遍历第二个文件夹，寻找并删除不匹配的文件
    deleted_count = 0
    for filename in os.listdir(target_dir):
        target_file_path = os.path.join(target_dir, filename)

        # 确保处理的是文件
        if os.path.isfile(target_file_path):
            prefix = filename[:6]

            # 如果前 6 位不在我们的参考集合中，则删除
            if prefix not in valid_prefixes:
                try:
                    os.remove(target_file_path)
                    print(f"已删除: {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"无法删除文件 {filename}, 错误原因: {e}")

    print("-" * 30)
    print(f"清理完成！共从第二个文件夹中删除了 {deleted_count} 个多余的文件。")


# ================= 使用说明 =================
if __name__ == "__main__":
    # 请将这里的路径替换为你实际的文件夹路径
    folder1_path = r"G:\ct\our_dataset\test\nochange\2025_RGB"  # 参考文件夹 (提取前6位)
    folder2_path = r"G:\ct\our_dataset\test\nochange\2024_RGB"  # 目标文件夹 (包含要删除的图片)

    # 强烈建议：在第一次运行前，先备份你的第二个文件夹，以免误删！
    clean_redundant_images(folder1_path, folder2_path)