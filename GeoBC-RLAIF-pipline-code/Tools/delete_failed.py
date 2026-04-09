import json
import os


def filter_failed_json_tasks(input_filepath, output_filepath):
    """
    读取包含多个 JSON 的文件，检查 'success' 字段。
    如果 'success' 为 False，则过滤掉该条数据；保留为 True 的数据。
    """
    if not os.path.exists(input_filepath):
        print(f"错误: 找不到输入文件 -> {input_filepath}")
        return

    kept_count = 0
    removed_count = 0

    # 以 UTF-8 编码读取和写入，防止中文或特殊字符乱码
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
            open(output_filepath, 'w', encoding='utf-8') as outfile:

        # 逐行读取（适用于 .jsonl 或者每行一个独立 {} 的 .txt/.json 文件）
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行

            try:
                # 解析当前行的 JSON
                data = json.loads(line)

                # 检查 success 状态。使用 .get() 防止某些行缺少 'success' 键而报错
                # 只有当 success 明确为 True 时才保留
                if data.get("success") is True:
                    # 将保留的数据重新转为字符串并写入新文件
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    kept_count += 1
                else:
                    # success 为 False 或 null
                    removed_count += 1

            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行解析失败，非标准 JSON 格式。跳过此行。错误信息: {e}")

    print("-" * 30)
    print("清理完成！")
    print(f"-> 保留了 {kept_count} 条成功的数据。")
    print(f"-> 删除了 {removed_count} 条失败的数据。")
    print(f"-> 清理后的数据已保存至: {output_filepath}")


# ================= 使用说明 =================
if __name__ == "__main__":
    # 替换为你的实际文件路径
    input_file = "judge/gemini_answer_change.jsonl"  # 包含原始数据的输入文件
    output_file = "judge/gemini_answer_change_clean.jsonl"  # 过滤掉 false 之后保存的新文件

    filter_failed_json_tasks(input_file, output_file)