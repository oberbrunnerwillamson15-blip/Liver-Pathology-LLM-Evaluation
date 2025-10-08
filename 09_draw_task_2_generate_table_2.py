# -*- coding: utf-8 -*-
"""
任务2: 生成 [表2] 所有模型性能总览表
- 读取 aggregated_summary.csv 文件
- 清理模型名称，并结合 reasoning_mode 创建唯一的行标识
- 将数据重塑为宽格式，以模型为行，任务和指标为多级列
- 在控制台打印一个带有高亮（Markdown加粗）的版本
- 将最终的、干净的表格保存为 CSV 文件
"""

import pandas as pd
import os
import re

# ==============================================================================
# 配置区 (Configuration Area)
# ==============================================================================
# --- 输入文件路径 ---
# 包含所有模型聚合性能数据的CSV文件
INPUT_CSV_PATH = r"D:\工作\小论文v2_20250801_v2\code\results\07_evaluation_llamacpp_linux_v1.1_output\aggregated_summary.csv"

# --- 输出文件路径 ---
# 所有生成的图表和表格将保存在此目录
OUTPUT_DIR = r"D:\工作\小论文v2_20250801_v2\code\results\09_draw"
OUTPUT_FILENAME = "Table_2_Model_Performance_Overview.csv"

# ==============================================================================
# 工作区 (Workspace Area)
# ==============================================================================

def clean_model_name(name):
    """
    清理模型文件名，移除版本、量化和文件后缀信息，以获得更干净的显示名称。
    例如： "Qwen3-14B-Q4_K_M.gguf" -> "Qwen3-14B"
    """
    # 移除 .gguf 后缀
    if name.endswith('.gguf'):
        name = name[:-5]
    # 使用正则表达式移除常见的量化标识，如 -Q4_K_M
    # 这个表达式会移除 -Q 跟随的任何数字、下划线、字母组合
    name = re.sub(r'-Q[A-Z0-9_]+$', '', name, flags=re.IGNORECASE)
    # 移除特定的、不遵循上述模式的后缀
    # name = name.replace('-Q4_K_M', '') # 一个更简单的替代方法
    return name

def main():
    """主执行函数"""
    # --- 1. 准备输出目录 ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"输出目录 '{OUTPUT_DIR}' 已准备就绪。")
    except OSError as e:
        print(f"错误：创建目录 '{OUTPUT_DIR}' 失败。错误信息: {e}")
        return

    # --- 2. 加载数据 ---
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        # 确保 reasoning_mode 列存在，如果不存在则创建并填充空值
        if 'reasoning_mode' not in df.columns:
            df['reasoning_mode'] = pd.NA
        print(f"成功从 '{INPUT_CSV_PATH}' 加载 {len(df)} 条记录。")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{INPUT_CSV_PATH}'。请检查路径是否正确。")
        return
    except Exception as e:
        print(f"错误：加载文件时出错。错误信息: {e}")
        return

    # --- 3. 数据预处理和清理 ---
    # 清理模型名称
    df['model_cleaned'] = df['model'].apply(clean_model_name)

    # 处理 reasoning_mode，填充 NaN 为空字符串以便拼接
    df['reasoning_mode'] = df['reasoning_mode'].fillna('')

    # 创建用于表格行的最终显示名称
    # 如果有 reasoning_mode，则格式为 "Model (mode)"，否则就是模型名
    df['display_name'] = df.apply(
        lambda row: f"{row['model_cleaned']} ({row['reasoning_mode']})" if row['reasoning_mode'] else row['model_cleaned'],
        axis=1
    )
    
    # --- 4. 数据重塑 (Pivot) ---
    # 将长格式数据转换为宽格式
    df_pivot = df.pivot_table(
        index='display_name',
        columns='task_column',
        values=['f1_mean', 'acc_mean']
    )

    # --- 5. 调整列的顺序和名称以匹配论文要求 ---
    # 定义期望的列顺序
    task_order = ['Fibrosis_Stage_0_4', 'Inflammation_Grade_0_4', 'Steatosis_Grade_1_3']
    metric_order = ['f1_mean', 'acc_mean']
    
    # 根据期望顺序重新组织列
    final_columns = [(metric, task) for task in task_order for metric in metric_order]
    df_pivot = df_pivot[final_columns]

    # 定义列名映射
    task_name_map = {
        'Fibrosis_Stage_0_4': 'Fibrosis Staging',
        'Inflammation_Grade_0_4': 'Inflammation Grading',
        'Steatosis_Grade_1_3': 'Steatosis Grading'
    }
    metric_name_map = {
        'f1_mean': 'F1-Score',
        'acc_mean': 'Accuracy'
    }

    # 重命名列
    df_pivot.rename(columns=task_name_map, level=1, inplace=True)
    df_pivot.rename(columns=metric_name_map, level=0, inplace=True)
    df_pivot.index.name = "Model"

    # --- 6. 保存为 CSV 文件 ---
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        df_pivot.to_csv(output_path, encoding='utf-8-sig', float_format='%.4f')
        print(f"\n表格已成功保存至: '{output_path}'")
    except IOError as e:
        print(f"错误: 无法写入文件 '{output_path}'. 错误信息: {e}")
        return

    # --- 7. 创建并打印用于控制台预览的 Markdown 表格 (带高亮) ---
    df_display = df_pivot.copy()
    
    # 格式化所有数值为字符串，并找到每个任务的最高F1分数进行加粗
    for task_name in task_name_map.values():
        f1_col = ('F1-Score', task_name)
        if f1_col in df_display.columns:
            # 找到F1分数最高的模型的索引（行名）
            max_f1_idx = df_display[f1_col].idxmax()
            
            # 将该单元格的值格式化并加粗
            max_f1_val = df_display.loc[max_f1_idx, f1_col]
            df_display.loc[max_f1_idx, f1_col] = f"**{max_f1_val:.4f}**"

    # 格式化其他所有数值
    for col in df_display.columns:
        df_display[col] = df_display[col].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (float, int)) and '**' not in str(x) else x
        )

    print("\n" + "="*80)
    print("任务2: [表2] 所有模型性能总览表 (控制台预览版，最高F1分数已加粗)")
    print("="*80)
    # 使用 to_markdown 打印格式化的表格
    print(df_display.to_markdown())
    print("="*80)

if __name__ == "__main__":
    main()