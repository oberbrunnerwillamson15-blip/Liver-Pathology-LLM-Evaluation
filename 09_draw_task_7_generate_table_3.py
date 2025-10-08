# -*- coding: utf-8 -*-
"""
任务7: 生成 [表3] 典型错误案例分析表
- 直接读取用户精心挑选的 `典型病例.csv` 文件。
- 从文件名中解析出模型名称并进行清理。
- 根据真实标签和预测标签，自动生成“错误类型”描述。
- 将所有信息整理成一个结构化的表格。
- 将最终表格保存为 Markdown (.md) 和 CSV (.csv) 格式。
"""

import pandas as pd
import os
import re

# ==============================================================================
# 配置区 (Configuration Area)
# ==============================================================================
# --- 输入文件路径 ---
# 注意：请确保您已将您在提示中提供的 "典型病例.csv内容" 保存为此文件。
# 如果您尚未创建该文件，请将内容复制粘贴到一个名为 典型病例.csv 的文件中。
INPUT_CSV_PATH = r"D:\工作\小论文v2_20250801_v2\code\results\典型病例.csv"

# --- 输出文件路径 ---
OUTPUT_DIR = r"D:\工作\小论文v2_20250801_v2\code\results\09_draw"
OUTPUT_FILENAME_MD = "Table_3_Error_Case_Analysis.md"
OUTPUT_FILENAME_CSV = "Table_3_Error_Case_Analysis.csv"

# --- 逻辑配置 ---
# 定义用于识别“伪错误”的特定 Patient_ID
PSEUDO_ERROR_IDS = [2320055019, 2220048409]

# ==============================================================================
# 工作区 (Workspace Area)
# ==============================================================================

def extract_and_clean_model_name(filename_str):
    """从 'model_names' 列中提取并清理模型名称"""
    # 提取 'preds_' 和 '_task_' 之间的部分
    match = re.search(r'preds_(.*?)_task_', filename_str)
    if not match:
        return "Unknown Model"
    
    name = match.group(1)
    # 移除 .gguf 后缀 (如果存在)
    if name.endswith('.gguf'):
        name = name[:-5]
    # 移除常见的量化标识
    name = re.sub(r'-Q[A-Z0-9_]+$', '', name, flags=re.IGNORECASE)
    return name

def map_task_display_name(task_column):
    """将任务的技术名称映射为更易读的显示名称"""
    mapping = {
        'Fibrosis_Stage_0_4': 'Fibrosis Staging',
        'Inflammation_Grade_0_4': 'Inflammation Grading',
        'Steatosis_Grade_1_3': 'Steatosis Grading'
    }
    return mapping.get(task_column, task_column)

def determine_error_type(row):
    """根据标签差异和 Patient_ID 确定错误类型"""
    patient_id = row['Patient_ID']
    true_label = row['True_Label']
    predicted_label = row['Predicted_Label']
    
    # 1. 检查是否为预定义的“伪错误”
    if patient_id in PSEUDO_ERROR_IDS:
        return "Pseudo-Error (Data Quality Issue)"
        
    # 2. 计算标签差异
    try:
        diff = abs(int(true_label) - int(predicted_label))
        if diff == 1:
            if predicted_label > true_label:
                return "Neighboring Mismatch (Overestimation)"
            else:
                return "Neighboring Mismatch (Underestimation)"
        elif diff > 1:
            if predicted_label > true_label:
                return "Major Misclassification (Severe Overestimation)"
            else:
                return "Major Misclassification (Severe Underestimation)"
        else: # diff == 0
            return "Correct Prediction"
    except (ValueError, TypeError):
        return "Invalid Label"

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
        print(f"成功从 '{INPUT_CSV_PATH}' 加载 {len(df)} 个典型案例。")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{INPUT_CSV_PATH}'。请确保文件存在并路径正确。")
        return
    except Exception as e:
        print(f"错误：加载文件时出错。错误信息: {e}")
        return
        
    # --- 3. 创建最终表格所需的列 ---
    df['Case_ID'] = [f"Case {i+1}" for i in range(len(df))]
    df['Model'] = df['model_names'].apply(extract_and_clean_model_name)
    df['Task_Display'] = df['Task'].apply(map_task_display_name)
    df['Error_Type'] = df.apply(determine_error_type, axis=1)

    # --- 4. 整理并选择最终的列 ---
    final_df = df[[
        'Case_ID',
        'Patient_ID',
        'Task_Display',
        'True_Label',
        'Predicted_Label',
        'Model',
        'Error_Type',
        'Pathology_Text'
    ]].rename(columns={
        'Task_Display': 'Task',
        'Pathology_Text': 'Key Pathology Snippet' # 按照论文要求重命名
    })
    
    # --- 5. 保存为 CSV 和 Markdown 文件 ---
    output_path_csv = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_CSV)
    output_path_md = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_MD)

    try:
        # 保存为 CSV
        final_df.to_csv(output_path_csv, index=False, encoding='utf-8-sig')
        print(f"\n表格已成功保存为 CSV 格式: '{output_path_csv}'")
        
        # 保存为 Markdown
        markdown_string = final_df.to_markdown(index=False)
        with open(output_path_md, 'w', encoding='utf-8') as f:
            f.write(markdown_string)
        print(f"表格已成功保存为 Markdown 格式: '{output_path_md}'")
        
    except Exception as e:
        print(f"错误：保存文件时出错。错误信息: {e}")
        return

    # --- 6. 在控制台打印预览 ---
    print("\n" + "="*80)
    print("任务7: [表3] 典型错误案例分析表 (控制台预览)")
    print("="*80)
    print(markdown_string)
    print("="*80)

if __name__ == "__main__":
    main()