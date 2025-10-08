# -*- coding: utf-8 -*-
"""
任务1 (修改版): 生成 [表1] 患者基线特征表
- 读取 data.json 文件
- 计算人口统计学、合并症和病理分级的统计数据
- 将结果构建成一个 DataFrame
- 将 DataFrame 保存为 CSV 文件
- 将 DataFrame 转换为 Markdown 格式并保存，同时在控制台打印
"""

import pandas as pd
import numpy as np
import os
import json

# ==============================================================================
# 配置区 (Configuration Area)
# ==============================================================================
# --- 输入文件路径 ---
DATA_JSON_PATH = r"D:\工作\小论文v2_20250801_v2\code\data.json"

# --- 输出文件路径 ---
OUTPUT_DIR = r"D:\工作\小论文v2_20250801_v2\code\results\09_draw"
# 为不同格式定义文件名
OUTPUT_FILENAME_MD = "Table_1_Baseline_Characteristics.md"
OUTPUT_FILENAME_CSV = "Table_1_Baseline_Characteristics.csv"

# ==============================================================================
# 工作区 (Workspace Area)
# ==============================================================================

def clean_and_convert_to_numeric(series):
    """辅助函数，用于处理可能包含分号分隔值的字符串列。"""
    first_value_str = series.astype(str).str.split(';').str[0]
    return pd.to_numeric(first_value_str, errors='coerce')

def main():
    """主执行函数"""
    # --- 1. 准备输出目录 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录 '{OUTPUT_DIR}' 已准备就绪。")

    # --- 2. 加载并预处理数据 ---
    try:
        df = pd.read_json(DATA_JSON_PATH)
        print(f"成功从 '{DATA_JSON_PATH}' 加载 {len(df)} 条患者记录。")
    except Exception as e:
        print(f"错误：无法加载或解析 '{DATA_JSON_PATH}'。错误信息: {e}")
        return

    if 'BMI_kg_m2' in df.columns and df['BMI_kg_m2'].dtype == 'object':
        df['BMI_kg_m2'] = clean_and_convert_to_numeric(df['BMI_kg_m2'])

    # --- 3. 计算统计数据 ---
    total_patients = len(df)
    
    # 人口统计学特征
    age_mean, age_std = df['Age_years'].mean(), df['Age_years'].std()
    bmi_mean, bmi_std = df['BMI_kg_m2'].mean(), df['BMI_kg_m2'].std()
    gender_counts = df['Gender_0F_1M'].value_counts()
    female_n, male_n = gender_counts.get(0, 0), gender_counts.get(1, 0)
    
    # 合并症
    hypertension_n = df['Hypertension_0N_1Y'].sum()
    diabetes_n = df['Diabetes_0N_1Y'].sum()
    
    # 病理分级分布
    fibrosis_dist = df['Fibrosis_Stage_0_4'].value_counts().sort_index()
    inflammation_dist = df['Inflammation_Grade_0_4'].value_counts().sort_index()
    steatosis_dist = df['Steatosis_Grade_1_3'].value_counts().sort_index()

    # --- 4. 构建结构化的数据列表以创建DataFrame ---
    table_data = []
    
    table_data.append({'Characteristic': f'Total Patients, N', 'Value': total_patients})
    table_data.append({'Characteristic': '--- Demographics ---', 'Value': ''})
    table_data.append({'Characteristic': 'Age, years (mean ± SD)', 'Value': f'{age_mean:.1f} ± {age_std:.1f}'})
    table_data.append({'Characteristic': 'Gender, n (%)', 'Value': ''})
    table_data.append({'Characteristic': '  Female', 'Value': f'{female_n} ({female_n / total_patients:.1%})'})
    table_data.append({'Characteristic': '  Male', 'Value': f'{male_n} ({male_n / total_patients:.1%})'})
    table_data.append({'Characteristic': 'BMI, kg/m² (mean ± SD)', 'Value': f'{bmi_mean:.1f} ± {bmi_std:.1f}'})
    
    table_data.append({'Characteristic': '--- Comorbidities ---', 'Value': ''})
    table_data.append({'Characteristic': 'Hypertension, n (%)', 'Value': f'{hypertension_n} ({hypertension_n / total_patients:.1%})'})
    table_data.append({'Characteristic': 'Diabetes, n (%)', 'Value': f'{diabetes_n} ({diabetes_n / total_patients:.1%})'})
    
    table_data.append({'Characteristic': '--- Pathological Features ---', 'Value': ''})
    table_data.append({'Characteristic': 'Fibrosis Stage (S0-S4), n (%)', 'Value': ''})
    for stage, count in fibrosis_dist.items():
        table_data.append({'Characteristic': f'  Stage {stage}', 'Value': f'{count} ({count / total_patients:.1%})'})
        
    table_data.append({'Characteristic': 'Inflammation Grade (G0-G4), n (%)', 'Value': ''})
    for grade, count in inflammation_dist.items():
        table_data.append({'Characteristic': f'  Grade {grade}', 'Value': f'{count} ({count / total_patients:.1%})'})
        
    table_data.append({'Characteristic': 'Steatosis Grade (S1-S3), n (%)', 'Value': ''})
    for grade, count in steatosis_dist.items():
        table_data.append({'Characteristic': f'  Grade {grade}', 'Value': f'{count} ({count / total_patients:.1%})'})

    # 将数据列表转换为 DataFrame
    df_table = pd.DataFrame(table_data)

    # --- 5. 输出到文件 ---
    # 保存为 CSV 文件
    output_path_csv = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_CSV)
    try:
        df_table.to_csv(output_path_csv, index=False, encoding='utf-8-sig')
        print(f"\n表格已成功保存为 CSV 格式: '{output_path_csv}'")
    except IOError as e:
        print(f"错误: 无法写入CSV文件 '{output_path_csv}'. 错误信息: {e}")
        return

    # 保存为 Markdown 文件
    output_path_md = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_MD)
    try:
        # 使用 to_markdown 生成更美观的表格
        markdown_string = df_table.to_markdown(index=False)
        with open(output_path_md, 'w', encoding='utf-8') as f:
            f.write(f"**Table 1: Baseline Characteristics of the Patient Cohort (N={total_patients})**\n\n")
            f.write(markdown_string)
        print(f"表格已成功保存为 Markdown 格式: '{output_path_md}'")
    except IOError as e:
        print(f"错误: 无法写入Markdown文件 '{output_path_md}'. 错误信息: {e}")
        return

    # --- 6. 输出到控制台 ---
    print("\n" + "="*80)
    print(f"任务1: [表1] 患者基线特征表 (N={total_patients})")
    print("="*80)
    print(markdown_string)
    print("="*80)

if __name__ == "__main__":
    main()