# -*- coding: utf-8 -*-
"""
任务3: 生成 [图1] 主要模型家族性能对比图
- 读取 aggregated_summary.csv 文件
- 筛选出 Qwen, Gemma, DeepSeek 三个家族的代表性模型
- 绘制分组条形图，对比它们在三个任务上的宏平均 F1 分数
- 使用专业配色方案，并添加数据标签
- 将图表以 300 DPI 的高分辨率保存为 PNG 和 TIF 格式
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# ==============================================================================
# 配置区 (Configuration Area)
# ==============================================================================
# --- 输入文件路径 ---
INPUT_CSV_PATH = r"D:\工作\小论文v2_20250801_v2\code\results\07_evaluation_llamacpp_linux_v1.1_output\aggregated_summary.csv"

# --- 输出文件路径 ---
OUTPUT_DIR = r"D:\工作\小论文v2_20250801_v2\code\results\09_draw"
# 为不同格式定义文件名
OUTPUT_FILENAME_PNG = "Figure_1_Model_Family_Comparison.png"
OUTPUT_FILENAME_TIF = "Figure_1_Model_Family_Comparison.tif"

# --- 图表配置 ---
# 选择要进行比较的代表性模型 (使用 .gguf 全名以确保精确匹配)
# 这些是您论文中提到的在各个任务上表现最佳的模型
REPRESENTATIVE_MODELS = [
    'Qwen3-14B-Q4_K_M.gguf',                  # Best in Fibrosis
    'gemma-3-4b-it-Q4_K_M.gguf',              # Best in Inflammation
    'DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf' # Best in Steatosis
]

# 为图例和标签创建更简洁的显示名称
MODEL_DISPLAY_NAMES = {
    'Qwen3-14B-Q4_K_M.gguf': 'Qwen3-14B',
    'gemma-3-4b-it-Q4_K_M.gguf': 'gemma-3-4b-it',
    'DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf': 'DeepSeek-R1-32B'
}

# 任务显示名称
TASK_DISPLAY_NAMES = {
    'Fibrosis_Stage_0_4': 'Fibrosis Staging',
    'Inflammation_Grade_0_4': 'Inflammation Grading',
    'Steatosis_Grade_1_3': 'Steatosis Grading'
}

# 商业级调色板 (例如 'viridis', 'mako', 'rocket', 'crest' 或自定义列表)
COLOR_PALETTE = "viridis"

# 图像分辨率
DPI = 300

# ==============================================================================
# 工作区 (Workspace Area)
# ==============================================================================

def main():
    """主执行函数"""
    # --- 1. 准备输出目录 ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"输出目录 '{OUTPUT_DIR}' 已准备就绪。")
    except OSError as e:
        print(f"错误：创建目录 '{OUTPUT_DIR}' 失败。错误信息: {e}")
        return

    # --- 2. 加载并筛选数据 ---
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        # 筛选出我们感兴趣的代表性模型
        df_filtered = df[df['model'].isin(REPRESENTATIVE_MODELS)].copy()
        print(f"成功加载并筛选出 {len(df_filtered)} 条代表性模型数据。")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{INPUT_CSV_PATH}'。")
        return
    except Exception as e:
        print(f"错误：加载或处理文件时出错。错误信息: {e}")
        return

    if df_filtered.empty:
        print("错误：筛选后数据为空，请检查 'REPRESENTATIVE_MODELS' 列表中的模型名称是否与CSV文件中的完全匹配。")
        return
        
    # --- 3. 数据准备以供绘图 ---
    # 应用更清晰的显示名称
    df_filtered['model_display'] = df_filtered['model'].map(MODEL_DISPLAY_NAMES)
    df_filtered['task_display'] = df_filtered['task_column'].map(TASK_DISPLAY_NAMES)

    # --- 4. 绘图 ---
    # 设置绘图风格和上下文，以获得更美观的字体和布局
    sns.set_theme(style="whitegrid", context="talk")

    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(14, 8))

    # 绘制分组条形图
    sns.barplot(
        data=df_filtered,
        x='task_display',
        y='f1_mean',
        hue='model_display',
        palette=COLOR_PALETTE,
        ax=ax
    )

    # 添加数据标签到每个条形图的顶部
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=12, padding=3)

    # --- 5. 美化图表 ---
    # ax.set_title('Performance Comparison of Representative Models Across Tasks', fontsize=20, pad=20)
    ax.set_xlabel('')  # X轴标签是自解释的，所以留空
    ax.set_ylabel('Macro F1-Score', fontsize=16)
    ax.set_ylim(0, 0.9)  # 设置Y轴范围以更好地显示差异
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=12)

    # 调整图例位置，防止遮挡
    plt.legend(title='Model', fontsize=12, title_fontsize=14)
    
    # 自动调整布局，确保所有元素都清晰可见
    plt.tight_layout()

    # --- 6. 保存图表 ---
    output_path_png = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_PNG)
    output_path_tif = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_TIF)
    
    try:
        # 保存为 PNG 格式
        plt.savefig(output_path_png, dpi=DPI, bbox_inches='tight')
        print(f"\n图表已成功保存为 PNG 格式: '{output_path_png}' (DPI={DPI})")
        
        # 保存为 TIF 格式
        plt.savefig(output_path_tif, dpi=DPI, bbox_inches='tight')
        print(f"图表已成功保存为 TIF 格式: '{output_path_tif}' (DPI={DPI})")
        
    except Exception as e:
        print(f"错误：保存图表时出错。错误信息: {e}")

    # (可选) 在脚本运行时显示图表
    # plt.show()
    
    print("\n" + "="*80)
    print("任务3: [图1] 主要模型家族性能对比图已生成并保存。")
    print("="*80)

if __name__ == "__main__":
    main()