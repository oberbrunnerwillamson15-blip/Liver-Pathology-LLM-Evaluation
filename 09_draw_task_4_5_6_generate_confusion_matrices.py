# -*- coding: utf-8 -*-
"""
任务4-6: 生成 [图2, 图3, 图4] 代表性模型的混淆矩阵
- 循环处理三个核心场景（Fibrosis/Qwen3, Inflammation/gemma, Steatosis/DeepSeek）。
- 为每个场景加载对应的详细预测CSV文件。
- 计算混淆矩阵并使用 Seaborn Heatmap 进行可视化。
- 使用专业配色方案，并在每个单元格中标注样本数量。
- 将图表以 300 DPI 的高分辨率保存为 PNG 和 TIF 格式。
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import numpy as np

# ==============================================================================
# 配置区 (Configuration Area)
# ==============================================================================
# --- 输入文件路径 ---
# 存放所有 detailed_predictions/*.csv 文件的目录
PREDICTIONS_DIR = r"D:\工作\小论文v2_20250801_v2\code\results\detailed_predictions"

# --- 输出文件路径 ---
OUTPUT_DIR = r"D:\工作\小论文v2_20250801_v2\code\results\09_draw"

# --- 图表配置 ---
# 定义要生成的三个混淆矩阵的配置信息
FIGURE_CONFIGS = {
    "Figure_2": {
        "model_file_name": "Qwen3-14B-Q4_K_M",
        "task_name": "fibrosis",
        "task_display_name": "Fibrosis Staging",
        "model_display_name": "Qwen3-14B",
        "output_base_name": "Figure_2_Confusion_Matrix_Fibrosis_Qwen3-14B"
    },
    "Figure_3": {
        "model_file_name": "gemma-3-4b-it-Q4_K_M",
        "task_name": "inflammation",
        "task_display_name": "Inflammation Grading",
        "model_display_name": "gemma-3-4b-it",
        "output_base_name": "Figure_3_Confusion_Matrix_Inflammation_gemma-3-4b"
    },
    "Figure_4": {
        "model_file_name": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M",
        "task_name": "steatosis",
        "task_display_name": "Steatosis Grading",
        "model_display_name": "DeepSeek-R1-32B",
        "output_base_name": "Figure_4_Confusion_Matrix_Steatosis_DeepSeek-32B"
    }
}

# 商业级调色板 (蓝色系，清晰且专业)
COLOR_PALETTE = "Blues"

# 图像分辨率
DPI = 300

# ==============================================================================
# 工作区 (Workspace Area)
# ==============================================================================

def main():
    """主执行函数，循环生成所有配置好的混淆矩阵"""
    # --- 1. 准备输出目录 ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"输出目录 '{OUTPUT_DIR}' 已准备就绪。")
    except OSError as e:
        print(f"错误：创建目录 '{OUTPUT_DIR}' 失败。错误信息: {e}")
        return

    # --- 2. 循环生成每个图表 ---
    for fig_key, config in FIGURE_CONFIGS.items():
        print("\n" + "="*80)
        print(f"正在生成: {fig_key} - {config['task_display_name']} ({config['model_display_name']})")
        print("="*80)
        
        # --- 2.1 构建输入文件路径 ---
        # 文件名格式: preds_{model_file_name}_task_{task_name}_run_1.csv
        input_filename = f"preds_{config['model_file_name']}_task_{config['task_name']}_run_1.csv"
        input_path = os.path.join(PREDICTIONS_DIR, input_filename)

        # --- 2.2 加载并预处理数据 ---
        try:
            df = pd.read_csv(input_path)
            print(f"成功加载数据: '{input_path}'")
        except FileNotFoundError:
            print(f"错误：找不到预测文件 '{input_path}'。请检查文件名和路径。跳过此图表。")
            continue
        
        # 清理数据：移除任何预测失败的行 (Predicted_Label 为空)
        df.dropna(subset=['True_Label', 'Predicted_Label'], inplace=True)
        # 确保标签是整数类型
        df['True_Label'] = df['True_Label'].astype(int)
        df['Predicted_Label'] = df['Predicted_Label'].astype(int)

        # --- 2.3 计算混淆矩阵 ---
        # 动态确定标签范围，以确保矩阵大小正确
        labels = sorted(pd.unique(df[['True_Label', 'Predicted_Label']].values.ravel()))
        
        cm = confusion_matrix(
            y_true=df['True_Label'], 
            y_pred=df['Predicted_Label'], 
            labels=labels
        )
        # 将numpy数组包装成带标签的DataFrame，便于Seaborn绘图
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        # --- 2.4 绘图 ---
        sns.set_theme(style="white", context="talk")
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm_df,
            annot=True,          # 在单元格中显示数值
            fmt='d',             # 格式化为整数
            cmap=COLOR_PALETTE,
            linewidths=0.5,      # 单元格之间的线条宽度
            cbar=False,          # 隐藏颜色条，因为数值已标注
            annot_kws={"size": 16} # 调整注释字体大小
        )

        # --- 2.5 美化图表 ---
        # title = f"Confusion Matrix for {config['task_display_name']}\nModel: {config['model_display_name']}"
        # ax.set_title(title, fontsize=20, pad=20)
        ax.set_xlabel('Predicted Label', fontsize=16)
        ax.set_ylabel('True Label', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        plt.tight_layout()

        # --- 2.6 保存图表 ---
        output_path_png = os.path.join(OUTPUT_DIR, f"{config['output_base_name']}.png")
        output_path_tif = os.path.join(OUTPUT_DIR, f"{config['output_base_name']}.tif")
        
        try:
            plt.savefig(output_path_png, dpi=DPI, bbox_inches='tight')
            print(f"图表已成功保存为 PNG 格式: '{output_path_png}'")
            
            plt.savefig(output_path_tif, dpi=DPI, bbox_inches='tight')
            print(f"图表已成功保存为 TIF 格式: '{output_path_tif}'")
        except Exception as e:
            print(f"错误：保存图表时出错。错误信息: {e}")
        
        # 关闭当前图表，防止在循环中重叠
        plt.close(fig)

    print("\n所有混淆矩阵已生成并保存。")

if __name__ == "__main__":
    main()