import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np

def plot_confusion_matrix(csv_path, task_name, model_name):
    """
    读取预测结果CSV文件，计算并绘制混淆矩阵。
    """
    if not os.path.exists(csv_path):
        print(f"错误: 文件未找到 - {csv_path}")
        return

    # 1. 加载数据并预处理
    df = pd.read_csv(csv_path)
    df.dropna(subset=['True_Label', 'Predicted_Label'], inplace=True) # 移除无法解析的预测
    df['True_Label'] = df['True_Label'].astype(int)
    df['Predicted_Label'] = df['Predicted_Label'].astype(int)
    
    y_true = df['True_Label']
    y_pred = df['Predicted_Label']

    # 2. 获取所有可能的标签
    labels = sorted(list(set(y_true) | set(y_pred)))

    # 3. 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 打印分类报告
    print(f"--- 分类报告 for {model_name} on {task_name} ---")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("-" * 50)

    # 4. 使用Seaborn和Matplotlib进行可视化
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 16}) # 增大字体

    plt.title(f'Confusion Matrix: {model_name}\nTask: {task_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # 创建保存图片的文件夹
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图像
    plot_filename = f"CM_{model_name}_{task_name}.png"
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
    print(f"混淆矩阵图已保存至: {os.path.join(output_dir, plot_filename)}\n")
    plt.close() # 关闭图像，防止在Jupyter中重复显示

# --- 主程序 ---
if __name__ == '__main__':
    # 在这里指定你要分析的CSV文件路径、任务名和模型名
    # 这是一个示例，请根据你实际生成的文件名进行修改
    files_to_analyze = [
    {
        "model_name": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M",
        "task_name": "fibrosis",
        "path": "D:\\工作\\小论文v2_20250801_v2\\code\\results\\detailed_predictions\\preds_DeepSeek-R1-Distill-Qwen-32B-Q4_K_M_task_fibrosis_run_1.csv"
    },
    {
        "model_name": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M",
        "task_name": "inflammation",
        "path": "D:\\工作\\小论文v2_20250801_v2\\code\\results\\detailed_predictions\\preds_DeepSeek-R1-Distill-Qwen-32B-Q4_K_M_task_inflammation_run_1.csv"
    },
    {
        "model_name": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M",
        "task_name": "steatosis",
        "path": "D:\\工作\\小论文v2_20250801_v2\\code\\results\\detailed_predictions\\preds_DeepSeek-R1-Distill-Qwen-32B-Q4_K_M_task_steatosis_run_1.csv"
    },
    {
        "model_name": "gemma-3-4b-it-Q4_K_M",
        "task_name": "fibrosis",
        "path": "D:\\工作\\小论文v2_20250801_v2\\code\\results\\detailed_predictions\\preds_gemma-3-4b-it-Q4_K_M_task_fibrosis_run_1.csv"
    },
    {
        "model_name": "gemma-3-4b-it-Q4_K_M",
        "task_name": "inflammation",
        "path": "D:\\工作\\小论文v2_20250801_v2\\code\\results\\detailed_predictions\\preds_gemma-3-4b-it-Q4_K_M_task_inflammation_run_1.csv"
    },
    {
        "model_name": "gemma-3-4b-it-Q4_K_M",
        "task_name": "steatosis",
        "path": "D:\\工作\\小论文v2_20250801_v2\\code\\results\\detailed_predictions\\preds_gemma-3-4b-it-Q4_K_M_task_steatosis_run_1.csv"
    },
    {
        "model_name": "gpt-oss-20b-Q4_K_M",
        "task_name": "fibrosis",
        "path": "D:\\工作\\小论文v2_20250801_v2\\code\\results\\detailed_predictions\\preds_gpt-oss-20b-Q4_K_M_task_fibrosis_run_1.csv"
    },
    {
        "model_name": "gpt-oss-20b-Q4_K_M",
        "task_name": "inflammation",
        "path": "D:\\工作\\小论文v2_20250801_v2\\code\\results\\detailed_predictions\\preds_gpt-oss-20b-Q4_K_M_task_inflammation_run_1.csv"
    },
    {
        "model_name": "gpt-oss-20b-Q4_K_M",
        "task_name": "steatosis",
        "path": "D:\\工作\\小论文v2_20250801_v2\\code\\results\\detailed_predictions\\preds_gpt-oss-20b-Q4_K_M_task_steatosis_run_1.csv"
    },
    {
        "model_name": "Qwen3-14B-Q4_K_M",
        "task_name": "fibrosis",
        "path": "D:\\工作\\小论文v2_20250801_v2\\code\\results\\detailed_predictions\\preds_Qwen3-14B-Q4_K_M_task_fibrosis_run_1.csv"
    },
    {
        "model_name": "Qwen3-14B-Q4_K_M",
        "task_name": "inflammation",
        "path": "D:\\工作\\小论文v2_20250801_v2\\code\\results\\detailed_predictions\\preds_Qwen3-14B-Q4_K_M_task_inflammation_run_1.csv"
    },
    {
        "model_name": "Qwen3-14B-Q4_K_M",
        "task_name": "steatosis",
        "path": "D:\\工作\\小论文v2_20250801_v2\\code\\results\\detailed_predictions\\preds_Qwen3-14B-Q4_K_M_task_steatosis_run_1.csv"
    }
]
    
    for file_info in files_to_analyze:
        plot_confusion_matrix(file_info["path"], file_info["task_name"], file_info["model_name"])