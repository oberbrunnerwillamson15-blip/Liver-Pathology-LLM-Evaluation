import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# =============================================================================
# 1. 全局设置与数据加载
# =============================================================================

def setup_environment_and_load_data():
    """配置绘图环境并加载、预处理数据"""
    
    # --- 全局绘图风格设置 (符合JMIR等期刊风格) ---
    plt.style.use('default')
    sns.set(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # --- 文件路径 ---
    base_path = "D:/工作/小论文v2_20250801_v2/code"
    perf_path = os.path.join(base_path, "results/07_evaluation_llamacpp_linux_v1.1_output/aggregated_summary.csv")
    meta_path = os.path.join(base_path, "所有模型说明.csv")
    output_path = os.path.join(base_path, "results/09_draw")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # --- 加载数据 ---
    try:
        df_perf = pd.read_csv(perf_path)
        df_meta = pd.read_csv(meta_path)
    except FileNotFoundError as e:
        print(f"错误: 无法找到文件 {e.filename}。请检查路径是否正确。")
        return None, None, None

    # --- 数据预处理 ---
    # 清理性能数据中的模型名称
    df_perf['model_clean'] = df_perf['model'].str.replace('-Q4_K_M.gguf', '', regex=False)
    
    # 处理 gpt-oss-20b 的特殊情况
    is_gpt_oss = df_perf['model_clean'] == 'gpt-oss-20b'
    df_perf.loc[is_gpt_oss, 'model_clean'] = df_perf.loc[is_gpt_oss, 'model_clean'] + ' (' + df_perf.loc[is_gpt_oss, 'reasoning_mode'] + ')'

    # 简化任务名称
    task_mapping = {
        'Fibrosis_Stage_0_4': 'Fibrosis Staging',
        'Inflammation_Grade_0_4': 'Inflammation Grading',
        'Steatosis_Grade_1_3': 'Steatosis Grading'
    }
    df_perf['task'] = df_perf['task_column'].map(task_mapping)

    # 清理元数据中的模型名称以便合并
    df_meta['model_clean'] = df_meta['Model']
    
    # 合并数据
    df_merged = pd.merge(df_perf, df_meta, on='model_clean', how='left')

    # 从模型名称中提取参数量 (B)
    def extract_size(model_name):
        match = re.search(r'(\d+)b', model_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    df_merged['size_b'] = df_merged['model_clean'].apply(extract_size)
    
    return df_merged, output_path

# =============================================================================
# 2. Figure 5: 各子图生成函数
# =============================================================================

def create_panel_a(ax, df):
    """Panel A: 模型规模 vs. 性能"""
    # 筛选数据
    qwen_models = ['Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B']
    gemma_models = ['gemma-3-4b-it', 'gemma-3-12b-it', 'gemma-3-27b-it']
    
    df_qwen = df[(df['model_clean'].isin(qwen_models)) & (df['task'] == 'Fibrosis Staging')].sort_values('size_b')
    df_gemma = df[(df['model_clean'].isin(gemma_models)) & (df['task'] == 'Inflammation Grading')].sort_values('size_b')
    
    # 绘图
    palette = sns.color_palette("viridis", 2)
    ax.plot(df_qwen['size_b'], df_qwen['f1_mean'], marker='o', linestyle='-', color=palette[0], label='Qwen (Fibrosis Staging)')
    ax.plot(df_gemma['size_b'], df_gemma['f1_mean'], marker='s', linestyle='--', color=palette[1], label='Gemma (Inflammation Grading)')
    
    ax.set_xlabel("Model Size (Billion Parameters)")
    ax.set_ylabel("Macro F1-Score")
    ax.legend(title="Model Family (Task)")
    ax.set_ylim(0, 0.8)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

def create_panel_b(ax, df):
    """Panel B: 医疗特化 vs. 泛化模型"""
    models_to_compare = ['gemma-3-4b-it', 'medgemma-4b-it', 'gemma-3-27b-it', 'medgemma-27b-it']
    df_subset = df[df['model_clean'].isin(models_to_compare)].copy()

    def model_type(name):
        if 'medgemma' in name:
            return 'Specialized'
        return 'General'
    
    def model_family(name):
        if '4b' in name:
            return '4B'
        return '27B'

    df_subset['Model Type'] = df_subset['model_clean'].apply(model_type)
    df_subset['Parameter Size'] = df_subset['model_clean'].apply(model_family)
    
    # 使用catplot在指定ax上绘图
    sns.barplot(data=df_subset, x='task', y='f1_mean', hue='Model Type', 
                palette={'General': '#1f77b4', 'Specialized': '#aec7e8'}, 
                hue_order=['General', 'Specialized'], ax=ax)

    # 旋转X轴标签
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_xlabel("Task")
    ax.set_ylabel("Macro F1-Score")
    ax.legend(title="Model Type")
    ax.set_ylim(0, 0.8)

def create_panel_c(ax, df):
    """Panel C: gpt-oss推理深度的影响"""
    models_to_compare = ['gpt-oss-20b (low)', 'gpt-oss-20b (medium)', 'gpt-oss-20b (high)']
    df_subset = df[df['model_clean'].isin(models_to_compare)].copy()
    
    df_subset['Reasoning Depth'] = df_subset['reasoning_mode'].str.capitalize()
    
    sns.barplot(data=df_subset, x='task', y='f1_mean', hue='Reasoning Depth', 
                hue_order=['Low', 'Medium', 'High'], 
                palette=sns.color_palette("plasma", 3), ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_xlabel("Task")
    ax.set_ylabel("Macro F1-Score")
    ax.legend(title="Reasoning Depth")
    ax.set_ylim(0, 0.8)

def create_panel_d(ax, df):
    """Panel D: 思维链的影响"""
    models_to_compare = ['Qwen3-4B-Instruct-2507', 'Qwen3-4B-Thinking-2507']
    df_subset = df[df['model_clean'].isin(models_to_compare)].copy()
    
    df_subset['Mode'] = df_subset['model_clean'].apply(lambda x: 'Thinking (CoT)' if 'Thinking' in x else 'Instruct')
    
    sns.barplot(data=df_subset, x='task', y='f1_mean', hue='Mode', 
                palette={'Instruct': '#2ca02c', 'Thinking (CoT)': '#98df8a'}, 
                hue_order=['Instruct', 'Thinking (CoT)'], ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_xlabel("Task")
    ax.set_ylabel("Macro F1-Score")
    ax.legend(title="Reasoning Mode")
    ax.set_ylim(0, 0.8)

# =============================================================================
# 3. Figure 6: 知识蒸馏影响
# =============================================================================

def create_figure_6(df, output_path):
    """创建并保存Figure 6"""
    model_pairs = {
        '8B': ('Qwen3-8B', 'DeepSeek-R1-0528-Qwen3-8B'),
        '14B': ('Qwen3-14B', 'DeepSeek-R1-Distill-Qwen-14B'),
        '32B': ('Qwen3-32B', 'DeepSeek-R1-Distill-Qwen-32B')
    }
    
    models_to_plot = [model for pair in model_pairs.values() for model in pair]
    df_subset = df[df['model_clean'].isin(models_to_plot)].copy()

    def get_model_type(name):
        return 'Distilled (DeepSeek)' if 'DeepSeek' in name else 'Base (Qwen)'
    
    def get_model_pair(name):
        for size, (base, distill) in model_pairs.items():
            if name in [base, distill]:
                return f"{size} Parameters"
        return None

    df_subset['Model Type'] = df_subset['model_clean'].apply(get_model_type)
    df_subset['Model Pair'] = df_subset['model_clean'].apply(get_model_pair)
    
    g = sns.catplot(
        data=df_subset,
        kind='bar',
        x='task',
        y='f1_mean',
        hue='Model Type',
        col='Model Pair',
        col_order=['8B Parameters', '14B Parameters', '32B Parameters'],
        hue_order=['Base (Qwen)', 'Distilled (DeepSeek)'],
        palette={'Base (Qwen)': '#ff7f0e', 'Distilled (DeepSeek)': '#ffbb78'},
        height=5,
        aspect=0.9
    )
    
    g.set_axis_labels("Task", "Macro F1-Score")
    g.set_titles("{col_name}")
    g.despine(left=True)
    g.set(ylim=(0, 0.8))
    g.set_xticklabels(rotation=15, ha="right")
    g.legend.set_title("Model Type")

    # 保存文件
    fig6_path_png = os.path.join(output_path, "Figure_6.png")
    fig6_path_tif = os.path.join(output_path, "Figure_6.tif")
    g.savefig(fig6_path_png)
    g.savefig(fig6_path_tif, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    
    print(f"Figure 6 已保存到:\n  {fig6_path_png}\n  {fig6_path_tif}")
    plt.close(g.fig)

# =============================================================================
# 4. 主程序：组合并保存所有图表
# =============================================================================

def main():
    """主执行函数"""
    df_merged, output_path = setup_environment_and_load_data()
    
    if df_merged is None:
        return

    # --- 生成 Figure 5 ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # 绘制四个子图
    create_panel_a(axs[0, 0], df_merged)
    create_panel_b(axs[0, 1], df_merged)
    create_panel_c(axs[1, 0], df_merged)
    create_panel_d(axs[1, 1], df_merged)

    # 添加 A, B, C, D 标签
    panel_labels = ['A', 'B', 'C', 'D']
    for ax, label in zip(axs.flatten(), panel_labels):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes, 
                fontsize=20, fontweight='bold', va='top', ha='right')

    # 调整布局
    plt.tight_layout(pad=3.0)
    
    # --- 保存 Figure 5 的独立和组合文件 ---
    # 保存组合图
    fig5_comp_path_png = os.path.join(output_path, "Figure_5_Composite.png")
    fig5_comp_path_tif = os.path.join(output_path, "Figure_5_Composite.tif")
    fig.savefig(fig5_comp_path_png)
    fig.savefig(fig5_comp_path_tif, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    print(f"Figure 5 (Composite) 已保存到:\n  {fig5_comp_path_png}\n  {fig5_comp_path_tif}")

    # 独立保存每个子图
    panel_funcs = [create_panel_a, create_panel_b, create_panel_c, create_panel_d]
    for i, (func, label) in enumerate(zip(panel_funcs, panel_labels)):
        fig_panel, ax_panel = plt.subplots(figsize=(7, 6))
        func(ax_panel, df_merged)
        panel_path_png = os.path.join(output_path, f"Figure_5{label}.png")
        panel_path_tif = os.path.join(output_path, f"Figure_5{label}.tif")
        fig_panel.savefig(panel_path_png)
        fig_panel.savefig(panel_path_tif, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
        print(f"  - Panel {label} 已保存。")
        plt.close(fig_panel)
    
    plt.close(fig)

    # --- 生成 Figure 6 ---
    create_figure_6(df_merged, output_path)

if __name__ == "__main__":
    main()