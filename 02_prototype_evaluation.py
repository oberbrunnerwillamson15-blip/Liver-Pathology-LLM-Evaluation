# 02_prototype_evaluation.py

import ollama
import pandas as pd
import os
import sys
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import time

# --- 1. 日志记录和全局配置 ---

# 创建必要的文件夹
os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

class Logger(object):
    def __init__(self, log_dir, timestamp):
        self.terminal = sys.stdout
        self.log_file_path = os.path.join(log_dir, f'log_prototype_eval_{timestamp}.txt')
        self.log = open(self.log_file_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def get_log_path(self):
        return self.log_file_path

sys.stdout = Logger('logs', timestamp)
sys.stderr = sys.stdout

print(f"--- 阶段1: 框架开发与原型验证 ---")
print(f"--- Timestamp: {timestamp} ---")
print(f"日志将保存在: {sys.stdout.get_log_path()}\n")

# --- 全局配置 ---
# 在这里更改模型和任务，即可运行不同实验
MODEL_TO_EVAL = 'llama3.1:8b'
# 任务配置: 'column_name'是数据列名, 'name'是任务名, 'labels'是所有可能的分类
TASK_CONFIG = {
    'fibrosis': {
        'column_name': 'Fibrosis_Stage_0_4',
        'name': '肝纤维化分期',
        'labels': [0, 1, 2, 3, 4]
    },
    # 我们先聚焦纤维化任务，后续可以轻松取消注释来运行其他任务
    # 'inflammation': {
    #     'column_name': 'Inflammation_Grade_0_4',
    #     'name': '肝脏炎症分级',
    #     'labels': [0, 1, 2, 3, 4]
    # },
    # 'steatosis': {
    #     'column_name': 'Steatosis_Grade_1_3',
    #     'name': '肝脂肪变性分级',
    #     'labels': [1, 2, 3]
    # }
}
TARGET_TASK = 'fibrosis' # 先从纤维化开始

# --- 2. Prompt 模板库 ---

class PromptFactory:
    def __init__(self, task_name, task_labels):
        self.task_name = task_name
        self.labels_str = ", ".join(map(str, task_labels))

    def get_zero_shot_prompt(self, text):
        return f"""
角色：你是一位顶尖的肝脏病理学专家。
任务：请仔细阅读以下肝脏活检病理描述，并根据描述，对“{self.task_name}”进行判断。分级标准为 {self.labels_str}。
要求：你的回答必须且只能是一个介于 {self.labels_str} 之间的阿拉伯数字。不要包含任何解释或多余的文字。

病理描述：
"{text}"

{self.task_name} ({self.labels_str}):
"""

    def get_few_shot_prompt(self, text, examples):
        # examples 是一个字典列表，每个字典包含 'text' 和 'label'
        example_str = ""
        for i, ex in enumerate(examples):
            example_str += f"[示例{i+1}]\n"
            example_str += f"病理描述：\n\"{ex['text']}\"\n"
            example_str += f"{self.task_name} ({self.labels_str}):\n{ex['label']}\n\n"

        return f"""
角色：你是一位顶尖的肝脏病理学专家。
任务：请仔细阅读以下肝脏活检病理描述，并根据描述，对“{self.task_name}”进行判断。分级标准为 {self.labels_str}。
要求：你的回答必须且只能是一个介于 {self.labels_str} 之间的阿拉伯数字。不要包含任何解释或多余的文字。
以下是一些示例，请参考它们的格式和逻辑：

{example_str}
现在，请根据以下新的病理描述进行判断：
病理描述：
"{text}"

{self.task_name} ({self.labels_str}):
"""

    def get_cot_prompt(self, text):
        return f"""
角色：你是一位顶尖的肝脏病理学专家。
任务：请仔细阅读以下肝脏活检病理描述，并对“{self.task_name}”进行判断。分级标准为 {self.labels_str}。
要求：请遵循以下两步进行回答：
1.  **思考过程：** 首先，分析病理描述中与“{self.task_name}”相关的关键描述。
2.  **最终分级：** 然后，基于你的分析，给出一个最终分级数字。

请以严格的JSON格式输出你的回答，不要有任何其他多余的文字。
JSON格式如下:
{{
  "reasoning": "此处是你的分析过程...",
  "final_grade": [数字]
}}

病理描述：
"{text}"
"""

# --- 3. 核心评估流程 ---

def get_llm_response(model, prompt_text, is_cot=False):
    """调用LLM并返回响应，处理重试逻辑"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt_text}],
                options={'temperature': 0},
                format='json' if is_cot else '' # 如果是CoT，要求返回JSON
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"调用Ollama时出错 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(5) # 等待5秒后重试
    return None # 所有重试失败后返回None

def parse_response(response_text, is_cot, labels):
    """解析LLM的输出，提取分级数字"""
    if response_text is None:
        return -1 # 表示调用失败
        
    if is_cot:
        try:
            # 移除可能存在于JSON字符串前后的markdown代码块标记
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            data = json.loads(response_text)
            grade = int(data.get('final_grade', -2))
        except (json.JSONDecodeError, TypeError, ValueError):
            print(f"警告: CoT响应解析失败。原始文本: '{response_text[:100]}...'")
            return -2 # 表示解析失败
    else:
        try:
            grade = int(response_text)
        except (ValueError, TypeError):
            print(f"警告: Zero/Few-shot响应解析失败。原始文本: '{response_text[:100]}...'")
            return -2 # 表示解析失败

    if grade not in labels:
        print(f"警告: 返回的级别 '{grade}' 不在有效标签 {labels} 中。")
        return -3 # 表示标签无效
        
    return grade

def run_evaluation(model, task_config, prompt_factory, val_df, train_df):
    """主评估函数"""
    print("验证集列名：", val_df.columns.tolist())
    print("训练集列名：", train_df.columns.tolist())

    task_name = task_config['name']
    col_name = task_config['column_name']
    labels = task_config['labels']

    # 准备Few-shot示例 (每个label一个)
    few_shot_examples = []
    for label in sorted(labels):
        example = train_df[train_df[col_name] == label].iloc[0]
        few_shot_examples.append({
            'text': example['病理描述_删除诊断'],
            'label': label
        })
    print(f"为Few-shot prompt准备了 {len(few_shot_examples)} 个示例。\n")
    
    # 定义要测试的prompt策略
    prompt_strategies = {
        'zero_shot': {'is_cot': False, 'needs_examples': False},
        'few_shot': {'is_cot': False, 'needs_examples': True},
        'cot': {'is_cot': True, 'needs_examples': False},
    }

    results = {}

    for strategy_name, config in prompt_strategies.items():
        print(f"--- 开始评估: {strategy_name.upper()} ---")
        predictions = []
        true_labels = []
        
        # 使用.itertuples()以获得更好的性能
        for row in val_df.itertuples():
            text = row.病理描述_删除诊断
            true_label = getattr(row, col_name)

            if pd.isna(true_label): continue

            # 构建prompt
            if strategy_name == 'zero_shot':
                prompt = prompt_factory.get_zero_shot_prompt(text)
            elif strategy_name == 'few_shot':
                prompt = prompt_factory.get_few_shot_prompt(text, few_shot_examples)
            else: # cot
                prompt = prompt_factory.get_cot_prompt(text)

            # 调用LLM并解析
            response = get_llm_response(model, prompt, is_cot=config['is_cot'])
            predicted_label = parse_response(response, is_cot=config['is_cot'], labels=labels)
            
            predictions.append(predicted_label)
            true_labels.append(int(true_label))

            # 打印进度
            print(f"处理中... Patient_ID: {row.Patient_ID}, 真实: {int(true_label)}, 预测: {predicted_label}")
        
        # 保存该策略的结果
        results[strategy_name] = {
            'predictions': predictions,
            'true_labels': true_labels
        }
        print(f"--- {strategy_name.upper()} 评估完成 ---\n")

    return results

def analyze_and_save_results(model, task_name, results, labels, timestamp):
    """分析结果并保存到文件"""
    print("\n\n" + "="*50)
    print(" " * 20 + "最终评估报告")
    print("="*50)
    print(f"模型: {model}")
    print(f"任务: {task_name}")
    print(f"时间: {timestamp}\n")

    full_report = ""
    summary = {}

    for strategy, data in results.items():
        report_str = f"\n--- 策略: {strategy.upper()} ---\n"
        
        # 清理预测中的错误代码
        valid_indices = [i for i, p in enumerate(data['predictions']) if p >= 0]
        valid_preds = [data['predictions'][i] for i in valid_indices]
        valid_trues = [data['true_labels'][i] for i in valid_indices]
        
        total_count = len(data['predictions'])
        valid_count = len(valid_preds)
        error_rate = (total_count - valid_count) / total_count if total_count > 0 else 0

        report_str += f"总样本数: {total_count}, 有效解析数: {valid_count} (解析成功率: {1-error_rate:.2%})\n"

        if valid_count > 0:
            accuracy = accuracy_score(valid_trues, valid_preds)
            macro_f1 = f1_score(valid_trues, valid_preds, average='macro', zero_division=0)
            
            report_str += f"准确率 (Accuracy): {accuracy:.4f}\n"
            report_str += f"宏平均 F1-Score (Macro F1): {macro_f1:.4f}\n\n"
            report_str += "分类报告 (Classification Report):\n"
            report_str += classification_report(valid_trues, valid_preds, labels=labels, target_names=[str(l) for l in labels], zero_division=0)
            report_str += "\n混淆矩阵 (Confusion Matrix):\n"
            report_str += str(confusion_matrix(valid_trues, valid_preds, labels=labels)) + "\n"
        
            summary[strategy] = {'accuracy': accuracy, 'macro_f1': macro_f1, 'error_rate': error_rate}
        else:
            report_str += "没有有效的预测结果可供分析。\n"
            summary[strategy] = {'accuracy': 0, 'macro_f1': 0, 'error_rate': 1}
            
        print(report_str)
        full_report += report_str

    # 保存详细报告
    result_filename = os.path.join('results', f'report_{model.replace(":", "_")}_{TARGET_TASK}_{timestamp}.txt')
    with open(result_filename, 'w', encoding='utf-8') as f:
        f.write(full_report)
    print(f"\n详细评估报告已保存至: {result_filename}")

    # 打印最终总结
    print("\n--- 性能总结 ---")
    print(f"{'策略':<15}{'准确率':<15}{'Macro F1':<15}{'解析失败率':<15}")
    for strategy, scores in summary.items():
        print(f"{strategy:<15}{scores['accuracy']:.4f}{'':<10}{scores['macro_f1']:.4f}{'':<10}{scores['error_rate']:.2%}")

# --- 4. 主程序入口 ---

if __name__ == "__main__":
    # 加载数据
    try:
        val_df = pd.read_csv(os.path.join('processed_data', 'val_set.csv'))
        train_df = pd.read_csv(os.path.join('processed_data', 'train_set.csv'))

        # 统一重命名列名：把中文逗号替换为英文下划线，并去除多余空格
        def normalize_col(c):
            return c.strip().replace('，', '_')
        
        val_df.columns   = [normalize_col(c) for c in val_df.columns]
        train_df.columns = [normalize_col(c) for c in train_df.columns]

        print(f"成功加载验证集 ({len(val_df)}条) 和训练集 ({len(train_df)}条) 数据。\n")
    except FileNotFoundError:
        print("错误: 找不到'processed_data'中的数据集CSV文件。")
        print("请确保'00_data_preparation.py'已成功运行。")
        exit()

    # 初始化任务和Prompt工厂
    task_info = TASK_CONFIG[TARGET_TASK]
    prompt_factory = PromptFactory(task_info['name'], task_info['labels'])
    
    # 运行评估
    evaluation_results = run_evaluation(MODEL_TO_EVAL, task_info, prompt_factory, val_df, train_df)
    
    # 分析并保存结果
    analyze_and_save_results(MODEL_TO_EVAL, task_info['name'], evaluation_results, task_info['labels'], timestamp)
    
    print("\n阶段1原型验证完成！")