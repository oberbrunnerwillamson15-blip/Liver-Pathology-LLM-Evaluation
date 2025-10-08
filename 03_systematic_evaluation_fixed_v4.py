# 03_systematic_evaluation_fixed_v4.py

import ollama
import pandas as pd
import os
import sys
import json
from datetime import datetime
import time
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --- 1. 日志记录和全局配置 ---
os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

class Logger(object):
    def __init__(self, log_dir, timestamp):
        self.terminal = sys.stdout
        self.log_file_path = os.path.join(log_dir, f'log_systematic_eval_v4_final_{timestamp}.txt')
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

print(f"--- 阶段2: 系统性评测 (V4 - 最终版) ---")
print(f"--- Timestamp: {timestamp} ---")
print(f"日志将保存在: {sys.stdout.get_log_path()}\n")

# --- 全局配置 ---
MODELS_TO_EVALUATE = [
    # 中模型组 (主战场)
    'llama3.1:8b',
    'glm4:9b',
    'qwen3:8b',
    'deepseek-r1:8b', # 加入主战场
    'deepseek-r1:7b',

    # 规模探索 (Qwen家族)
    'qwen3:4b',
    'qwen3:14b',
]

TASKS_TO_EVALUATE = {
    'fibrosis': {'column_name': 'Fibrosis_Stage_0_4', 'name': '肝纤维化分期', 'labels': [0, 1, 2, 3, 4]},
    'inflammation': {'column_name': 'Inflammation_Grade_0_4', 'name': '肝脏炎症分级', 'labels': [0, 1, 2, 3, 4]},
    'steatosis': {'column_name': 'Steatosis_Grade_1_3', 'name': '肝脂肪变性分级', 'labels': [1, 2, 3]}
}

# --- 鲁棒性配置 ---
# 将需要特殊处理的模型名称放入此列表
ROBUST_MODELS = ['deepseek-r1:8b', 'deepseek-r1:7b']
FLUSHER_MODEL = 'llama3.2:3b'
STREAM_TIMEOUT_CHARS = 4096
MAX_ATTEMPTS = 3
CACHE_FLUSH_THRESHOLD = 2

# --- 2. Prompt 模板和解析函数 ---
def get_system_prompt(task_name, labels_str):
    return f"你是一位顶尖的肝脏病理学专家。你的任务是根据病理描述对“{task_name}”进行分级。你的回答必须严格遵守规则：只输出一个属于 {labels_str} 的阿拉伯数字，禁止任何其他文字。"

def get_user_prompt(text, examples, task_name, labels_str):
    example_str = "".join([f"[示例{i+1}]\n病理描述：\n\"{ex['text']}\"\n{task_name} ({labels_str}):\n{ex['label']}\n\n" for i, ex in enumerate(examples)])
    return f"请参考以下示例，并对新的病理描述进行判断。\n\n{example_str}新的病理描述：\n\"{text}\""

def parse_response(response_text, labels):
    if response_text == "LLM_CALL_ERROR": return -1
    think_end_tag = "</think>"
    valid_part = response_text.rsplit(think_end_tag, 1)[-1] if think_end_tag in response_text else response_text
    cleaned_text = valid_part.strip()
    try:
        grade = int(cleaned_text)
        if grade in labels: return grade
        return -3 # 数字有效，但标签无效
    except (ValueError, TypeError):
        numbers = re.findall(r'-?\d+', cleaned_text)
        if numbers:
            grade = int(numbers[0])
            if grade in labels: return grade
            return -3 # 数字有效，但标签无效
        return -2 # 无法解析

# --- 3. 核心调用逻辑 (标准 + 鲁棒) ---
def flush_ollama_cache():
    print(f"\n[CACHE FLUSH!] 正在调用 {FLUSHER_MODEL} 刷新缓存...")
    try:
        ollama.chat(model=FLUSHER_MODEL, messages=[{'role': 'user', 'content': 'Reset.'}], options={'num_ctx': 256})
        print("[CACHE FLUSH!] 刷新完成。")
    except Exception as e:
        print(f"[CACHE FLUSH!] 刷新失败: {e}")

def get_llm_response_standard(model, system_prompt, user_prompt):
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = ollama.chat(model=model, messages=messages, options={'temperature': 0})
        return response['message']['content'].strip()
    except Exception as e:
        print(f"调用Ollama时出错 (模型: {model}): {e}")
        return "LLM_CALL_ERROR"

def get_llm_response_robust(model, system_prompt, user_prompt):
    consecutive_failures = 0
    for attempt in range(MAX_ATTEMPTS):
        if consecutive_failures >= CACHE_FLUSH_THRESHOLD: flush_ollama_cache(); consecutive_failures = 0
        full_response, char_count, is_timeout = "", 0, False
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            stream = ollama.chat(model=model, messages=messages, options={'temperature': 0.1 * attempt}, stream=True)
            for chunk in stream:
                content_piece = chunk['message']['content']
                full_response += content_piece; char_count += len(content_piece)
                if char_count > STREAM_TIMEOUT_CHARS: is_timeout = True; break
            if not is_timeout: return full_response.strip()
        except Exception as e:
            print(f"!!!!!! 调用Ollama时发生错误 (尝试 {attempt+1}): {e} !!!!!!")
        consecutive_failures += 1
        if attempt < MAX_ATTEMPTS - 1: time.sleep(3)
    return "LLM_CALL_ERROR"

# --- 4. 主评估流程 ---
def run_evaluation(val_df, train_df):
    all_results_summary = []
    for model_name in MODELS_TO_EVALUATE:
        for task_key, task_config in TASKS_TO_EVALUATE.items():
            start_time = time.time()
            task_name, col_name, labels = task_config['name'], task_config['column_name'], task_config['labels']
            print(f"\n\n{'='*25} 开始评估 {'='*25}\n模型: {model_name} | 任务: {task_name}")
            try: ollama.show(model_name)
            except Exception as e: print(f"!!! 严重错误: 找不到模型 {model_name}。跳过。错误: {e}"); continue
            
            labels_str = ", ".join(map(str, labels))
            few_shot_examples = [{'text': row['病理描述_删除诊断'], 'label': row[col_name]} for _, row in train_df.groupby(col_name).head(1).iterrows() if row[col_name] in labels]
            system_prompt = get_system_prompt(task_name, labels_str)
            
            predictions, true_labels = [], []
            for i, row in enumerate(val_df.itertuples()):
                text, true_label = row.病理描述_删除诊断, getattr(row, col_name)
                if pd.isna(true_label): continue
                
                user_prompt = get_user_prompt(text, few_shot_examples, task_name, labels_str)
                response_text = get_llm_response_robust(model_name, system_prompt, user_prompt) if model_name in ROBUST_MODELS else get_llm_response_standard(model_name, system_prompt, user_prompt)
                
                predicted_label = parse_response(response_text, labels)
                predictions.append(predicted_label)
                true_labels.append(int(true_label))
                print(f"  > 进度: {i+1}/{len(val_df)} | 真实: {int(true_label)} | 预测: {predicted_label}")
            
            elapsed_time = time.time() - start_time
            report, summary_data = analyze_single_result(model_name, task_config, predictions, true_labels, elapsed_time)
            all_results_summary.append(summary_data)
            
            result_filename = os.path.join('results', f'report_{model_name.replace(":", "_")}_{task_key}_{timestamp}.txt')
            with open(result_filename, 'w', encoding='utf-8') as f: f.write(report)
            print(f"  > 详细报告已保存至: {result_filename}")
    return all_results_summary

def analyze_single_result(model, task_config, predictions, true_labels, elapsed_time):
    task_name, col_name, labels = task_config['name'], task_config['column_name'], task_config['labels']
    report_str = f"--- 模型: {model} | 任务: {task_name} ---\n"
    valid_indices = [i for i, p in enumerate(predictions) if p >= 0]
    valid_preds, valid_trues = [predictions[i] for i in valid_indices], [true_labels[i] for i in valid_indices]
    total, valid = len(predictions), len(valid_preds)
    error_rate = (total - valid) / total if total > 0 else 0
    report_str += f"评估耗时: {elapsed_time:.2f} 秒 (平均: {elapsed_time/total:.2f} 秒/样本)\n"
    report_str += f"总样本数: {total}, 有效解析数: {valid} (成功率: {1-error_rate:.2%})\n"
    summary = {'model': model, 'task_column': col_name, 'accuracy': 0, 'macro_f1': 0, 'error_rate': error_rate, 'time_seconds': elapsed_time}
    if valid > 0:
        accuracy = accuracy_score(valid_trues, valid_preds)
        macro_f1 = f1_score(valid_trues, valid_preds, average='macro', zero_division=0)
        summary.update({'accuracy': accuracy, 'macro_f1': macro_f1})
        report_str += f"准确率 (Accuracy): {accuracy:.4f}\n宏平均 F1-Score (Macro F1): {macro_f1:.4f}\n\n分类报告:\n"
        report_str += classification_report(valid_trues, valid_preds, labels=labels, target_names=[str(l) for l in labels], zero_division=0)
        report_str += "\n混淆矩阵:\n" + str(confusion_matrix(valid_trues, valid_preds, labels=labels)) + "\n"
    print(report_str)
    return report_str, summary

# --- 5. 主程序入口 ---
if __name__ == "__main__":
    try:
        val_df = pd.read_csv(os.path.join('processed_data', 'val_set.csv'))
        train_df = pd.read_csv(os.path.join('processed_data', 'train_set.csv'))
        print(f"成功加载数据集 ({len(val_df)}条验证集, {len(train_df)}条训练集)。\n")
    except FileNotFoundError: print("错误: 找不到数据集CSV文件。"); exit()
    try: ollama.show(FLUSHER_MODEL)
    except Exception: print(f"警告: 找不到刷新模型 '{FLUSHER_MODEL}'。如果评估包含鲁棒模型，可能会失败。");
    
    all_results = run_evaluation(val_df, train_df)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv_path = os.path.join('results', f'summary_results_{timestamp}.csv')
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n\n{'='*25} 全局评估总结 {'='*25}")
        print(summary_df.to_string())
        print(f"\n全局总结报告已保存至: {summary_csv_path}")
    print("\n阶段2系统性评测完成！")