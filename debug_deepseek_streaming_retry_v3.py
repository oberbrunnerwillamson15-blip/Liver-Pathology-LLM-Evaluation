# debug_deepseek_streaming_retry_v3.py

import ollama
import pandas as pd
import os
import sys
import json
from datetime import datetime
import time

# --- 1. 日志记录和全局配置 (无变化) ---
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

class Logger(object):
    def __init__(self, log_dir, timestamp):
        self.terminal = sys.stdout
        self.log_file_path = os.path.join(log_dir, f'log_debug_deepseek_v3_{timestamp}.txt')
        self.log = open(self.log_file_path, 'a', encoding='utf-8')
    def write(self, message): self.terminal.write(message); self.log.write(message)
    def flush(self): self.terminal.flush(); self.log.flush()

sys.stdout = Logger('logs', timestamp)
sys.stderr = sys.stdout

print(f"--- DeepSeek 家族终极调试脚本 V3 (新解析逻辑) ---\n")

# --- 全局配置 (无变化) ---
MODELS_TO_DEBUG = ['deepseek-r1:8b', 'deepseek-r1:7b']
FLUSHER_MODEL = 'llama3.2:3b'
TASKS_TO_DEBUG = {
    'fibrosis': {'column_name': 'Fibrosis_Stage_0_4', 'name': '肝纤维化分期', 'labels': [0, 1, 2, 3, 4]},
    'inflammation': {'column_name': 'Inflammation_Grade_0_4', 'name': '肝脏炎症分级', 'labels': [0, 1, 2, 3, 4]},
    'steatosis': {'column_name': 'Steatosis_Grade_1_3', 'name': '肝脂肪变性分级', 'labels': [1, 2, 3]}
}
SAMPLE_COUNT_PER_TASK = 3
STREAM_TIMEOUT_CHARS = 4096
MAX_ATTEMPTS = 3
CACHE_FLUSH_THRESHOLD = 2

# --- 2. Prompt 模板 (无变化) ---
def get_system_prompt(task_name, labels_str):
    return f"""
你是一位顶尖的、经验丰富的肝脏病理学专家。
你的任务是根据提供的肝脏活检病理描述，对“{task_name}”进行精确分级。
你的回答必须严格遵守以下规则：
1.  只输出一个阿拉伯数字，这个数字必须是 {labels_str} 中的一个。
2.  禁止包含任何额外的解释、思考过程、单位、标点或任何其他文字。
3.  你的输出必须可以直接被程序解析为一个整数。
"""
def get_user_prompt(text, examples, task_name, labels_str):
    example_str = "".join([f"[示例{i+1}]\n病理描述：\n\"{ex['text']}\"\n{task_name} ({labels_str}):\n{ex['label']}\n\n" for i, ex in enumerate(examples)])
    return f"""
以下是一些参考示例：
{example_str}
现在，请根据以下新的病理描述进行判断：
病理描述：
"{text}"
"""

# --- 核心修改：全新的解析函数 ---
def parse_final_answer(raw_response, labels):
    """
    (V3 - Final) 最终版解析函数。
    专为处理带有 <think>...</think> 块的模型设计，基于split方案。
    逻辑：
    1. 寻找 </think> 闭合标签。
    2. 如果找到，使用split将字符串分割，并取第二部分作为有效回答。
    3. 如果找不到，则认为整个响应都是有效回答。
    4. 对有效回答部分进行清理(strip)和数字提取。
    """
    think_end_tag = "</think>"
    
    # 步骤1 & 2: 检查并分割字符串
    if think_end_tag in raw_response:
        parts = raw_response.split(think_end_tag)
        # 健壮性检查：确保分割后至少有两部分
        if len(parts) > 1:
            valid_part = parts[1]
        else: # 如果</think>在字符串末尾，split后可能只有一部分
            valid_part = "" 
    else:
        # 步骤3: 如果没有思考块，整个响应都是有效部分
        valid_part = raw_response

    # 步骤4: 清理并提取数字
    cleaned_text = valid_part.strip()
    
    if not cleaned_text: # 如果清理后是空字符串
        return None

    try:
        # 优先假设清理后的文本就是纯数字
        grade = int(cleaned_text)
        if grade in labels:
            return grade
    except (ValueError, TypeError):
        # 如果直接转换失败，说明可能混有其他文字，用正则作为后备
        import re
        numbers = re.findall(r'-?\d+', cleaned_text) # 支持负数以防万一
        if numbers:
            grade = int(numbers[0])
            if grade in labels:
                return grade

    # 如果所有方法都失败了
    return None

# --- 3. 核心鲁棒调用逻辑 (无变化) ---
def flush_ollama_cache(flusher_model):
    print(f"\n[CACHE FLUSH!] 正在调用 {flusher_model} 以刷新Ollama缓存...")
    try:
        ollama.chat(model=flusher_model, messages=[{'role': 'user', 'content': 'reset'}], options={'num_ctx': 256})
        print("[CACHE FLUSH!] 缓存刷新操作完成。\n")
        return True
    except Exception as e:
        print(f"[CACHE FLUSH!] 刷新缓存时发生错误: {e}")
        return False

def get_llm_response_robust(model, system_prompt, user_prompt, labels):
    consecutive_failures = 0
    for attempt in range(MAX_ATTEMPTS):
        print(f"\n--- 正在进行第 {attempt + 1}/{MAX_ATTEMPTS} 次尝试 ---")
        if consecutive_failures >= CACHE_FLUSH_THRESHOLD: flush_ollama_cache(FLUSHER_MODEL); consecutive_failures = 0
        full_response, char_count, is_timeout = "", 0, False
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            stream = ollama.chat(model=model, messages=messages, options={'temperature': 0.1 * (attempt + 1)}, stream=True)
            print("实时接收模型输出: ", end='', flush=True)
            for chunk in stream:
                content_piece = chunk['message']['content']
                print(content_piece, end='', flush=True)
                full_response += content_piece; char_count += len(content_piece)
                if char_count > STREAM_TIMEOUT_CHARS: print(f"\n[长度熔断!]"); is_timeout = True; break
            print("\n--- 流结束 ---")
            if not is_timeout:
                parsed_answer = parse_final_answer(full_response, labels)
                if parsed_answer is not None: print(f"✅ 验证成功！提取到有效答案: {parsed_answer}"); return parsed_answer
                else: print(f"❌ 验证失败。输出: '{full_response[:100]}...'")
        except Exception as e: print(f"!!!!!! 调用Ollama时发生错误: {e} !!!!!!")
        consecutive_failures += 1
        if attempt < MAX_ATTEMPTS - 1: print(f"将在5秒后重试..."); time.sleep(5)
    print("已达到最大重试次数，仍未获得有效答案。"); return -2

# --- 4. 主程序入口 (无变化) ---
if __name__ == "__main__":
    try:
        val_df = pd.read_csv(os.path.join('processed_data', 'val_set.csv'))
        train_df = pd.read_csv(os.path.join('processed_data', 'train_set.csv'))
        print("成功加载数据集。\n")
    except FileNotFoundError: print("错误: 找不到数据集CSV文件。"); exit()
    try: ollama.show(FLUSHER_MODEL)
    except Exception: print(f"错误: 找不到刷新模型 '{FLUSHER_MODEL}'"); exit()
    for model_name in MODELS_TO_DEBUG:
        print(f"\n\n{'='*30}\n开始测试模型: {model_name}\n{'='*30}")
        try: ollama.show(model_name)
        except Exception: print(f"跳过模型 {model_name}，未找到。"); continue
        for task_key, task_config in TASKS_TO_DEBUG.items():
            print(f"\n\n--- 开始测试任务: {task_config['name']} ---")
            col_name, labels, labels_str = task_config['column_name'], task_config['labels'], ", ".join(map(str, task_config['labels']))
            few_shot_examples = []
            for label in sorted(labels):
                example_row = train_df[train_df[col_name] == label]
                if not example_row.empty: few_shot_examples.append({'text': example_row.iloc[0]['病理描述_删除诊断'], 'label': label})
            system_prompt = get_system_prompt(task_config['name'], labels_str)
            for i, row in enumerate(val_df.head(SAMPLE_COUNT_PER_TASK).itertuples()):
                print(f"\n\n--- 处理样本 {i+1}/{SAMPLE_COUNT_PER_TASK} (Patient_ID: {row.Patient_ID}) ---")
                user_prompt = get_user_prompt(row.病理描述_删除诊断, few_shot_examples, task_config['name'], labels_str)
                final_answer = get_llm_response_robust(model_name, system_prompt, user_prompt, labels)
                print(f"\n===> 样本 {row.Patient_ID} 的最终预测结果为: {final_answer} (真实值: {getattr(row, col_name)})")
    print("\n\n--- 所有调试任务完成 ---")