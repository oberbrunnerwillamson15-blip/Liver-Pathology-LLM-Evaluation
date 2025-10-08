# 07_final_evaluation_llamacpp.py

import os
import sys
import subprocess
import time
import requests
import json
import pandas as pd
import re
import random
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import datetime

# ==============================================================================
# --- 1. 用户配置区域 (请务必检查并修改此部分) ---
# ==============================================================================
# llama.cpp server.exe 的完整路径
LLAMACPP_SERVER_EXE = "/root/autodl-tmp/llama.cpp/build/bin/llama-server"

# 存放所有 .gguf 模型文件的目录
MODELS_DIR = "/root/autodl-tmp/paper2/models"

# 脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 要评估的模型文件名列表 (您可以注释掉不想测试的模型)
MODELS_TO_EVALUATE = [
    "Qwen3-14B-Q4_K_M.gguf",
    "gemma-3-4b-it-Q4_K_M.gguf",
    "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
]

# 每个模型的重复测试次数
NUM_RUNS = 1

# llama.cpp 服务器参数
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_THREADS = 36
SERVER_CTX_SIZE = 8192 # 8k 上下文
SERVER_GPU_LAYERS = 99 # 尽可能多地使用GPU

LLAMACPP_API_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions"
LLAMACPP_HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
MODELS_URL          = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/models"
os.makedirs('logs', exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join('logs', f'smoke_test_{ts}.log')

# ------------------------- Tee 日志类 -------------------------
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

Tee(log_path)

# ==============================================================================
# --- 2. 日志与任务配置 ---
# ==============================================================================
os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'log_final_llamacpp_eval_{timestamp}.txt'

class Logger(object):
    def __init__(self, filename): self.terminal = sys.stdout; self.log = open(filename, 'a', encoding='utf-8')
    def write(self, message): self.terminal.write(message); self.log.write(message)
    def flush(self): self.terminal.flush(); self.log.flush()

sys.stdout = Logger(os.path.join('logs', log_filename))
sys.stderr = sys.stdout
print(f"--- 终极评估流水线 (llama.cpp) ---\nTimestamp: {timestamp}\n")

TASKS_TO_EVALUATE = {
    'fibrosis': {'column_name': 'Fibrosis_Stage_0_4', 'name': '肝纤维化分期', 'labels': [0, 1, 2, 3, 4]},
    'inflammation': {'column_name': 'Inflammation_Grade_0_4', 'name': '肝脏炎症分级', 'labels': [0, 1, 2, 3, 4]},
    'steatosis': {'column_name': 'Steatosis_Grade_1_3', 'name': '肝脂肪变性分级', 'labels': [1, 2, 3]}
}

# ==============================================================================
# --- 3. 核心功能函数 ---
# ==============================================================================
def get_system_prompt(task_name, labels_str): return f"你是一位顶尖的肝脏病理学专家。你的任务是根据病理描述对“{task_name}”进行分级。你的回答必须严格遵守规则：只输出一个属于 {labels_str} 的阿拉伯数字，禁止任何其他文字。"
def get_user_prompt(text, examples, task_name, labels_str):
    example_str = "".join([f"[示例{i+1}]\n病理描述：\n\"{ex['text']}\"\n{task_name} ({labels_str}):\n{ex['label']}\n\n" for i, ex in enumerate(examples)])
    return f"请参考以下示例，并对新的病理描述进行判断。\n\n{example_str}新的病理描述：\n\"{text}\""

# def parse_response(response_text, labels):
#     if response_text == "LLM_CALL_ERROR":
#         return -1
#     # 去掉所有 <think>...</think> 标签
#     cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
#     numbers = re.findall(r'\b([0-4])\b', cleaned)   # 只抓 0-4 的独立数字
#     if numbers:
#         grade = int(numbers[-1])
#         return grade if grade in labels else -3
#     return -2

def parse_response(text: str, labels):
    """
    统一解析 0-4 分期的最终数字
    """
    # 1️⃣ 优先匹配 <answer>...</answer>
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if m:
        cleaned = m.group(1).strip()
    else:
        # 2️⃣ 去掉 <think>...</think> 后剩余内容
        cleaned = re.sub(r'<think>.*?</think>', '', text,
                         flags=re.DOTALL | re.IGNORECASE).strip()
    
    # 3️⃣ 抽取独立数字 0-4
    nums = re.findall(r'\b([0-4])\b', cleaned)
    return int(nums[-1]) if nums and int(nums[-1]) in labels else None

def get_llm_response_llamacpp(system_prompt, user_prompt, model_filename,
                              max_retries: int = 3,
                              base_timeout: int = 60):
    """
    带指数退避 + 超时递增的 llama.cpp API 调用
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "temperature": 0.0
    }

    for attempt in range(1, max_retries + 1):
        # 超时时间线性递增：60 → 120 → 180
        timeout_now = base_timeout + (attempt - 1) * 60
        try:
            response = requests.post(
                LLAMACPP_API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout_now
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()

        except requests.exceptions.ReadTimeout as e:
            print(f"[Retry {attempt}/{max_retries}] ReadTimeout ({e}) → "
                  f"timeout={timeout_now}s, back-off {2 ** attempt}s")
            if attempt == max_retries:
                print("最终失败，标记为 LLM_CALL_ERROR")
                return "LLM_CALL_ERROR"
            time.sleep(2 ** attempt + random.uniform(0, 1))   # 指数退避 + 抖动

        except requests.exceptions.RequestException as e:
            # 其它网络错误也重试
            print(f"[Retry {attempt}/{max_retries}] {e}")
            if attempt == max_retries:
                return "LLM_CALL_ERROR"
            time.sleep(2 ** attempt + random.uniform(0, 1))

# def wait_for_server_ready(timeout=60):
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         try:
#             response = requests.get(LLAMACPP_HEALTH_URL, timeout=1)
#             if response.status_code == 200 and response.json().get("status") == "ok":
#                 print("\n服务器已就绪！")
#                 return True
#         except requests.exceptions.RequestException: pass
#         time.sleep(1)
#     print("\n错误：等待服务器超时！")
#     return False

def log(msg):
    """统一日志函数"""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def wait_for_server_ready(timeout=300):
    """等待服务器就绪（修复版）"""
    start_time = time.time()
    attempts = 0
    while time.time() - start_time < timeout:
        attempts += 1
        try:
            # 1. 检查 /health 状态
            health_r = requests.get(LLAMACPP_HEALTH_URL, timeout=5)
            if health_r.status_code == 200:
                health_data = health_r.json()
                if health_data.get("status") != "ok":
                    log(f"🟡 尝试#{attempts} /health 未就绪: {health_data}")
                    time.sleep(3)
                    continue
            else:
                log(f"🟡 尝试#{attempts} /health 状态码: {health_r.status_code}")
                time.sleep(3)
                continue
                
            # 2. 检查 /v1/models 状态
            models_r = requests.get(MODELS_URL, timeout=5)
            if models_r.status_code == 200:
                models_data = models_r.json()
                if models_data.get("object") == "list" and len(models_data.get("data", [])) > 0:
                    model_id = models_data["data"][0]["id"]
                    log(f"🟢 模型加载完成: {model_id}")
                    return True
            log(f"🟡 尝试#{attempts} /models 未就绪: {models_r.status_code} {models_r.text[:100]}")
            
        except requests.exceptions.ConnectionError:
            log(f"🟡 尝试#{attempts} 连接拒绝，等待启动...")
        except Exception as e:
            log(f"⚠️ 尝试#{attempts} 检查错误: {str(e)}")
        
        time.sleep(5)
    log(f"❌ 等待超时({timeout}秒)")
    return False

# ==============================================================================
# --- 4. 主执行逻辑 ---
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(LLAMACPP_SERVER_EXE):
        print(f"致命错误：找不到 llama-server，路径不正确: {LLAMACPP_SERVER_EXE}"); exit()
    if not os.path.exists(MODELS_DIR):
        print(f"致命错误：找不到模型目录，路径不正确: {MODELS_DIR}"); exit()
        
    val_df   = pd.read_csv(os.path.join(SCRIPT_DIR, 'processed_data', 'val_set.csv'))
    train_df = pd.read_csv(os.path.join(SCRIPT_DIR, 'processed_data', 'train_set.csv'))
    print(f"成功加载数据集。\n")
    
    all_runs_data = []

    for run_num in range(1, NUM_RUNS + 1):
        for model_filename in MODELS_TO_EVALUATE:
            model_path = os.path.join(MODELS_DIR, model_filename)
            if not os.path.exists(model_path):
                print(f"警告：找不到模型文件 {model_path}，跳过此模型。")
                continue

            print(f"\n\n{'='*70}\n===== 开始第 {run_num}/{NUM_RUNS} 轮测试 | 模型: {model_filename} =====\n{'='*70}")
            
            # --- 自动启动服务器 ---
            server_command = [
                LLAMACPP_SERVER_EXE,
                "--model", model_path,
                "--threads", str(SERVER_THREADS),
                "--ctx-size", str(SERVER_CTX_SIZE),
                "--host", SERVER_HOST,
                "--port", str(SERVER_PORT),
                "--n-gpu-layers", str(SERVER_GPU_LAYERS),
                "--flash-attn"
            ]
            server_proc = None
            try:
                print("正在启动 llama.cpp 服务器...")
                server_proc = subprocess.Popen(server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # if not wait_for_server_ready(): continue # 如果服务器启动失败，跳到下一个模型
                if not wait_for_server_ready():
                    log("  ❌ 服务器启动失败，跳过")
                    subprocess.run(["pkill", "-f", "llama-server"]); 
                    continue
            # 显示cmd报错内容
            # try:
            #     print("正在启动 llama.cpp 服务器...")
            #     # 核心修改：移除stdout和stderr的PIPE，让服务器信息直接打印到控制台
            #     server_proc = subprocess.Popen(server_command) 
                
            #     if not wait_for_server_ready(): 
            #         # 如果等待失败，我们手动终止可能卡住的进程
            #         if server_proc:
            #             server_proc.terminate()
            #         continue

                # --- 服务器就绪，开始对所有任务进行评估 ---
                for task_key, task_config in TASKS_TO_EVALUATE.items():
                    task_start_time = time.time()
                    task_name, col_name, labels = task_config['name'], task_config['column_name'], task_config['labels']
                    print(f"\n\n--- 开始任务: {task_name} ---")
                    
                    labels_str = ", ".join(map(str, labels))
                    few_shot_examples = [{'text': r['病理描述_删除诊断'], 'label': r[col_name]} for _, r in train_df.groupby(col_name).head(1).iterrows() if r[col_name] in labels]
                    system_prompt = get_system_prompt(task_name, labels_str)
                    
                    predictions, true_labels, sample_times = [], [], []
                    for i, row in enumerate(val_df.itertuples()):
                        text, true_label = row.病理描述_删除诊断, getattr(row, col_name)
                        if pd.isna(true_label): continue
                        
                        sample_start_time = time.time()
                        user_prompt = get_user_prompt(text, few_shot_examples, task_name, labels_str)
                        response_text = get_llm_response_llamacpp(system_prompt, user_prompt, model_filename)
                        predicted_label = parse_response(response_text, labels)
                        sample_end_time = time.time()
                        
                        sample_duration = sample_end_time - sample_start_time
                        predictions.append(predicted_label); true_labels.append(int(true_label)); sample_times.append(sample_duration)
                        print(f"  > 进度 {i+1}/{len(val_df)} | 真实: {int(true_label)} | 预测: {predicted_label} | 耗时: {sample_duration:.2f}s")

                    # --- 分析单个任务结果 ---
                    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
                    if len(valid_indices) > 0:
                        accuracy = accuracy_score([true_labels[i] for i in valid_indices], [predictions[i] for i in valid_indices])
                        macro_f1 = f1_score([true_labels[i] for i in valid_indices], [predictions[i] for i in valid_indices], average='macro', zero_division=0)
                    else:
                        accuracy, macro_f1 = 0, 0
                    
                    all_runs_data.append({
                        "run": run_num, "model": model_filename, "task_column": col_name,
                        "accuracy": accuracy, "macro_f1": macro_f1,
                        "error_rate": (len(predictions) - len(valid_indices)) / len(predictions),
                        "avg_time_per_sample": sum(sample_times) / len(sample_times) if sample_times else 0
                    })
                    # 保存详细预测结果
                    # 创建一个专门存放详细结果的文件夹
                    predictions_dir = os.path.join('results', 'detailed_predictions')
                    os.makedirs(predictions_dir, exist_ok=True)

                    # 准备要保存的数据
                    detailed_results = []
                    for i in range(len(val_df)):
                        # 从原始DataFrame获取Patient_ID和病理描述
                        patient_id = val_df.iloc[i]['Patient_ID']
                        pathology_text = val_df.iloc[i]['病理描述_删除诊断']
                        
                        # 获取该样本的真实标签和预测结果
                        true_label_for_sample = true_labels[i] if i < len(true_labels) else None
                        predicted_label_for_sample = predictions[i] if i < len(predictions) else None
                        
                        detailed_results.append({
                            'Patient_ID': patient_id,
                            'True_Label': true_label_for_sample,
                            'Predicted_Label': predicted_label_for_sample,
                            'Task': col_name,
                            'Pathology_Text': pathology_text
                        })

                    # 将详细结果保存为CSV
                    model_name_for_file = os.path.splitext(model_filename)[0]
                    output_filename = f"preds_{model_name_for_file}_task_{task_key}_run_{run_num}.csv"
                    output_path = os.path.join(predictions_dir, output_filename)

                    pd.DataFrame(detailed_results).to_csv(output_path, index=False, encoding='utf-8-sig')
                    print(f"  -> 详细预测结果已保存至: {output_path}")
                    
            finally:
                # --- 自动关闭服务器 ---
                if server_proc:
                    print("\n正在关闭 llama.cpp 服务器...")
                    server_proc.terminate()
                    try:
                        server_proc.wait(timeout=30)        # 给 30 秒
                    except subprocess.TimeoutExpired:
                        print("服务器未在 30 秒内退出，强制 kill")
                        subprocess.run(["pkill", "-f", "llama-server"]); 
                    print("服务器已关闭。")

    # --- 5. 保存并分析最终结果 ---
    print("\n\n" + "="*70)
    print("===== 所有评估轮次完成！开始生成最终报告... =====")
    print("="*70)

    if all_runs_data:
        # 保存包含所有轮次细节的原始数据
        raw_df = pd.DataFrame(all_runs_data)
        raw_csv_path = os.path.join('results', f'raw_results_{timestamp}.csv')
        raw_df.to_csv(raw_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n原始数据已保存至: {raw_csv_path}")

        # 计算聚合统计数据 (均值和标准差)
        aggregated_df = raw_df.groupby(['model', 'task_column']).agg(
            f1_mean=('macro_f1', 'mean'),
            f1_std=('macro_f1', 'std'),
            acc_mean=('accuracy', 'mean'),
            acc_std=('accuracy', 'std'),
            time_mean=('avg_time_per_sample', 'mean')
        ).reset_index()

        # 保存聚合后的摘要数据
        agg_csv_path = os.path.join('results', f'aggregated_summary_{timestamp}.csv')
        aggregated_df.to_csv(agg_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n聚合统计报告已保存至: {agg_csv_path}\n")

        print("--- 最终聚合结果 ---")
        print(aggregated_df.to_string())
    else:
        print("没有生成任何结果。")