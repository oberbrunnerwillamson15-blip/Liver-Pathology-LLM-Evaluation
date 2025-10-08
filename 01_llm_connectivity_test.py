import ollama
import pandas as pd
import os
import sys
from datetime import datetime

# --- 日志记录设置 ---
# 创建logs文件夹（如果不存在）
os.makedirs('logs', exist_ok=True)
# 获取当前时间作为日志文件名的一部分
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'logs/log_connectivity_test_{timestamp}.txt'

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_filename)
sys.stderr = sys.stdout  # 将错误输出也重定向到日志文件

print("--- 阶段0: LLM连接性测试 ---")
print(f"日志将保存在: {log_filename}\n")

# --- 1. 选择一个你已经下载好的中型模型进行测试 ---
# 比如 llama3.1:8b。如果没下载，可以换成其他已有的，例如 glm4:9b
MODEL_TO_TEST = 'llama3.1:8b' 
print(f"--- 步骤1: 准备测试模型 ---")
print(f"将要测试的模型是: {MODEL_TO_TEST}")

try:
    print("正在检查模型是否存在于Ollama中...")
    ollama.show(MODEL_TO_TEST)
    print(f"成功找到模型 '{MODEL_TO_TEST}'。\n")
except Exception as e:
    print(f"\n错误: 未在Ollama中找到模型 '{MODEL_TO_TEST}'。")
    print(f"请确认Ollama服务正在运行，并且您已经通过 'ollama pull {MODEL_TO_TEST}' 下载了该模型。")
    print(f"Ollama返回的原始错误信息: {e}")
    exit()

# --- 2. 加载一条验证集数据作为测试样本 ---
print("--- 步骤2: 准备测试数据 ---")
try:
    val_set_path = os.path.join('processed_data', 'val_set.csv')
    val_df = pd.read_csv(val_set_path)
    sample_case = val_df.iloc[0] # 取验证集的第一条数据
    
    patient_id = sample_case['Patient_ID']
    pathology_text = sample_case['病理描述_删除诊断']
    ground_truth_fibrosis = sample_case['Fibrosis_Stage_0_4']
    
    print(f"成功加载测试样本 (Patient_ID: {patient_id})。")
    print(f"样本的真实纤维化分期 (Ground Truth): {ground_truth_fibrosis}")
    print("样本的病理描述 (前100字符):")
    print(pathology_text[:100] + "...\n")
    
except FileNotFoundError:
    print(f"错误: 找不到验证集文件 '{val_set_path}'。")
    print("请先成功运行 '00_data_preparation.py' 脚本。")
    exit()
except Exception as e:
    print(f"加载测试数据时发生错误: {e}")
    exit()

# --- 3. 构建一个简单的Prompt并发送给LLM ---
print("--- 步骤3: 构建Prompt并调用LLM ---")
prompt = f"""
请你扮演一位专业的肝脏病理医生。
请阅读以下的肝脏病理描述，并判断其纤维化分期（0-4级）。
你的回答只能是一个阿拉伯数字，不要包含任何其他文字。

病理描述:
{pathology_text}

纤维化分期 (0-4):
"""

print("构建的Prompt (为简洁仅显示部分):")
print(prompt[:200] + "...\n")
print("正在向LLM发送请求，请稍候...")

try:
    response = ollama.chat(
        model=MODEL_TO_TEST,
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0} # 使用0温度确保可复现性
    )
    
    llm_output = response['message']['content'].strip()
    
    print("\n--- 步骤4: 分析LLM的返回结果 ---")
    print("LLM成功返回响应！")
    print(f"原始输出: '{llm_output}'")
    
    # --- 4. 检查输出格式 ---
    if llm_output.isdigit() and 0 <= int(llm_output) <= 4:
        print("格式正确: LLM返回了一个0-4之间的数字。")
        print(f"LLM的判断: {llm_output}")
        print(f"真实分期  : {ground_truth_fibrosis}")
        if int(llm_output) == ground_truth_fibrosis:
            print("结论: 判断正确！")
        else:
            print("结论: 判断错误。")
    else:
        print("格式错误: LLM的输出不是一个有效的数字或超出了范围。")

    print("\n🎉 祝贺您！阶段0已全部完成！您的开发环境已准备就绪。")

except Exception as e:
    print("\n错误: 调用LLM时发生严重错误。")
    print("请检查：")
    print("1. Ollama服务是否正在后台运行。")
    print("2. 您的网络连接或防火墙设置是否阻止了本地应用间的通信。")
    print(f"Ollama返回的原始错误信息: {e}")