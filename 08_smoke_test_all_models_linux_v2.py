#!/usr/bin/env python3
"""
08_smoke_test_all_models.py
冒烟测试：每个模型跑 2 条样本，控制台实时输出 + 日志文件保存
"""
import os, sys, json, time, random, subprocess, re, datetime, glob, requests, pandas as pd

# ------------------------- 配置区域 -------------------------
LLAMACPP_SERVER_EXE = "/root/autodl-tmp/llama.cpp/build/bin/llama-server"
MODELS_DIR          = "/root/autodl-tmp/paper2/models"
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
SERVER_HOST, SERVER_PORT = "127.0.0.1", 8080
SERVER_THREADS      = 36
SERVER_CTX_SIZE     = 4096
SERVER_GPU_LAYERS   = 99
API_URL             = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions"
HEALTH_URL          = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
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

# ------------------------- 数据加载 -------------------------
val_df = pd.read_csv(os.path.join(SCRIPT_DIR, 'processed_data', 'val_set.csv'))
test_samples = val_df.sample(2, random_state=42)[['病理描述_删除诊断', 'Fibrosis_Stage_0_4']].to_dict(orient='records')

# ------------------------- 工具函数 -------------------------
def log(msg):
    """统一日志函数"""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def wait_ready(timeout=300):
    """等待服务器就绪（修复版）"""
    start_time = time.time()
    attempts = 0
    while time.time() - start_time < timeout:
        attempts += 1
        try:
            # 1. 检查 /health 状态
            health_r = requests.get(HEALTH_URL, timeout=5)
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

def parse_resp(text: str, labels):
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

def call_llm(sys_prompt, usr_prompt, model_name):
    payload = {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": usr_prompt}
        ],
        "temperature": 0.0
    }
    if "GLM-4.1V-9B-Thinking" in model_name:
        payload["max_tokens"] = 256
    try:
        r = requests.post(API_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"ERROR:{e}"

# ------------------------- 主循环 -------------------------
models = sorted(glob.glob(os.path.join(MODELS_DIR, "*.gguf")))
for mdl_path in models:
    mdl_name = os.path.basename(mdl_path)
    log(f"\n>>> 测试 {mdl_name} ...")
    cmd = [LLAMACPP_SERVER_EXE,
           "--model", mdl_path,
           "--threads", str(SERVER_THREADS),
           "--ctx-size", str(SERVER_CTX_SIZE),
           "--host", SERVER_HOST,
           "--port", str(SERVER_PORT),
           "--n-gpu-layers", str(SERVER_GPU_LAYERS),
          "--flash-attn"
          ]

    # 子进程输出直接打到终端
    proc = subprocess.Popen(cmd)
    if not wait_ready():
        log("  ❌ 服务器启动失败，跳过")
        subprocess.run(["pkill", "-f", "llama-server"]); 
        continue

    risk_flag = False
    for i, row in enumerate(test_samples, 1):
        resp = call_llm(
            "你是一位肝脏病理专家，只输出一个 0-4 的阿拉伯数字。",
            row['病理描述_删除诊断'][:200],
            mdl_name
        )
        label = parse_resp(resp, [0, 1, 2, 3, 4])
        token_len = len(resp.split())
        log(f"  样本{i}: 标签={label} 长度={token_len}")
        log(f"  原始输出:\n{resp}\n")   # 关键：实时显示原始输出
        if token_len > 512 or label is None:
            risk_flag = True
    subprocess.run(["pkill", "-f", "llama-server"])
    if risk_flag:
        log(f"  ⚠️ {mdl_name} 存在潜在无限输出风险")
    else:
        log(f"  ✅ {mdl_name} 通过冒烟测试")

log("\n=== 冒烟测试完成 ===")