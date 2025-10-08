import os
import subprocess
import re
import argparse
import sys

def export_ollama_models(output_dir):
    """将Ollama模型通过硬链接导出到目标目录"""
    # 获取Ollama存储目录（Windows默认路径）
    ollama_dir = r"D:\soft\to_run\ai\chatai\model\OLLAMA\blobs"
    if not os.path.exists(ollama_dir):
        print(f"错误: Ollama目录不存在 - {ollama_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 获取模型列表（显式指定编码）
        list_result = subprocess.check_output(
            ['ollama', 'list'], 
            text=True,
            encoding='utf-8',  # 显式指定UTF-8编码
            stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        print(f"执行ollama list失败: {e.output}")
        return
    
    model_names = []
    for line in list_result.splitlines()[1:]:
        if line.strip():
            model_name = line.split()[0].rstrip(':')
            if model_name != "NAME":
                model_names.append(model_name)
    
    print(f"找到 {len(model_names)} 个模型: {', '.join(model_names)}")
    
    for model_name in model_names:
        try:
            # 关键修复：使用字节流输出+多重解码机制[[20][46]]
            show_result_bytes = subprocess.check_output(
                ['ollama', 'show', '--modelfile', model_name],
                stderr=subprocess.STDOUT
            )
            
            # 尝试多种解码方式[[1][20][46]]
            try:
                show_result = show_result_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    show_result = show_result_bytes.decode('gbk')
                except UnicodeDecodeError:
                    show_result = show_result_bytes.decode('cp437', errors='replace')  # 最终兜底方案
            
            # 提取文件路径
            file_path = None
            for line in show_result.splitlines():
                if line.startswith('FROM'):
                    file_path = line.split(' ', 1)[1].strip()
                    break
            
            if not file_path or not os.path.exists(file_path):
                print(f"跳过 {model_name}: 未找到有效文件路径")
                continue
            
            # 创建安全文件名
            safe_name = re.sub(r'[:/\\*?"<>|]', '_', model_name).replace('_latest', '')
            output_path = os.path.join(output_dir, f'{safe_name}.gguf')
            
            # 创建硬链接
            if not os.path.exists(output_path):
                os.link(file_path, output_path)
                print(f"已创建: {model_name} -> {output_path}")
            else:
                print(f"已存在: {model_name}")

        except subprocess.CalledProcessError as e:
            print(f"处理 {model_name} 失败: {e.output}")
        except OSError as e:
            print(f"创建链接失败({model_name}): {e.strerror}")

if __name__ == "__main__":
    # 设置系统默认编码
    sys.stdout.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(description='导出Ollama模型到指定目录')
    parser.add_argument('output_dir', help='目标输出目录路径')
    args = parser.parse_args()
    
    export_ollama_models(args.output_dir)