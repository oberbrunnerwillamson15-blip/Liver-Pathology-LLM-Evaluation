import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

# --- 1. 加载数据 ---
print("--- 步骤1: 正在加载数据... ---")
try:
    # 确保文件路径正确，我们假设data.json和脚本在同一个文件夹
    json_path = 'data.json' 
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"成功加载 {len(df)} 条数据。")
    print("数据前5行预览:")
    print(df.head())
    print("\n")

except FileNotFoundError:
    print(f"错误: 未在当前目录找到 '{json_path}' 文件。请确保文件名正确且文件存在。")
    exit() # 如果文件找不到，则退出程序
except Exception as e:
    print(f"加载数据时发生未知错误: {e}")
    exit()

# --- 2. 数据划分 (分层抽样) ---
print("--- 步骤2: 正在划分数据集... ---")
# 我们选择 'Fibrosis_Stage_0_4' 作为分层依据，因为它通常是临床上最重要的指标
# 如果该列有缺失值，需要先处理
if df['Fibrosis_Stage_0_4'].isnull().any():
    print("警告: 'Fibrosis_Stage_0_4' 列中存在缺失值，将填充为-1以便分层。")
    df['Fibrosis_Stage_0_4'].fillna(-1, inplace=True)

# 划分测试集 (15%)
train_val_df, test_df = train_test_split(
    df, 
    test_size=0.15, 
    stratify=df['Fibrosis_Stage_0_4'], 
    random_state=42
)

# 从剩余数据中划分训练集和验证集
# 验证集大小约为总数的15% (40/268 ≈ 0.15), 所以在剩余数据(train_val_df)中占比为 40 / (268 * 0.85) ≈ 0.175
val_size_in_remaining = len(test_df) / len(train_val_df)

train_df, val_df = train_test_split(
    train_val_df, 
    test_size=val_size_in_remaining, 
    stratify=train_val_df['Fibrosis_Stage_0_4'], 
    random_state=42
)

print(f"数据集划分完成:")
print(f"训练集 (Training Set)  : {len(train_df)} 条")
print(f"验证集 (Validation Set): {len(val_df)} 条")
print(f"测试集 (Test Set)      : {len(test_df)} 条")
print(f"总计                  : {len(train_df) + len(val_df) + len(test_df)} 条")
print("\n")

# --- 3. 验证分层效果 ---
print("--- 步骤3: 正在验证分层效果 (以Fibrosis_Stage_0_4为例)... ---")
print("原始数据分布:")
print(df['Fibrosis_Stage_0_4'].value_counts(normalize=True).sort_index())
print("\n训练集数据分布:")
print(train_df['Fibrosis_Stage_0_4'].value_counts(normalize=True).sort_index())
print("\n验证集数据分布:")
print(val_df['Fibrosis_Stage_0_4'].value_counts(normalize=True).sort_index())
print("\n测试集数据分布:")
print(test_df['Fibrosis_Stage_0_4'].value_counts(normalize=True).sort_index())
print("\n")

# --- 4. 保存划分好的数据集 ---
print("--- 步骤4: 正在保存数据集到CSV文件... ---")
# 创建一个子目录来存放处理好的数据
output_dir = 'processed_data'
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, 'train_set.csv'), index=False, encoding='utf-8-sig')
val_df.to_csv(os.path.join(output_dir, 'val_set.csv'), index=False, encoding='utf-8-sig')
test_df.to_csv(os.path.join(output_dir, 'test_set.csv'), index=False, encoding='utf-8-sig')

print(f"数据集已保存至 '{output_dir}' 文件夹下。")
print("阶段0-数据准备部分完成！")