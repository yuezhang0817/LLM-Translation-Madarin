import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
import glob

# 创建output目录（如果不存在）
if not os.path.exists('output'):
    os.makedirs('output')

# 获取data目录下所有Excel文件
excel_files = glob.glob('data/*.xlsx')
print(f'找到 {len(excel_files)} 个Excel文件')

# 定义翻译列
translation_columns = ['Original texts', 'H.Bencraft Joly', 'Deepseek', 'gpt4', 'gpt4o', 'google translation']

# 加载MPNet模型
model = SentenceTransformer('all-mpnet-base-v2')

# 清理文本函数
def clean_text(text):
    # 移除非字母数字字符，但保留空格
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # 转换为小写
    text = text.lower()
    # 移除多余空格
    text = ' '.join(text.split())
    return text

# 分割段落函数
def split_into_paragraphs(text):
    # 按双换行符或单换行符分割段落
    if '\n\n' in text:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    else:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return paragraphs if paragraphs else [text]

# 计算两个文本列之间的余弦相似度（基于MPNet模型和段落级别）
def calculate_similarity(df, col1, col2):
    similarities = []
    
    for i in range(len(df)):
        text1 = str(df[col1].iloc[i]) if pd.notna(df[col1].iloc[i]) else ''
        text2 = str(df[col2].iloc[i]) if pd.notna(df[col2].iloc[i]) else ''
        
        # 如果任一文本为空，则相似度为0
        if not text1.strip() or not text2.strip():
            similarities.append(0.0)
            continue
        
        # 分割成段落
        paragraphs1 = split_into_paragraphs(text1)
        paragraphs2 = split_into_paragraphs(text2)
        
        # 如果段落数量不匹配，使用整体文本比较
        if len(paragraphs1) != len(paragraphs2):
            try:
                # 清理并编码整个文本
                cleaned_text1 = clean_text(text1)
                cleaned_text2 = clean_text(text2)
                
                if not cleaned_text1 or not cleaned_text2:
                    similarities.append(0.0)
                    continue
                    
                embeddings = model.encode([cleaned_text1, cleaned_text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                similarities.append(round(similarity, 4))
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                similarities.append(0.0)
        else:
            # 段落对段落比较
            paragraph_similarities = []
            try:
                for p1, p2 in zip(paragraphs1, paragraphs2):
                    cleaned_p1 = clean_text(p1)
                    cleaned_p2 = clean_text(p2)
                    
                    if not cleaned_p1 or not cleaned_p2:
                        paragraph_similarities.append(0.0)
                        continue
                        
                    embeddings = model.encode([cleaned_p1, cleaned_p2])
                    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                    paragraph_similarities.append(sim)
                
                # 取平均相似度
                avg_similarity = sum(paragraph_similarities) / len(paragraph_similarities) if paragraph_similarities else 0.0
                similarities.append(round(avg_similarity, 4))
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                similarities.append(0.0)
    
    return similarities

# 计算不同翻译版本之间的相似度组合
combinations = [
    ('H.Bencraft Joly', 'google translation', 'Joly-Google'),
    ('Deepseek', 'google translation', 'Deepseek-Google'),
    ('gpt4', 'google translation', 'GPT4-Google'),
    ('gpt4o', 'google translation', 'GPT4o-Google'),
    ('H.Bencraft Joly', 'Deepseek', 'Joly-Deepseek'),
    ('H.Bencraft Joly', 'gpt4', 'Joly-GPT4'),
    ('H.Bencraft Joly', 'gpt4o', 'Joly-GPT4o'),
    ('Deepseek', 'gpt4', 'Deepseek-GPT4'),
    ('Deepseek', 'gpt4o', 'Deepseek-GPT4o'),
    ('gpt4', 'gpt4o', 'GPT4-GPT4o')
]

# 处理每个Excel文件
for file_path in excel_files:
    print(f'正在处理文件: {file_path}')
    
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 打印列名用于调试
    print('Excel文件列名:', list(df.columns))
    
    # 确保所有文本列都是字符串类型并处理缺失值
    for col in translation_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('')
    
    # 计算所有翻译版本与原始文本之间的相似度
    # 使用实际的列名，注意'Chapter '后面有空格
    result_df = df[['Chapter ', 'Verse']].copy()
    # 重命名列以去除空格
    result_df = result_df.rename(columns={'Chapter ': 'Chapter'})
    
    # 添加原始文本和所有翻译列
    for col in translation_columns:
        if col in df.columns:
            result_df[col] = df[col]
    
    # 计算不同翻译版本之间的相似度
    for col1, col2, result_col in combinations:
        if col1 in df.columns and col2 in df.columns:
            result_df[result_col] = calculate_similarity(df, col1, col2)
    
    # 保存结果到Excel文件
    # 生成输出文件名
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_file1 = f'output/{name_without_ext}_similarity_results.xlsx'
    output_file2 = f'output/{name_without_ext}_similarity_results_formatted.xlsx'
    
    result_df.to_excel(output_file1, index=False)
    
    # 创建一个更符合参考格式的表格（只包含相似度数据）
    similarity_cols = [col for _, _, col in combinations if col in result_df.columns]
    
    # 保存一个只包含关键列的版本
    final_df = result_df[['Chapter', 'Verse', 'Original texts', 'H.Bencraft Joly', 'Deepseek', 'gpt4', 'gpt4o', 'google translation'] + similarity_cols].copy()
    final_df.to_excel(output_file2, index=False)
    
    print(f"分析完成！结果已保存到 '{output_file1}' 和 '{output_file2}'")

print("所有文件处理完成！")