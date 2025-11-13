import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer, util
import torch
from matplotlib import font_manager
import networkx as nx
from collections import Counter
import spacy
import nltk
from nltk.corpus import stopwords
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ==================== 全局设置 ====================
# 设置全局主题和样式
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Category label definitions
LABELS = ['Optimistic', 'Thankful', 'Empathetic', 'Pessimistic', 'Anxious', 'Sad', 'Annoyed', 'Denial', 'Joking']
POSITIVE_LABELS = ['Optimistic', 'Thankful', 'Joking']
NEUTRAL_LABELS = ['Empathetic']
NEGATIVE_LABELS = ['Pessimistic', 'Anxious', 'Sad', 'Annoyed', 'Denial']
LABEL_DISPLAY_MAP = {'Joking': 'Humour'}
DISPLAY_LABELS = [LABEL_DISPLAY_MAP.get(label, label) for label in LABELS]

# 定义现代化配色方案
COLOR_PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'success': '#52B788',
    'warning': '#F77F00',
    'danger': '#D62828',
    'info': '#3F88C5',
    'light': '#F1FAEE',
    'dark': '#1D3557'
}

# 情感标签配色
# 极性分析配置
POLARITY_CATEGORIES = ['Positive', 'Neutral', 'Negative']
POLARITY_COLORS = ['#52B788', '#74C69D', '#D62828']  # 绿色(积极), 蓝绿色(中性), 红色(消极)

SENTIMENT_COLORS = {
    'Optimistic': '#52B788',
    'Thankful': '#95D5B2',
    'Empathetic': '#74C69D',
    'Pessimistic': '#D62828',
    'Anxious': '#F77F00',
    'Sad': '#EE6C4D',
    'Annoyed': '#C73E1D',
    'Denial': '#E76F51',
    'Joking': '#3F88C5'
}

# ========================
# 1. 数据加载和预处理
# ========================
def load_data(folder_path, translation_type):
    """Load all text from CSV files in folder, return sentence list and full text"""
    sentences = []
    full_texts = []
    split_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?]$)'
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} for {translation_type} does not exist")
        return sentences, ""
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, header=None, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception:
                    continue
            
            if df is None:
                print(f"Warning: Could not read {filename} with any encoding, skipping...")
                continue
            
            try:
                for index, row in df.iterrows():
                    for col in df.columns:
                        cell_value = row[col]
                        if pd.notna(cell_value):
                            text = str(cell_value).strip()
                            if text:
                                full_texts.append(text)
                                parts = re.split(split_pattern, text)
                                for part in parts:
                                    stripped = part.strip()
                                    if stripped and len(stripped) > 5:
                                        sentences.append(stripped)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    full_text = " ".join(full_texts)
    return sentences, full_text

# ========================
# 2. 情感分析模块
# ========================
def download_and_cache_model(model_name, cache_dir=None):
    """Automatically download and cache model"""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    
    print(f"Downloading/loading model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            force_download=False,
            resume_download=True
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            force_download=False,
            resume_download=True,
            num_labels=len(LABELS)
        )
        
        print(f"Model {model_name} loaded successfully!")
        return tokenizer, model
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Trying fallback model...")
        
        fallback_models = [
            "bert-base-multilingual-uncased",
            "xlm-roberta-base",
            "distilbert-base-multilingual-cased"
        ]
        
        for fallback_model in fallback_models:
            try:
                print(f"Trying to load fallback model: {fallback_model}")
                tokenizer = AutoTokenizer.from_pretrained(
                    fallback_model,
                    cache_dir=cache_dir
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    fallback_model,
                    cache_dir=cache_dir,
                    num_labels=len(LABELS)
                )
                print(f"Fallback model {fallback_model} loaded successfully!")
                return tokenizer, model
            except:
                continue
        
        raise Exception("Unable to load any model, please check network connection")

def load_training_data(training_folders):
    """Load all training set data (maintain multi-label format)"""
    all_texts = []
    all_labels = []
    
    label_mapping = {'Joking': 'Humour', 'Humour': 'Joking'}
    
    for folder in training_folders:
        for filename in os.listdir(folder):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder, filename)
                encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'iso-8859-1']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        print(f"Successfully read {filename} with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error reading {filename} with {encoding}: {e}")
                        continue
                
                if df is None:
                    print(f"Failed to read {filename} with any encoding, skipping...")
                    continue
                
                try:
                    texts = df['Tweet'].tolist()
                    available_labels = []
                    for label in LABELS:
                        if label in df.columns:
                            available_labels.append(label)
                        elif label in label_mapping and label_mapping[label] in df.columns:
                            available_labels.append(label_mapping[label])
                        else:
                            print(f"Warning: Label '{label}' not found in {filename}")
                    
                    if len(available_labels) == len(LABELS):
                        label_data = []
                        for i, label in enumerate(LABELS):
                            if label in df.columns:
                                label_data.append(df[label].values)
                            else:
                                mapped_label = label_mapping.get(label)
                                if mapped_label and mapped_label in df.columns:
                                    label_data.append(df[mapped_label].values)
                        
                        labels = np.column_stack(label_data).tolist()
                        all_texts.extend(texts)
                        all_labels.extend(labels)
                    else:
                        print(f"Skipping {filename}: Missing required label columns")
                        print(f"Available columns: {df.columns.tolist()}")
                        
                except KeyError as e:
                    print(f"Column not found in {filename}: {e}")
                    print(f"Available columns: {df.columns.tolist()}")
                    continue
                    
    return all_texts, all_labels

def train_sentiment_model(training_folders, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", save_dir='./trained_model'):
    """Train multi-label sentiment analysis model"""
    tokenizer, model = download_and_cache_model(model_name)
    texts, labels = load_training_data(training_folders)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    
    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
            return item
        def __len__(self):
            return len(self.labels)
    
    dataset = SentimentDataset(encodings, labels)
    
    model.config.problem_type = "multi_label_classification"
    model.config.id2label = {i: label for i, label in enumerate(LABELS)}
    model.config.label2id = {label: i for i, label in enumerate(LABELS)}
    model.loss = torch.nn.BCEWithLogitsLoss()
    
    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=3,              
        per_device_train_batch_size=16,  
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs',            
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir

def setup_sentiment_analyzer(training_folders):
    """Initialize multi-label sentiment analysis model"""
    save_dir = './trained_model'
    if os.path.exists(save_dir):
        print(f"Loading trained model: {save_dir}")
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            save_dir,
            problem_type="multi_label_classification"
        )
    else:
        print("Trained model not found, starting training...")
        save_dir = train_sentiment_model(training_folders)
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            save_dir,
            problem_type="multi_label_classification"
        )
    
    def multi_label_predict(texts, model, tokenizer, batch_size=32, threshold=0.5):
        """Multi-label prediction"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            for prob in probs:
                labels = [LABELS[j] for j, p in enumerate(prob) if p > threshold]
                results.append(labels if labels else [LABELS[np.argmax(prob)]])
        return results
    
    return lambda texts: multi_label_predict(texts, model, tokenizer)

def analyze_sentiment(sentences, sentiment_pipeline):
    """Batch analyze multi-label sentiment"""
    return sentiment_pipeline(sentences)

# ========================
# 3. 语义分析模块
# ========================
def setup_semantic_model():
    """Initialize semantic similarity model"""
    print("Downloading/loading semantic analysis model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Semantic analysis model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load default model: {e}")
        print("Trying fallback semantic model...")
        
        fallback_models = [
            'paraphrase-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'multi-qa-MiniLM-L6-cos-v1'
        ]
        
        for fallback_model in fallback_models:
            try:
                model = SentenceTransformer(fallback_model)
                print(f"Fallback model {fallback_model} loaded successfully!")
                return model
            except:
                continue
        
        raise Exception("Unable to load semantic analysis model")

def compute_text_similarity(expert_text, translation_text, model, translation_name):
    """Calculate overall text semantic similarity between expert translation and other translations"""
    if not expert_text or not translation_text:
        print(f"Warning: {translation_name} has no valid text for comparison")
        return None
    
    try:
        max_length = 512
        
        def split_text(text, max_len):
            """Segment text"""
            words = text.split()
            segments = []
            current_segment = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > max_len:
                    segments.append(' '.join(current_segment))
                    current_segment = [word]
                    current_length = len(word)
                else:
                    current_segment.append(word)
                    current_length += len(word) + 1
            
            if current_segment:
                segments.append(' '.join(current_segment))
            
            return segments
        
        expert_segments = split_text(expert_text, max_length)
        translation_segments = split_text(translation_text, max_length)
        
        expert_embeddings = model.encode(expert_segments, convert_to_tensor=True)
        translation_embeddings = model.encode(translation_segments, convert_to_tensor=True)
        
        expert_mean_embedding = torch.mean(expert_embeddings, dim=0)
        translation_mean_embedding = torch.mean(translation_embeddings, dim=0)
        
        similarity = util.cos_sim(expert_mean_embedding, translation_mean_embedding).item()
        
        print(f"{translation_name} overall text similarity with expert translation: {similarity:.4f}")
        
        return similarity
        
    except Exception as e:
        print(f"Semantic analysis error ({translation_name}): {e}")
        return None

# ========================
# 4. 综合分析模块
# ========================
def analyze_all_translations(translation_data, full_texts, training_folders):
    """Comprehensive analysis of all translation versions"""
    print("\nInitializing analysis models...")
    sentiment_pipeline = setup_sentiment_analyzer(training_folders)
    semantic_model = setup_semantic_model()
    
    sentiment_results = {}
    
    print("\nStarting sentiment analysis...")
    for name, sentences in translation_data.items():
        if sentences:
            print(f"Analyzing sentiment for {name}...")
            sentiment_results[name] = analyze_sentiment(sentences, sentiment_pipeline)
        else:
            sentiment_results[name] = []
    
    similarity_results = {}
    
    if full_texts.get('Expert Translation'):
        print("\nStarting semantic similarity analysis (using expert translation as reference)...")
        expert_text = full_texts['Expert Translation']
        
        for name, text in full_texts.items():
            if name != 'Expert Translation' and text:
                print(f"Calculating similarity between {name} and expert translation...")
                similarity = compute_text_similarity(
                    expert_text, text, semantic_model, name
                )
                if similarity is not None:
                    similarity_results[name] = similarity
    else:
        print("\nWarning: Expert translation data not found, cannot perform similarity analysis")
    
    return sentiment_results, similarity_results

# ========================
# 5. 新的评分计算模块
# ========================
def calculate_sentiment_similarity(expert_sentiments, translation_sentiments):
    """
    计算翻译版本与专家翻译的情感相似度
    使用正负情感比例的余弦相似度
    """
    def get_sentiment_proportions(sentiments):
        """获取正面和负面情感的比例"""
        if not sentiments:
            return [0.5, 0.5]  # 默认值
        
        total = len(sentiments)
        positive_count = 0
        negative_count = 0
        
        for sentence_labels in sentiments:
            has_positive = any(label in POSITIVE_LABELS for label in sentence_labels)
            has_negative = any(label in NEGATIVE_LABELS for label in sentence_labels)
            
            if has_positive:
                positive_count += 1
            if has_negative:
                negative_count += 1
        
        pos_ratio = positive_count / total if total > 0 else 0
        neg_ratio = negative_count / total if total > 0 else 0
        
        # 归一化到总和为1（处理中性情感的情况）
        total_ratio = pos_ratio + neg_ratio
        if total_ratio > 0:
            pos_ratio = pos_ratio / total_ratio
            neg_ratio = neg_ratio / total_ratio
        else:
            pos_ratio = 0.5
            neg_ratio = 0.5
            
        return [pos_ratio, neg_ratio]
    
    # 获取专家翻译的情感比例
    expert_props = get_sentiment_proportions(expert_sentiments)
    
    # 获取当前翻译的情感比例
    trans_props = get_sentiment_proportions(translation_sentiments)
    
    # 计算余弦相似度
    similarity = cosine_similarity([expert_props], [trans_props])[0][0]
    
    return similarity

def calculate_overall_scores(sentiment_results, similarity_results):
    """
    计算所有翻译版本的综合评分
    overall_score = 0.5 * 语义相似度 + 0.5 * 情感相似度
    """
    overall_scores = {}
    sentiment_similarities = {}
    
    # 获取专家翻译的情感数据
    expert_sentiments = sentiment_results.get('Expert Translation', [])
    
    if not expert_sentiments:
        print("Warning: No Expert Translation sentiment data found")
        # 如果没有专家翻译数据，使用原始方法
        for name in similarity_results.keys():
            if name in sentiment_results:
                overall_scores[name] = similarity_results[name]
                sentiment_similarities[name] = 0
        return overall_scores, sentiment_similarities
    
    # 计算每个翻译版本的情感相似度和综合评分
    for name in similarity_results.keys():
        if name in sentiment_results:
            # 计算情感相似度
            sentiment_sim = calculate_sentiment_similarity(
                expert_sentiments, 
                sentiment_results[name]
            )
            sentiment_similarities[name] = sentiment_sim
            
            # 计算综合评分：50%语义相似度 + 50%情感相似度
            semantic_sim = similarity_results[name]
            overall_score = 0.5 * semantic_sim + 0.5 * sentiment_sim
            overall_scores[name] = overall_score
    
    return overall_scores, sentiment_similarities

# ========================
# 6. 优化的可视化模块
# ========================
def create_modern_style():
    """创建现代化的图表风格"""
    style = {
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 1.5,
        'grid.alpha': 0.2,
        'grid.linestyle': '--',
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.shadow': True,
        'legend.fancybox': True
    }
    plt.rcParams.update(style)

def prepare_sentiment_data(sentiment_results):
    """准备情感数据用于可视化"""
    sentiment_data = []
    for name, sentiments in sentiment_results.items():
        if sentiments:
            sentiment_counts = {label: 0 for label in LABELS}
            for sentence_labels in sentiments:
                for label in sentence_labels:
                    sentiment_counts[label] += 1
            sentiment_data.append({
                'Translation': name,
                **sentiment_counts,
                'Total': len(sentiments)
            })
    return sentiment_data

def calculate_polarity_distribution(sentiment_results):
    """计算每个翻译版本的极性分布"""
    polarity_data = []
    
    for name, sentiments in sentiment_results.items():
        if not sentiments:
            continue
            
        # 初始化极性计数
        polarity_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
        
        # 遍历每个句子的情感标签
        for sentence_labels in sentiments:
            has_positive = any(label in POSITIVE_LABELS for label in sentence_labels)
            has_negative = any(label in NEGATIVE_LABELS for label in sentence_labels)
            has_neutral = any(label in NEUTRAL_LABELS for label in sentence_labels)
            
            # 根据规则确定句子的极性
            if has_positive and not has_negative:
                polarity_counts['Positive'] += 1
            elif has_negative and not has_positive:
                polarity_counts['Negative'] += 1
            else:
                # 包括混合情感和中性情感
                polarity_counts['Neutral'] += 1
        
        row_data = {'Translation': name}
        row_data.update(polarity_counts)
        row_data['Total'] = sum(polarity_counts.values())
        polarity_data.append(row_data)
    
    return polarity_data

def plot_sentiment_radar_fixed(sentiment_results, ax):
    """绘制情感分布雷达图 - 优化版本"""
    categories = LABELS[:5]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd']
    
    for idx, (name, sentiments) in enumerate(sentiment_results.items()):
        if sentiments and name != 'Expert Translation':
            values = []
            total = len(sentiments)
            for category in categories:
                count = sum(1 for s_labels in sentiments if category in s_labels)
                values.append(count / total * 100)
            values += values[:1]
            
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            
            ax.plot(angles, values, linestyle=line_style, linewidth=2, 
                   label=name, marker=marker, markersize=6)
            ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0, 50)
    ax.set_title('Sentiment Distribution Radar Chart', 
                fontsize=11, fontweight='bold', pad=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
             ncol=2, frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_similarity_with_style_fixed(similarity_results, ax):
    """绘制带样式的语义相似度图 - 修复版本"""
    names = list(similarity_results.keys())
    similarities = list(similarity_results.values())
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    x_positions = np.arange(len(names))
    bar_width = 0.6
    
    bars = ax.bar(x_positions, similarities, width=bar_width, 
                  color=colors, edgecolor='white', linewidth=2, alpha=0.8)
    
    for bar, sim, name in zip(bars, similarities, names):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{sim:.3f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
        
        stars = '★' * min(int(sim * 5), 5)
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                stars, ha='center', va='center', 
                fontsize=12, color='white')
    
    ax.axhline(y=0.9, color='#52B788', linestyle='--', alpha=0.5, label='Excellent (≥0.9)')
    ax.axhline(y=0.8, color='#F77F00', linestyle='--', alpha=0.5, label='Good (≥0.8)')
    ax.axhline(y=0.7, color='#D62828', linestyle='--', alpha=0.5, label='Fair (≥0.7)')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax.set_xlabel('Translation Version', fontsize=12)
    ax.set_ylabel('Cosine Similarity Score', fontsize=12)
    ax.set_title('Semantic Similarity with Expert Translation', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=9)
    ax.yaxis.grid(True, linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)

def plot_comprehensive_heatmap_updated(sentiment_results, similarity_results, ax):
    """绘制综合评分热力图 - 使用新的计算方法"""
    metrics_data = []
    
    # 计算综合评分和情感相似度
    overall_scores, sentiment_similarities = calculate_overall_scores(
        sentiment_results, similarity_results
    )
    
    for name in similarity_results.keys():
        if name in sentiment_results and sentiment_results[name]:
            sentiments = sentiment_results[name]
            
            positive_count = sum(1 for s_labels in sentiments 
                               if any(label in POSITIVE_LABELS for label in s_labels))
            negative_count = sum(1 for s_labels in sentiments 
                               if any(label in NEGATIVE_LABELS for label in s_labels))
            
            total = len(sentiments)
            pos_ratio = positive_count / total if total > 0 else 0
            neg_ratio = negative_count / total if total > 0 else 0
            
            metrics_data.append({
                'Translation': name,
                'Positive Ratio': pos_ratio,
                'Negative Ratio': neg_ratio,
                'Semantic Similarity': similarity_results[name],
                'Sentiment Similarity': sentiment_similarities[name],
                'Overall Score': overall_scores[name]
            })
    
    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics = df_metrics.set_index('Translation')
        
        display_columns = ['Positive Ratio', 'Negative Ratio', 
                          'Semantic Similarity', 'Sentiment Similarity', 'Overall Score']
        df_display = df_metrics[display_columns]
        
        sns.heatmap(df_display.T, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlGn', 
                   center=0.5, 
                   vmin=0, 
                   vmax=1, 
                   ax=ax, 
                   cbar_kws={'label': 'Score', 'shrink': 0.8, 'pad': 0.02},
                   linewidths=2, 
                   linecolor='white',
                   annot_kws={'fontsize': 9})
        
        ax.set_title('Comprehensive Quality Metrics Heatmap\n(Overall Score = 50% Semantic + 50% Sentiment Similarity)', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('')
        ax.set_ylabel('Metrics', fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

def visualize_comprehensive_results(sentiment_results, similarity_results, output_prefix=""):
    """生成现代化的综合对比图表（5合1）- 使用新的评分计算，包含极性分析"""
    create_modern_style()
    
    fig = plt.figure(figsize=(30, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.4)
    
    # 1. 情感分布对比（堆叠柱状图）
    ax1 = fig.add_subplot(gs[0, :2])
    sentiment_data = prepare_sentiment_data(sentiment_results)
    
    if sentiment_data:
        df_sentiment = pd.DataFrame(sentiment_data)
        df_sentiment = df_sentiment.set_index('Translation')
        
        sentiment_cols = [col for col in df_sentiment.columns if col != 'Total']
        df_plot = df_sentiment[sentiment_cols].T
        
        colors = [SENTIMENT_COLORS.get(label, '#888888') for label in sentiment_cols]
        
        df_plot.plot(kind='bar', stacked=True, ax=ax1, color=colors, width=0.65)
        
        ax1.set_title('Sentiment Distribution Across Translations', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Sentiment Categories', fontsize=12)
        ax1.set_ylabel('Number of Sentences', fontsize=12)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax1.legend(title='Translation Version', 
                  bbox_to_anchor=(1.05, 1), 
                  loc='upper left', 
                  frameon=True,
                  fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        for container in ax1.containers:
            labels = ax1.bar_label(container, label_type='center', fontsize=7, fmt='%.0f')
            for label, height in zip(labels, container.datavalues):
                if height < 5:
                    label.set_visible(False)
    
    # 2. 情感比例雷达图
    ax2 = fig.add_subplot(gs[0, 2], projection='polar')
    plot_sentiment_radar_fixed(sentiment_results, ax2)
    
    # 3. 极性分布对比
    ax3 = fig.add_subplot(gs[0, 3])
    polarity_data = calculate_polarity_distribution(sentiment_results)
    if polarity_data:
        df_polarity = pd.DataFrame(polarity_data)
        df_polarity = df_polarity.set_index('Translation')
        df_polarity[POLARITY_CATEGORIES].plot(kind='bar', ax=ax3, 
                                             color=POLARITY_COLORS, 
                                             width=0.65, edgecolor='white')
        ax3.set_title('Polarity Distribution', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Translation Version', fontsize=11)
        ax3.set_ylabel('Number of Sentences', fontsize=11)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha='right', fontsize=9)
        ax3.legend(title='Polarity', fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. 语义相似度对比
    ax4 = fig.add_subplot(gs[1, :2])
    if similarity_results:
        plot_similarity_with_style_fixed(similarity_results, ax4)
    
    # 5. 综合评分热力图 - 使用新的计算方法
    ax5 = fig.add_subplot(gs[1, 2:])
    plot_comprehensive_heatmap_updated(sentiment_results, similarity_results, ax5)
    
    plt.suptitle('Translation Quality Comprehensive Analysis Dashboard (with Polarity Analysis)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f'{output_prefix}_modern_comprehensive_analysis.png' if output_prefix else 'modern_comprehensive_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

def visualize_sentiment_distribution(sentiment_results, output_prefix=""):
    """生成现代化的情感分布对比图 - 修复重叠"""
    create_modern_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    sentiment_data = prepare_sentiment_data(sentiment_results)
    
    if sentiment_data:
        df_sentiment = pd.DataFrame(sentiment_data)
        
        # 左图：分组柱状图
        x = np.arange(len(df_sentiment))
        width = 0.08
        num_bars = len(LABELS)
        
        for i, label in enumerate(LABELS):
            offset = (i - num_bars/2 + 0.5) * width
            color = SENTIMENT_COLORS.get(label, '#888888')
            bars = ax1.bar(x + offset, df_sentiment[label], width * 0.9,
                          label=LABEL_DISPLAY_MAP.get(label, label),
                          color=color, alpha=0.8, edgecolor='white', linewidth=1)
            
            for bar, value in zip(bars, df_sentiment[label]):
                if value > 5:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{int(value)}', ha='center', va='bottom', fontsize=7)
        
        ax1.set_xlabel('Translation Version', fontsize=12)
        ax1.set_ylabel('Number of Sentences', fontsize=12)
        ax1.set_title('Sentiment Distribution by Category', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_sentiment['Translation'], rotation=30, ha='right')
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                  ncol=1, frameon=True, fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        
        # 右图：百分比堆叠图
        df_percent = df_sentiment.set_index('Translation')[LABELS]
        df_percent = df_percent.div(df_percent.sum(axis=1), axis=0) * 100
        
        # 重命名列以应用 LABEL_DISPLAY_MAP
        df_percent.columns = [LABEL_DISPLAY_MAP.get(label, label) for label in df_percent.columns]
        
        # 同样更新颜色字典的键
        colors = [SENTIMENT_COLORS.get(label, '#888888') for label in LABELS]
        
        df_percent.plot(kind='bar', stacked=True, ax=ax2, 
                       color=colors, width=0.6, edgecolor='white', linewidth=0.5)
        
        ax2.set_xlabel('Translation Version', fontsize=12)
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_title('Sentiment Distribution Percentage', fontsize=13, fontweight='bold')
        ax2.legend(title='Sentiment', bbox_to_anchor=(1.02, 1), 
                  loc='upper left', fontsize=8, ncol=1)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
        ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Sentiment Analysis Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f'{output_prefix}_modern_sentiment_distribution.png' if output_prefix else 'modern_sentiment_distribution.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_sentiment_proportion(sentiment_results, output_prefix=""):
    """生成情感比例对比饼图"""
    create_modern_style()
    
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    all_sentiments = []
    labels = []
    for name, sentiments in sentiment_results.items():
        if sentiments and name != 'Expert Translation':
            positive_count = 0

def visualize_polarity_distribution(sentiment_results, output_prefix=""):
    """生成极性分布对比图"""
    create_modern_style()
    
    # 计算极性分布数据
    polarity_data = calculate_polarity_distribution(sentiment_results)
    
    if not polarity_data:
        print("No polarity data available for visualization")
        return
    
    df_polarity = pd.DataFrame(polarity_data)
    df_polarity = df_polarity.set_index('Translation')
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 左图：分组柱状图
    x = np.arange(len(df_polarity))
    width = 0.25
    
    for i, category in enumerate(POLARITY_CATEGORIES):
        offset = (i - len(POLARITY_CATEGORIES)/2 + 0.5) * width
        color = POLARITY_COLORS[i]
        bars = ax1.bar(x + offset, df_polarity[category], width * 0.9,
                      label=category, color=color, alpha=0.8, 
                      edgecolor='white', linewidth=1)
        
        # 添加数值标签
        for bar, value in zip(bars, df_polarity[category]):
            if value > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(value)}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Translation Version', fontsize=12)
    ax1.set_ylabel('Number of Sentences', fontsize=12)
    ax1.set_title('Polarity Distribution by Category', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_polarity.index, rotation=30, ha='right')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
              ncol=1, frameon=True, fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # 右图：百分比堆叠图
    df_percent = df_polarity[POLARITY_CATEGORIES]
    df_percent = df_percent.div(df_percent.sum(axis=1), axis=0) * 100
    
    df_percent.plot(kind='bar', stacked=True, ax=ax2, 
                   color=POLARITY_COLORS, width=0.6, 
                   edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel('Translation Version', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Polarity Distribution Percentage', fontsize=13, fontweight='bold')
    ax2.legend(title='Polarity', bbox_to_anchor=(1.02, 1), 
              loc='upper left', fontsize=10, ncol=1)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Polarity Analysis Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f'{output_prefix}_polarity_distribution.png' if output_prefix else 'polarity_distribution.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_sentiment_percentage_only(sentiment_results, output_prefix=""):
    """单独生成情感分布百分比图"""
    create_modern_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    sentiment_data = prepare_sentiment_data(sentiment_results)
    
    if sentiment_data:
        df_sentiment = pd.DataFrame(sentiment_data)
        
        # 百分比堆叠图
        df_percent = df_sentiment.set_index('Translation')[LABELS]
        df_percent = df_percent.div(df_percent.sum(axis=1), axis=0) * 100
        
        # 重命名列以应用 LABEL_DISPLAY_MAP
        df_percent.columns = [LABEL_DISPLAY_MAP.get(label, label) for label in df_percent.columns]
        
        # 更新颜色
        colors = [SENTIMENT_COLORS.get(label, '#888888') for label in LABELS]
        
        df_percent.plot(kind='bar', stacked=True, ax=ax, 
                       color=colors, width=0.6, edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Translation Version', fontsize=12)  # 增大轴标签字体
        ax.set_ylabel('Percentage (%)', fontsize=12)  # 增大轴标签字体
        ax.set_title('Sentiment Distribution Percentage', fontsize=13, fontweight='bold')  # 增大标题字体
        ax.legend(title='Sentiment', bbox_to_anchor=(1.02, 1), 
                  loc='upper left', fontsize=11, ncol=1, title_fontsize=11)  # 增大图例字体
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)  # 数值标签字体
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f'{output_prefix}_sentiment_percentage_only.png' if output_prefix else 'sentiment_percentage_only.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_polarity_categories_only(sentiment_results, output_prefix=""):
    """单独生成极性分类柱状图"""
    create_modern_style()
    
    # 计算极性分布数据
    polarity_data = calculate_polarity_distribution(sentiment_results)
    
    if not polarity_data:
        print("No polarity data available for visualization")
        return
    
    df_polarity = pd.DataFrame(polarity_data)
    df_polarity = df_polarity.set_index('Translation')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # 分组柱状图
    x = np.arange(len(df_polarity))
    width = 0.25
    
    for i, category in enumerate(POLARITY_CATEGORIES):
        offset = (i - len(POLARITY_CATEGORIES)/2 + 0.5) * width
        color = POLARITY_COLORS[i]
        bars = ax.bar(x + offset, df_polarity[category], width * 0.9,
                      label=category, color=color, alpha=0.8, 
                      edgecolor='white', linewidth=1)
        
        # 添加数值标签
        for bar, value in zip(bars, df_polarity[category]):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(value)}', ha='center', va='bottom', fontsize=9)  # 数值标签字体
    
    ax.set_xlabel('Translation Version', fontsize=12)  # 增大字体
    ax.set_ylabel('Number of Sentences', fontsize=12)  # 增大字体
    ax.set_title('Polarity Distribution by Category', fontsize=13, fontweight='bold')  # 标题字体
    ax.set_xticks(x)
    ax.set_xticklabels(df_polarity.index, rotation=30, ha='right', fontsize=10)  # 增大字体
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
              ncol=1, frameon=True, fontsize=11, title_fontsize=11)  # 增大字体
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f'{output_prefix}_polarity_categories_only.png' if output_prefix else 'polarity_categories_only.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_polarity_percentage_only(sentiment_results, output_prefix=""):
    """单独生成极性百分比堆叠图"""
    create_modern_style()
    
    # 计算极性分布数据
    polarity_data = calculate_polarity_distribution(sentiment_results)
    
    if not polarity_data:
        print("No polarity data available for visualization")
        return
    
    df_polarity = pd.DataFrame(polarity_data)
    df_polarity = df_polarity.set_index('Translation')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # 百分比堆叠图
    df_percent = df_polarity[POLARITY_CATEGORIES]
    df_percent = df_percent.div(df_percent.sum(axis=1), axis=0) * 100
    
    df_percent.plot(kind='bar', stacked=True, ax=ax, 
                   color=POLARITY_COLORS, width=0.6, 
                   edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Translation Version', fontsize=12)  # 增大字体
    ax.set_ylabel('Percentage (%)', fontsize=12)  # 增大字体
    ax.set_title('Polarity Distribution Percentage', fontsize=13, fontweight='bold')  # 增大字体
    ax.legend(title='Polarity', bbox_to_anchor=(1.02, 1), 
              loc='upper left', fontsize=11, ncol=1, title_fontsize=11)  # 增大字体
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)  # 增大字体
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f'{output_prefix}_polarity_percentage_only.png' if output_prefix else 'polarity_percentage_only.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_sentiment_proportion(sentiment_results, output_prefix=""):
    """生成情感比例对比饼图"""
    create_modern_style()
    
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    all_sentiments = []
    labels = []
    for name, sentiments in sentiment_results.items():
        if sentiments and name != 'Expert Translation':
            positive_count = 0
            total = len(sentiments)
            for sentence_labels in sentiments:
                has_positive = any(label in POSITIVE_LABELS for label in sentence_labels)
                if has_positive:
                    positive_count += 1
            pos_ratio = positive_count / total * 100
            all_sentiments.append(pos_ratio)
            labels.append(f"{name}\n({pos_ratio:.1f}% Positive)")
    
    if all_sentiments:
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_sentiments)))
        ax.pie(all_sentiments, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Positive Sentiment Proportion Across Translation Versions', fontsize=14)
        ax.axis('equal')
    
    plt.tight_layout()
    filename = f'{output_prefix}_sentiment_proportion.png' if output_prefix else 'sentiment_proportion.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_sentiment_distribution_single(sentiment_results, output_prefix=""):
    """Generate only the sentiment distribution grouped bar chart"""
    create_modern_style()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sentiment_data = prepare_sentiment_data(sentiment_results)
    
    if sentiment_data:
        df_sentiment = pd.DataFrame(sentiment_data)
        
        # Grouped bar chart
        x = np.arange(len(df_sentiment))
        width = 0.08
        num_bars = len(LABELS)
        
        for i, label in enumerate(LABELS):
            offset = (i - num_bars/2 + 0.5) * width
            color = SENTIMENT_COLORS.get(label, '#888888')
            bars = ax.bar(x + offset, df_sentiment[label], width * 0.9,
                          label=LABEL_DISPLAY_MAP.get(label, label),
                          color=color, alpha=0.8, edgecolor='white', linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars, df_sentiment[label]):
                if value > 20:  # Only show labels for values > 20
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{int(value)}', ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Translation Version', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Sentences', fontsize=12, fontweight='bold')
        ax.set_title('Sentiment Distribution by Category', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df_sentiment['Translation'], rotation=30, ha='right', fontsize=10)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                  ncol=1, frameon=True, fontsize=11, title='Sentiment Category')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    filename = f'{output_prefix}_sentiment_distribution_grouped.png' if output_prefix else 'sentiment_distribution_grouped.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_semantic_similarity(similarity_results, output_prefix=""):
    """生成现代化的语义相似度对比图 - 修改版：增大文字，减小条形高度"""
    create_modern_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))  # 稍微增大图表尺寸
    
    if similarity_results:
        names = list(similarity_results.keys())
        similarities = list(similarity_results.values())
        
        # 左图：柱状图with渐变色
        colors = plt.cm.coolwarm(similarities)
        # 减小条形高度 (原来是默认值，现在设为0.5)
        bars = ax1.barh(names, similarities, height=0.5, color=colors, edgecolor='white', linewidth=1)
        
        for bar, sim in zip(bars, similarities):
            width = bar.get_width()
            # 增大数值标签字体
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{sim:.4f}', ha='left', va='center', fontsize=12, fontweight='bold')  # 从11改为14
            
            if sim >= 0.9:
                grade = 'A+'
                grade_color = '#52B788'
            elif sim >= 0.8:
                grade = 'A'
                grade_color = '#74C69D'
            elif sim >= 0.7:
                grade = 'B'
                grade_color = '#F77F00'
            else:
                grade = 'C'
                grade_color = '#D62828'
            
            # 增大评分标签字体
            ax1.text(width/2, bar.get_y() + bar.get_height()/2,
                    grade, ha='center', va='center', fontsize=16,  
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=grade_color, alpha=0.8))
        
        # 增大轴标签和标题字体
        ax1.set_xlabel('Cosine Similarity Score', fontsize=14)  
        ax1.set_ylabel('Translation Version', fontsize=15)  
        ax1.set_title('Semantic Similarity Ranking', fontsize=16, fontweight='bold')  
        ax1.set_xlim(0, 1.05)
        ax1.grid(axis='x', alpha=0.3)
        
        # 增大y轴刻度标签字体
        ax1.tick_params(axis='y', labelsize=13)  
        ax1.tick_params(axis='x', labelsize=12)  
        
        # 右图：极坐标图
        theta = np.linspace(0, 2*np.pi, len(names), endpoint=False)
        theta = np.concatenate([theta, [theta[0]]])
        similarities_plot = similarities + [similarities[0]]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(theta, similarities_plot, 'o-', linewidth=2.5, markersize=12,  
                color=COLOR_PALETTE['primary'])
        ax2.fill(theta, similarities_plot, alpha=0.3, color=COLOR_PALETTE['primary'])
        
        ax2.set_xticks(theta[:-1])
        ax2.set_xticklabels(names, size=13)  # 从10改为13
        ax2.set_ylim(0, 1)
        ax2.set_title('Similarity Scores Polar View', fontsize=13, fontweight='bold', pad=20)  
        ax2.grid(True, alpha=0.3)
        
        # 增大径向标签字体
        ax2.tick_params(axis='y', labelsize=11)  
        
        for level in [0.3, 0.5, 0.7, 0.9]:
            ax2.plot(theta, [level]*len(theta), 'k--', alpha=0.2, linewidth=0.5)
    
    plt.suptitle('Semantic Similarity Analysis', fontsize=15, fontweight='bold', y=1.02)  
    plt.tight_layout()
    
    filename = f'{output_prefix}_modern_semantic_similarity.png' if output_prefix else 'modern_semantic_similarity.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_scoring_table_updated(sentiment_results, similarity_results, output_prefix=""):
    """生成现代化的综合评分表格 - 使用新的评分计算"""
    create_modern_style()
    
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis('tight')
    ax.axis('off')
    
    # 计算综合评分和情感相似度
    overall_scores, sentiment_similarities = calculate_overall_scores(
        sentiment_results, similarity_results
    )
    
    table_data = []
    for name in similarity_results.keys():
        if name in sentiment_results and sentiment_results[name]:
            sentiments = sentiment_results[name]
            
            positive_count = sum(1 for s_labels in sentiments 
                               if any(label in POSITIVE_LABELS for label in s_labels))
            negative_count = sum(1 for s_labels in sentiments 
                               if any(label in NEGATIVE_LABELS for label in s_labels))
            total = len(sentiments)
            pos_ratio = positive_count / total if total > 0 else 0
            neg_ratio = negative_count / total if total > 0 else 0
            
            table_data.append([
                name,
                f"{similarity_results[name]:.4f}",
                f"{sentiment_similarities[name]:.4f}",
                f"{pos_ratio:.2%}",
                f"{neg_ratio:.2%}",
                f"{overall_scores[name]:.4f}"
            ])
    
    if table_data:
        table_data.sort(key=lambda x: float(x[5]), reverse=True)
        
        for i, row in enumerate(table_data):
            row.insert(0, f"#{i+1}")
        
        headers = ['Rank', 'Translation', 'Semantic\nSimilarity', 'Sentiment\nSimilarity',
                  'Positive\nRatio', 'Negative\nRatio', 'Overall\nScore']
        
        col_widths = [0.06, 0.20, 0.145, 0.145, 0.12, 0.12, 0.145]
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=col_widths)
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2.2)
        
        for i in range(len(headers)):
            table[(0, i)].set_facecolor(COLOR_PALETTE['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.12)
        
        colors_rank = ['#FFD700', '#C0C0C0', '#CD7F32']
        for i in range(1, len(table_data) + 1):
            if i <= 3 and i <= len(colors_rank):
                color = colors_rank[i-1]
                alpha = 0.2
            else:
                color = '#F0F0F0'
                alpha = 0.5
            
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)
                table[(i, j)].set_alpha(alpha)
                
                if j == 6 and i == 1:
                    table[(i, j)].set_text_props(weight='bold', color=COLOR_PALETTE['success'])
        
        fig.text(0.5, 0.92, 'Translation Quality Comprehensive Ranking', 
                ha='center', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.08, 
                'Scoring Formula: 50% Semantic Similarity + 50% Sentiment Similarity (Cosine)', 
                ha='center', fontsize=10, style='italic', color='gray')
    
    filename = f'{output_prefix}_modern_scoring_table.png' if output_prefix else 'modern_scoring_table.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# ========================
# 7. 其他分析函数
# ========================
def extract_triples(text):
    """Extract triples from text"""
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(text)
    triples = []
    for token in doc:
        if token.dep_ == "ROOT":
            subj = None
            obj = None
            for child in token.children:
                if child.dep_.startswith("nsubj") and child.text.lower() not in stop_words:
                    subj = child.text
                elif child.dep_ in ["dobj", "obj"] and child.text.lower() not in stop_words:
                    obj = child.text
            if subj and obj and token.text.lower() not in stop_words:
                triples.append((subj, token.text, obj))
    return triples

def visualize_triples(triple_results, filename="triple_analysis.png", top_n=10):
    """生成现代化的三元组频率对比图"""
    create_modern_style()
    
    versions = list(triple_results.keys())
    num_versions = len(versions)
    
    if num_versions == 0:
        print("No triple data available for visualization")
        return
    
    fig, axes = plt.subplots(
        (num_versions + 1) // 2, 2,
        figsize=(14, 4 * ((num_versions + 1) // 2)),
        constrained_layout=True
    )
    
    if num_versions == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(num_versions, len(axes)):
        axes[i].axis('off')
    
    for idx, version in enumerate(versions):
        ax = axes[idx]
        triples = triple_results[version]
        
        triple_counts = Counter(triples)
        top_triples = triple_counts.most_common(top_n)
        
        if not top_triples:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                   fontsize=12, transform=ax.transAxes)
            ax.set_title(f'{version}', fontsize=12, fontweight='bold')
            ax.axis('off')
            continue
        
        triple_strs = [" → ".join(triple) for triple, _ in top_triples]
        counts = [count for _, count in top_triples]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(triple_strs)))
        bars = ax.barh(range(len(triple_strs)), counts, color=colors, 
                       edgecolor='white', linewidth=1.5, alpha=0.8)
        
        ax.set_yticks(range(len(triple_strs)))
        ax.set_yticklabels(triple_strs, fontsize=9)
        
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{count}', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Frequency', fontsize=10)
        ax.set_title(f'{version}', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(counts) * 1.15)
    
    plt.suptitle('Triple Extraction Frequency Analysis', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def extract_word_frequency(text):
    """Extract word frequency from text"""
    words = [word for word in text.split() if word.lower() not in stop_words]
    word_freq = Counter(words)
    return word_freq

def visualize_top10_word_frequency(word_freq, translation_name="", filename="word_freq.png"):
    """生成现代化的词频统计图"""
    create_modern_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    top15 = word_freq.most_common(15)
    words = [word for word, freq in top15]
    freqs = [freq for word, freq in top15]
    
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(words)))
    bars = ax1.barh(words[::-1], freqs[::-1], color=colors)
    
    for bar, freq in zip(bars, freqs[::-1]):
        width = bar.get_width()
        ax1.text(width + max(freqs)*0.01, bar.get_y() + bar.get_height()/2,
                f'{freq}', ha='left', va='center', fontsize=10)
    
    ax1.set_xlabel('Frequency', fontsize=12)
    ax1.set_title(f'Top 15 High-Frequency Words - {translation_name}', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 右图：词云风格的气泡图
    np.random.seed(42)
    x = np.random.uniform(0, 10, len(words[:10]))
    y = np.random.uniform(0, 10, len(words[:10]))
    
    sizes = [(f/max(freqs[:10]))**0.8 * 2000 for f in freqs[:10]]
    
    scatter = ax2.scatter(x, y, s=sizes, c=freqs[:10], cmap='viridis', 
                         alpha=0.6, edgecolors='white', linewidth=2)
    
    for i, (xi, yi, word, freq) in enumerate(zip(x, y, words[:10], freqs[:10])):
        fontsize = min(12, max(8, int(12 * freq/max(freqs[:10]))))
        ax2.annotate(word, (xi, yi), ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color='white')
    
    ax2.set_xlim(-1, 11)
    ax2.set_ylim(-1, 11)
    ax2.set_title('Word Frequency Bubble Chart', fontsize=13, fontweight='bold')
    ax2.axis('off')
    
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.1, fraction=0.046)
    cbar.set_label('Frequency', fontsize=10)
    
    plt.suptitle(f'Word Frequency Analysis - {translation_name}', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def print_detailed_statistics_updated(sentiment_results, similarity_results, text_type=""):
    """打印详细统计 - 包含新的情感相似度"""
    print("\n" + "="*60)
    print(f"Translation Quality Comprehensive Analysis Report - {text_type}")
    print("="*60)
    
    # 计算综合评分和情感相似度
    overall_scores, sentiment_similarities = calculate_overall_scores(
        sentiment_results, similarity_results
    )
    
    print("\n【Sentiment Analysis Results】")
    for name, sentiments in sentiment_results.items():
        if sentiments:
            sentiment_counts = {label: 0 for label in LABELS}
            for sentence_labels in sentiments:
                for label in sentence_labels:
                    sentiment_counts[label] += 1
            total = len(sentiments)
            
            positive_count = sum(sentiment_counts[label] for label in POSITIVE_LABELS)
            pos_ratio = positive_count / total if total > 0 else 0
            
            print(f"\n{name}:")
            print(f"  Total sentences: {total}")
            for label, count in sentiment_counts.items():
                display_label = LABEL_DISPLAY_MAP.get(label, label)
                print(f"  {display_label}: {count} ({count/total*100:.1f}%)")
            print(f"  Positive sentiment ratio: {pos_ratio:.4f}")
    
    if similarity_results:
        print("\n\n【Similarity Analysis with Expert Translation】")
        print("\n1. Semantic Similarity Scores:")
        
        sorted_semantic = sorted(similarity_results.items(), key=lambda x: x[1], reverse=True)
        
        for i, (name, similarity) in enumerate(sorted_semantic, 1):
            print(f"\n{i}. {name}:")
            print(f"   Semantic similarity: {similarity:.4f}")
            
            if similarity >= 0.9:
                level = "Very High"
            elif similarity >= 0.8:
                level = "High"
            elif similarity >= 0.7:
                level = "Medium-High"
            elif similarity >= 0.6:
                level = "Medium"
            elif similarity >= 0.5:
                level = "Medium-Low"
            else:
                level = "Low"
            
            print(f"   Level: {level}")
        
        print("\n\n2. Sentiment Similarity Scores (Cosine):")
        sorted_sentiment = sorted(sentiment_similarities.items(), key=lambda x: x[1], reverse=True)
        
        for i, (name, similarity) in enumerate(sorted_sentiment, 1):
            print(f"\n{i}. {name}:")
            print(f"   Sentiment similarity: {similarity:.4f}")
    
    print("\n\n【Comprehensive Evaluation】")
    print("Formula: Overall Score = 50% Semantic Similarity + 50% Sentiment Similarity")
    
    sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (name, score) in enumerate(sorted_overall, 1):
        print(f"\n{i}. {name}:")
        print(f"   Semantic similarity: {similarity_results[name]:.4f}")
        print(f"   Sentiment similarity: {sentiment_similarities[name]:.4f}")
        print(f"   Overall score: {score:.4f}")
        
        if score >= 0.9:
            grade = "Excellent (A+)"
        elif score >= 0.8:
            grade = "Very Good (A)"
        elif score >= 0.7:
            grade = "Good (B)"
        elif score >= 0.6:
            grade = "Fair (C)"
        else:
            grade = "Needs Improvement (D)"
        
        print(f"   Grade: {grade}")
    
    if sorted_overall:
        print(f"\n\nBest Overall Translation: {sorted_overall[0][0]} (Score: {sorted_overall[0][1]:.4f})")
    
    print("\n" + "="*60)

def qualitative_analysis_updated(sentiment_results, similarity_results, text_type=""):
    """定性分析 - 使用新的评分方法"""
    print("\n" + "="*60)
    print(f"Translation Quality Qualitative Analysis Report - {text_type}")
    print("="*60)
    
    # 计算综合评分和情感相似度
    overall_scores, sentiment_similarities = calculate_overall_scores(
        sentiment_results, similarity_results
    )
    
    for name in similarity_results.keys():
        if name in sentiment_results and sentiment_results[name]:
            semantic_sim = similarity_results[name]
            sentiment_sim = sentiment_similarities[name]
            overall = overall_scores[name]
            
            print(f"\n{name}:")
            print(f"  Semantic Similarity: {semantic_sim:.4f}")
            print(f"  Sentiment Similarity: {sentiment_sim:.4f}")
            print(f"  Overall Score: {overall:.4f}")
            
            if semantic_sim >= 0.9 and sentiment_sim >= 0.9:
                evaluation = "Excellent - Both semantic and sentiment aspects are highly aligned with expert translation."
            elif semantic_sim >= 0.9 and sentiment_sim < 0.9:
                evaluation = "Very Good - Excellent semantic accuracy, slight sentiment variation."
            elif semantic_sim < 0.9 and sentiment_sim >= 0.9:
                evaluation = "Good - Strong sentiment preservation, minor semantic differences."
            elif semantic_sim >= 0.8 and sentiment_sim >= 0.8:
                evaluation = "Good - Well-balanced translation with good preservation of both aspects."
            elif semantic_sim >= 0.8 or sentiment_sim >= 0.8:
                evaluation = "Fair - Strong in one aspect but needs improvement in the other."
            else:
                evaluation = "Needs Improvement - Both semantic and sentiment aspects could be enhanced."
            
            print(f"  Qualitative Evaluation: {evaluation}")
    
    print("\n" + "="*60)

# ========================
# 8. 主程序
# ========================
def analyze_text_type(translation_folders, training_folders, text_type):
    """分析特定文本类型 - 使用新的评分方法"""
    print(f"\n{'='*80}")
    print(f"ANALYZING {text_type.upper()}")
    print(f"{'='*80}")
    
    # 1. 加载所有翻译数据
    translation_data = {}
    full_texts = {}
    
    for name, folder_path in translation_folders.items():
        print(f"\nLoading {name} data from {folder_path}...")
        sentences, full_text = load_data(folder_path, name)
        translation_data[name] = sentences
        full_texts[name] = full_text
        print(f"  Successfully loaded {len(sentences)} sentences")
        print(f"  Full text length: {len(full_text)} characters")
    
    # 2. 执行综合分析
    sentiment_results, similarity_results = analyze_all_translations(
        translation_data, full_texts, training_folders
    )
    
    # 3. 生成优化的可视化报告 - 使用新的评分方法
    print(f"\nGenerating modern analysis charts for {text_type}...")
    visualize_comprehensive_results(sentiment_results, similarity_results, output_prefix=text_type)
    visualize_sentiment_distribution(sentiment_results, output_prefix=text_type)
    visualize_sentiment_distribution_single(sentiment_results, output_prefix=text_type)
    visualize_sentiment_proportion(sentiment_results, output_prefix=text_type)
    visualize_semantic_similarity(similarity_results, output_prefix=text_type)
    visualize_scoring_table_updated(sentiment_results, similarity_results, output_prefix=text_type)
    # 单独图表
    visualize_polarity_distribution(sentiment_results, output_prefix=text_type)
    visualize_sentiment_percentage_only(sentiment_results, output_prefix=text_type)
    visualize_polarity_categories_only(sentiment_results, output_prefix=text_type)
    visualize_polarity_percentage_only(sentiment_results, output_prefix=text_type)

    
    
    # 4. 打印详细统计 - 使用新的方法
    print_detailed_statistics_updated(sentiment_results, similarity_results, text_type)
    
    # 5. 三元组提取和可视化
    print(f"\n=== Triple Analysis for {text_type} ===")
    triple_results = {}
    for name, text in full_texts.items():
        print(f"Extracting triples for {name}...")
        triples = extract_triples(text)
        triple_results[name] = triples
    
    visualize_triples(triple_results, 
                     filename=f"{text_type}_modern_triple_frequency.png", 
                     top_n=10)
    
    # 6. 词频提取和可视化
    for name, text in full_texts.items():
        print(f"\nExtracting word frequency for {name}...")
        word_freq = extract_word_frequency(text)
        visualize_top10_word_frequency(word_freq, 
                                       translation_name=name,
                                       filename=f"{text_type}_{name}_modern_word_freq.png")
    
    # 7. 定性分析 - 使用新的方法
    qualitative_analysis_updated(sentiment_results, similarity_results, text_type)

if __name__ == "__main__":
    print("=== Multi-Translation Version Quality Analysis Program Started ===")
    print("=== Using Modern Visualization Style with New Scoring System ===")
    print("=== Overall Score = 50% Semantic Similarity + 50% Sentiment Similarity ===")
    
    # 训练集文件夹路径
    TRAINING_FOLDERS = [
        r"C:\Users\ASUS\Desktop\LLM translate\SenWave-main\labeledtweets"
    ]
    
    # 定义每种文本类型的翻译文件夹
    NEWS_TRANSLATION_FOLDERS = {
        'Expert Translation': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\News_English_Version",
        'GPT-4o': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\gpt4o_news",
        'GPT-4': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\gpt4_news",
        'Google Translate': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\google_news",
        'DeepSeek': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\deepseek_news"
    }
    
    RED_MANSIONS_TRANSLATION_FOLDERS = {
        'Expert Translation': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\H.Bencraft Joly",
        'GPT-4o': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\gpt4o_Red_Mansions",
        'GPT-4': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\gpt4_Red_Mansions",
        'Google Translate': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\google_Red_Mansions",
        'DeepSeek': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\deepseek_Red_Mansions"
    }
    
    RED_SORGHUM_TRANSLATION_FOLDERS = {
        'Expert Translation': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\Howard Goldblatt",
        'GPT-4o': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\gpt4o_Red_sorghum",
        'GPT-4': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\gpt4_Red_sorghum",
        'Google Translate': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\google_Red_sorghum",
        'DeepSeek': r"C:\Users\ASUS\Desktop\LLM translate\Data_set\deepseek_Red_sorghum"
    }
    
    # 分析每种文本类型
    analyze_text_type(NEWS_TRANSLATION_FOLDERS, TRAINING_FOLDERS, "news")
    analyze_text_type(RED_MANSIONS_TRANSLATION_FOLDERS, TRAINING_FOLDERS, "red_mansions")
    analyze_text_type(RED_SORGHUM_TRANSLATION_FOLDERS, TRAINING_FOLDERS, "red_sorghum")
    
    print("\n=== All Analyses Complete ===")
    print("Modern visualization results saved with prefixes: news_, red_mansions_, red_sorghum_")
    print("All charts have been generated with enhanced visual styles and new scoring system!")