# LLM Translation Quality Evaluation: Mandarin to English

A comprehensive research project for evaluating and comparing the translation quality of various Large Language Models (LLMs) and machine translation systems from Mandarin Chinese to English.

## Project Overview

This project analyzes the translation quality of multiple translation systems by comparing them against professional human translations using advanced NLP techniques including sentiment analysis and semantic similarity measurement.

### Translation Systems Evaluated

- **GPT-4**: OpenAI's GPT-4 model
- **GPT-4o**: OpenAI's GPT-4o (optimized) model
- **DeepSeek**: DeepSeek LLM
- **Google Translate**: Google's machine translation service
- **Expert Translation**: Professional human translations (baseline)

### Text Corpora

1. **Literary Works**
   - *A Dream of Red Mansions* (红楼梦) - Chapters 1-3
     - Expert Translation: H. Bencraft Joly
   - *Red Sorghum* (红高粱) - Chapters 1-4
     - Expert Translation: Howard Goldblatt

2. **News Articles**
   - Global Times news articles (8 articles)
   - English versions as expert baseline

## Key Features

### 1. Semantic Similarity Analysis
- Uses Sentence Transformers (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`)
- Computes cosine similarity between translations and expert versions
- Supports both sentence-level and paragraph-level comparison

### 2. Sentiment Analysis
- Multi-label sentiment classification with 9 emotion categories:
  - **Positive**: Optimistic, Thankful, Humour
  - **Neutral**: Empathetic
  - **Negative**: Pessimistic, Anxious, Sad, Annoyed, Denial
- Fine-tuned on labeled tweet datasets
- Analyzes sentiment preservation across translations

### 3. Polarity Analysis
- Categorizes text into Positive, Neutral, and Negative polarity
- Compares sentiment polarity distribution across different translations

### 4. Comprehensive Scoring System
```
Overall Score = 0.5 × Semantic Similarity + 0.5 × Sentiment Similarity
```
- Semantic Similarity: Cosine similarity with expert translation
- Sentiment Similarity: Cosine similarity of positive/negative ratio vectors

### 5. Advanced Visualizations
- Sentiment distribution charts (stacked bar charts, radar charts)
- Semantic similarity comparison (bar charts, polar charts)
- Polarity distribution analysis
- Comprehensive quality metrics heatmap
- Word frequency analysis and triple extraction
- Modern, publication-ready visualizations with high DPI (300)

## Project Structure

```
LLM-Translation-Madarin/
├── data set/                           # Translation corpora
│   ├── Human/                          # Expert human translations
│   │   ├── News_English_Version/
│   │   ├── Red_mansion_expert_merge/
│   │   └── Red_sorghum_expert_merge/
│   ├── Madarin original/               # Original Chinese texts
│   ├── gpt4/                           # GPT-4 translations
│   ├── gpt4o/                          # GPT-4o translations
│   ├── deepseek/                       # DeepSeek translations
│   └── google/                         # Google Translate translations
│
├── data_pre_processing/                # Data preprocessing scripts
│   ├── data_pre_processing.ipynb       # Convert DOCX to CSV
│   ├── gpt_translate.ipynb             # GPT translation script
│   └── deepseek_translate.ipynb        # DeepSeek translation script
│
├── semantic similarity compare table/  # Sentence-level similarity analysis
│   ├── translation_similarity_analysis.py
│   └── *.xlsx                          # Similarity comparison results
│
├── plot/                               # Generated visualizations
│
├── sentiment and semantic analysis.py  # Main analysis script
│
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)

### Required Packages

```bash
pip install pandas numpy matplotlib seaborn
pip install transformers sentence-transformers torch
pip install scikit-learn networkx spacy nltk
pip install python-docx openpyxl
```

### Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

## Usage

### 1. Data Preprocessing

Convert DOCX files to CSV format:
```python
# See data_pre_processing/data_pre_processing.ipynb
# Converts Word documents into structured CSV files
```

### 2. Run Comprehensive Analysis

Execute the main analysis script:

```python
python "sentiment and semantic analysis.py"
```

This will:
- Load all translation datasets
- Train/load sentiment analysis model
- Perform semantic similarity analysis
- Generate comprehensive visualizations
- Output detailed statistical reports

### 3. Sentence-Level Similarity Analysis

For detailed sentence-level comparison:

```python
cd "semantic similarity compare table"
python translation_similarity_analysis.py
```

## Analysis Workflow

```
1. Data Loading
   └─> Load translations from CSV files

2. Model Initialization
   ├─> Sentiment Analysis Model (multi-label classifier)
   └─> Semantic Model (Sentence Transformers)

3. Analysis
   ├─> Sentiment Distribution Analysis
   ├─> Semantic Similarity Computation
   ├─> Polarity Classification
   └─> Comprehensive Scoring

4. Visualization
   ├─> Sentiment charts (distribution, radar, polarity)
   ├─> Semantic similarity charts
   ├─> Heatmaps and ranking tables
   └─> Word frequency and triple extraction

5. Output
   ├─> PNG visualizations (300 DPI)
   ├─> Console statistical reports
   └─> Qualitative analysis reports
```

## Output Files

The script generates the following visualizations for each text type (news, red_mansions, red_sorghum):

- `{type}_modern_comprehensive_analysis.png` - 5-in-1 comprehensive dashboard
- `{type}_modern_sentiment_distribution.png` - Sentiment distribution charts
- `{type}_polarity_distribution.png` - Polarity analysis
- `{type}_modern_semantic_similarity.png` - Semantic similarity comparison
- `{type}_modern_scoring_table.png` - Comprehensive ranking table
- `{type}_modern_triple_frequency.png` - Triple extraction analysis
- `{type}_{translation}_modern_word_freq.png` - Word frequency analysis

## Key Metrics

### Sentiment Categories
- **Optimistic**: Positive, hopeful expressions
- **Thankful**: Gratitude and appreciation
- **Empathetic**: Understanding and compassion
- **Pessimistic**: Negative, doubtful outlook
- **Anxious**: Worry and nervousness
- **Sad**: Sorrow and melancholy
- **Annoyed**: Irritation and frustration
- **Denial**: Rejection and disbelief
- **Humour**: Jokes and playful content

## Configuration

### Modify Analysis Parameters

Edit `sentiment and semantic analysis.py` to customize:

```python
# Training data folders
TRAINING_FOLDERS = [r"path/to/training/data"]

# Translation folder mappings
NEWS_TRANSLATION_FOLDERS = {
    'Expert Translation': r"path/to/expert",
    'GPT-4o': r"path/to/gpt4o",
    # Add more translations...
}
```

### Adjust Visualization Settings

```python
# Color schemes
COLOR_PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    # Customize colors...
}

# Figure DPI
plt.rcParams['savefig.dpi'] = 300  # High-resolution output
```

## Performance Considerations

- **Memory Usage**: Large datasets may require 8GB+ RAM
- **Processing Time**: Full analysis may take 30-60 minutes depending on dataset size
- **Model Caching**: Models are cached in `~/.cache/huggingface` for faster subsequent runs
- **GPU Acceleration**: Recommended for faster embedding generation

## Research Applications

This project can be used for:

1. **Translation Quality Assessment**: Benchmark LLM translation capabilities
2. **Comparative Studies**: Analyze differences between translation systems
3. **Sentiment Preservation**: Study emotional tone preservation in translations
4. **Literary Translation Analysis**: Evaluate translation quality for literary works
5. **Machine Translation Research**: Develop better evaluation metrics

## Limitations

- Requires expert translations as baseline for comparison
- Sentiment analysis model trained on English tweets (may not perfectly capture literary sentiment)
- Computational resources needed for large-scale analysis
- Current version focuses on Mandarin-to-English translation only

## Future Enhancements

- [ ] Support for more language pairs
- [ ] Real-time translation quality evaluation
- [ ] Integration with more LLM APIs
- [ ] Interactive web dashboard
- [ ] Fine-tuned sentiment models for literary texts
- [ ] Cross-cultural sentiment analysis

## Citation

If you use this project in your research, please cite:

```
@misc{llm-translation-evaluation,
  title={LLM Translation Quality Evaluation: Mandarin to English},
  author={Yue Zhang},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/LLM-Translation-Madarin}
}
```


## Acknowledgments

- Expert translators: H. Bencraft Joly, Howard Goldblatt
- Hugging Face Transformers library
- Sentence Transformers framework
- SenWave sentiment analysis dataset

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.


**Note**: This project is for research and educational purposes. Translation quality assessment is subjective and should be interpreted in context with domain expertise.
