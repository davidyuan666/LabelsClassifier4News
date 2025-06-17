# LabelsClassifier4News

## Project Overview
LabelsClassifier4News is a machine learning project specifically designed for multi-label news classification. This project implements various text classification models, including BERT-based deep learning models and traditional models like TextCNN, capable of performing multi-label classification predictions on news text.

This project serves as the implementation for the scientific report research paper:

**"Qwens vs. TextCNN, BERT: Multi-label News Classification for Mobile Apps"**



## Key Features
- Supports text classification using pre-trained models like BERT and Chinese RoBERTa
- Implements traditional text classification models like TextCNN
- Provides multiple versions of classifier implementations (v1, v2, v3)
- Supports model training, evaluation, and prediction functions
- Optimized for Chinese news text processing
- Includes ablation study and error analysis functionality
- Supports threshold-based classification strategies
- Provides comprehensive experimental result analysis and visualization

## Project Structure
### Core Directories
- **`bert/`** - BERT-related model implementations
  - `classifier_v1/` - Classifier v1 implementation
  - `classifier_v2/` - Classifier v2 implementation
  - `classifier_v3/` - Classifier v3 implementation
  - `finetune/` - Model fine-tuning related code
  - `models/` - Model definitions
  - `pretrain/` - Pre-trained models

- **`textcnn/`** - TextCNN model implementation
  - `models/` - Model definitions
  - `train_eval.py` - Training and evaluation code
  - `run.py` - Entry point
  - `run_test.py` - Test run script

- **`utils/`** - Utility functions
  - `data/` - Data processing utilities
  - `embedding_utils.py` - Embedding utilities
  - `csv_util.py` - CSV processing utilities

- **`deploy/`** - Deployment-related code
  - `cls_infer/` - Classification inference
  - `gpt_infer/` - GPT inference

### Data and Results
- **`dataset/`** - Dataset storage
- **`records/`** - Training and experiment records
- **`training_result/`** - Training results
- **`plots/`** - Charts and visualization results
- **`ablation_results/`** - Ablation study results
- **`error_analysis_results/`** - Error analysis results
- **`other_threshold_results/`** - Other threshold experiment results

### Analysis Scripts
- **`bert_ablation_study.py`** - BERT ablation study script
- **`error_analysis.py`** - Error analysis script
- **`threshold_based_classification.py`** - Threshold-based classification
- **`train_qwen.py`** - Qwen model training script
- **`data_analyzer.py`** - Data analysis tool
- **`test_resource.py`** - Resource testing script

### Configuration
- **`requirement.txt`** - Project dependencies
- **`Makefile`** - Build automation
- **`.gitignore`** - Git ignore rules
- **`experiment_results.zip`** - Complete experimental results


## Requirements
- Python 3.10+
- PyTorch
- Transformers
- For complete dependency list, please check `requirement.txt`

## Installation
```bash
uv pip install -r requirement.txt
```

## Usage

### 1. Prepare Pre-trained Models
- Place pre-trained models in the `bert_pretrain_models` directory
- Supports models like chinese_roberta_wwm_large_ext_pytorch

### 2. Train Models
```bash
# BERT model training
python bert/classifier_v3/train_bert_p1.py

# TextCNN model training
python textcnn/run.py --model TextCNN

# Qwen model training
python train_qwen.py
```

### 3. Model Prediction
```bash
# BERT prediction
python bert/classifier_v3/predict_p1.py

# Threshold-based classification
python threshold_based_classification.py
```

### 4. Experimental Analysis
```bash
# Ablation study
python bert_ablation_study.py

# Error analysis
python error_analysis.py

# Data analysis
python data_analyzer.py
```

## Dataset
To obtain the dataset, please contact the author at wu.xiguanghua2014@gmail.com. Due to company policy restrictions, we cannot provide the complete dataset, but a subset is available for testing and research purposes.

## Experimental Results
The project contains detailed experimental results and analysis:
- `experiment_results.zip`: Complete experimental results
- `ablation_results/`: Ablation study results
- `error_analysis_results/`: Error analysis results
- `other_threshold_results/`: Threshold optimization experiment results
- `plots/`: Various charts and visualization results

## Contact
For any questions, please contact the project author.

## License
Please refer to the project license file.