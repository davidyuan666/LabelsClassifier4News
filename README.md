# LabelsClassifier4News

## Project Overview
LabelsClassifier4News is a machine learning project for multi-label news classification. This project implements various text classification models, including BERT-based deep learning models and traditional models like TextCNN, capable of performing multi-label classification predictions on news text.

## Key Features
- Supports text classification using pre-trained models like BERT and Chinese RoBERTa
- Implements traditional text classification models like TextCNN
- Provides multiple versions of classifier implementations (v1, v2, v3)
- Supports model training, evaluation, and prediction functions
- Optimized for Chinese news text processing

## Project Structure
- `/bert`: BERT-related model implementations
  - `/classifier_v1`, `/classifier_v2`, `/classifier_v3`: Different versions of classifier implementations
  - `/finetune`: Model fine-tuning related code
  - `/models`: Model definitions
  - `/pretrain`: Pre-trained models
- `/textcnn`: TextCNN model implementation
  - `/models`: Model definitions
  - `train_eval.py`: Training and evaluation code
  - `run.py`: Entry point
- `/utils`: Utility functions
- `/deploy`: Deployment-related code
- `/records`: Training and experiment records

## Requirements
- Python 3.6+
- PyTorch
- Transformers
- For other dependencies, please check `bert/classifier_v3/requirements.txt`

## Usage
1. Install Dependencies
   ```
   pip install -r bert/classifier_v3/requirements.txt
   ```

2. Prepare Pre-trained Models
   - Place pre-trained models in the `bert_pretrain_models` directory
   - Supports models like chinese_roberta_wwm_large_ext_pytorch

3. Train Models
   - BERT model: `python bert/classifier_v3/train_bert_p1.py`
   - TextCNN model: `python textcnn/run.py --model TextCNN`

4. Prediction
   - Use scripts like `bert/classifier_v3/predict_p1.py` for prediction

## Dataset
To obtain the dataset, please contact the author (wu.xiguanghua2014@gmail.com). Due to company policy restrictions, we cannot provide the complete dataset, but a portion is available for testing and research purposes.

## Contact
For any questions, please contact the project author