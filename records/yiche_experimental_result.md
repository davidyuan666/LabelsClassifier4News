| Model(模型)                                                  | Best F1 | Details                                                      | Explaination                     | Ration                                   |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ | -------------------------------- | ---------------------------------------- |
| **distilbert-base-uncased-fr (Original tokenizer)**          | 7.60%   | Eval Loss: 2.1131, Accuracy: 25.52%, Precision: 9.51%, Recall: 7.19% | 自带分词器(不合理，存在太多英文) |                                          |
| **distilbert-base-uncased-fr (Jieba tokenizer)**             | 0.29%   | Eval Loss: 3.1229, Accuracy: 20.40%, Precision: 0.35%, Recall: 0.78% | 分词器需要和预训练模型适配       |                                          |
| **distilbert-base-uncased-fr-tag1 (Original tokenizer, k=2)** | 49.05%  | Accuracy: 73.58%, Precision: 36.79%, Recall: 73.58%          | 一级标签（所有样本）             | P0: 0.29  (2个标签) P1: 0.17(1个标签)    |
| **distilbert-base-uncased-fr-tag2 (Original tokenizer, k=2)** | 36.63%  | Accuracy: 54.95%, Precision: 27.47%, Recall: 54.95%          | 二级标签（所有样本）             | P0: 0.44  (14个标签) P1: 0.13  (1个标签) |
| **distilbert-base-uncased-fr-tag3 (Original tokenizer, k=1)** | 46.94%  | Accuracy: 46.94%, Precision: 46.94%, Recall: 46.94%          | 三级标签（所有样本）             | P0: 0.13  (6个标签) P1: 0.37 (12个标签)  |
| **distilbert-base-uncased-fr-tag4 (Original tokenizer, k=1)** | 79.67%  | Accuracy: 79.67%, Precision: 79.67%, Recall: 79.67%          | 四级标签（所有样本）暂不考虑     |                                          |
| **distilbert-base-uncased-fr-tag1-p0 (Original tokenizer, P0 label, k=1)** | 77.83%  | Accuracy: 77.83%, Precision: 77.83%, Recall: 77.83%          | 一级标签P0专门微调               |                                          |
| **distilbert-base-uncased-fr-tag2-p0 (Original tokenizer, P0 label, k=3)** | 41.35%  | Accuracy: 82.71%, Precision: 27.57%, Recall: 82.71%          | 二级标签P0专门微调               |                                          |
| **distilbert-base-uncased-fr-tag3-p0 (Original tokenizer, P0 label, k=1)** | 99.48%  | Accuracy: 99.48%, Precision: 99.48%, Recall: 99.48%          | 三级标签P0专门微调               |                                          |
| **distilbert-base-uncased-fr-tag3-p0 (Original tokenizer, P1 label, k=1)** | 71.80%  | Accuracy: 71.80%, Precision: 71.80%, Recall: 71.80%          | 三级标签P1专门微调               |                                          |

优化和排期：

一周:

1，训练一个汽车或者中文相关的分词器，提高编码效率 

2，尝试二分类模型，假设一级标签存在N1个，二级存在N2，三级存在N3,构建沙漏过滤 ,但是要考虑到时间cost（并行处理）

一周:

3,   构建多标签分类模型，损失函数使用MAP更合理

4，多样本不平衡导致accuracy指标偏高，尝试用ChatGPT生成平衡样本

其他:

5，使用其他的分类器，包括GPT的指令微调分类器



实验真实数据参考 by David Yuan:

```
没有区分标签，基于所有叶子节点推断的结果

distilbert-based-uncased-fr模型:  原始tokenizer
{'eval_loss': 2.113126516342163, 'eval_accuracy': '25.52%', 'eval_f1': '7.60%', 'eval_precision': '9.51%', 'eval_recall': '7.19%', 'eval_runtime': 48.5463, 'eval_samples_per_second': 245.827, 'eval_steps_per_second': 30.734}

distilbert-based-uncased-fr模型:  jieba tokenizer  (分词和预训练模型没有进行替换和适配导致的)
{'eval_loss': 3.1228644847869873, 'eval_accuracy': '20.40%', 'eval_f1': '0.29%', 'eval_precision': '0.35%', 'eval_recall': '0.78%', 'eval_runtime': 7.03, 'eval_samples_per_second': 1697.581, 'eval_steps_per_second': 212.233}



设置阈值后的指标
distilbert-based-uncased-fr-tag1模型, 原始tokenizer:

For k=1: Accuracy: 43.13%, F1: 43.13%, Precision: 43.13%, Recall: 43.13%
For k=2: Accuracy: 73.58%, F1: 49.05%, Precision: 36.79%, Recall: 73.58% (这个F1最高)
For k=3: Accuracy: 89.37%, F1: 44.69%, Precision: 29.79%, Recall: 89.37%
For k=4: Accuracy: 95.04%, F1: 38.02%, Precision: 23.76%, Recall: 95.04%
For k=5: Accuracy: 98.00%, F1: 32.67%, Precision: 19.60%, Recall: 98.00%
For k=6: Accuracy: 99.33%, F1: 28.38%, Precision: 16.55%, Recall: 99.33%
For k=7: Accuracy: 99.79%, F1: 24.95%, Precision: 14.26%, Recall: 99.79%
For k=8: Accuracy: 100.00%, F1: 22.22%, Precision: 12.50%, Recall: 100.00%
For k=9: Accuracy: 100.00%, F1: 20.00%, Precision: 11.11%, Recall: 100.00%
For k=10: Accuracy: 100.00%, F1: 18.18%, Precision: 10.00%, Recall: 100.00%

distilbert-based-uncased-fr-tag2模型, 原始tokenizer:

For k=1: Accuracy: 29.50%, F1: 29.50%, Precision: 29.50%, Recall: 29.50%
For k=2: Accuracy: 54.95%, F1: 36.63%, Precision: 27.47%, Recall: 54.95%  (这个F1最高)
For k=3: Accuracy: 72.70%, F1: 36.35%, Precision: 24.23%, Recall: 72.70%
For k=4: Accuracy: 84.68%, F1: 33.87%, Precision: 21.17%, Recall: 84.68%
For k=5: Accuracy: 90.47%, F1: 30.16%, Precision: 18.09%, Recall: 90.47%
For k=6: Accuracy: 93.74%, F1: 26.78%, Precision: 15.62%, Recall: 93.74%
For k=7: Accuracy: 95.45%, F1: 23.86%, Precision: 13.64%, Recall: 95.45%
For k=8: Accuracy: 96.66%, F1: 21.48%, Precision: 12.08%, Recall: 96.66%
For k=9: Accuracy: 97.43%, F1: 19.49%, Precision: 10.83%, Recall: 97.43%
For k=10: Accuracy: 98.07%, F1: 17.83%, Precision: 9.81%, Recall: 98.07%


distilbert-based-uncased-fr-tag3模型, 原始tokenizer:

For k=1: Accuracy: 46.94%, F1: 46.94%, Precision: 46.94%, Recall: 46.94%  (这个F1最高)
For k=2: Accuracy: 64.31%, F1: 42.88%, Precision: 32.16%, Recall: 64.31%
For k=3: Accuracy: 74.19%, F1: 37.10%, Precision: 24.73%, Recall: 74.19%
For k=4: Accuracy: 80.86%, F1: 32.34%, Precision: 20.22%, Recall: 80.86%
For k=5: Accuracy: 85.19%, F1: 28.40%, Precision: 17.04%, Recall: 85.19%
For k=6: Accuracy: 88.01%, F1: 25.15%, Precision: 14.67%, Recall: 88.01%
For k=7: Accuracy: 90.08%, F1: 22.52%, Precision: 12.87%, Recall: 90.08%
For k=8: Accuracy: 91.63%, F1: 20.36%, Precision: 11.45%, Recall: 91.63%
For k=9: Accuracy: 92.92%, F1: 18.58%, Precision: 10.32%, Recall: 92.92%
For k=10: Accuracy: 93.90%, F1: 17.07%, Precision: 9.39%, Recall: 93.90%

distilbert-based-uncased-fr-tag4模型, 原始tokenizer:

For k=1: Accuracy: 79.67%, F1: 79.67%, Precision: 79.67%, Recall: 79.67% (这个F1最高)
For k=2: Accuracy: 87.36%, F1: 58.24%, Precision: 43.68%, Recall: 87.36%
For k=3: Accuracy: 90.11%, F1: 45.05%, Precision: 30.04%, Recall: 90.11%
For k=4: Accuracy: 91.21%, F1: 36.48%, Precision: 22.80%, Recall: 91.21%
For k=5: Accuracy: 92.86%, F1: 30.95%, Precision: 18.57%, Recall: 92.86%
For k=6: Accuracy: 93.96%, F1: 26.84%, Precision: 15.66%, Recall: 93.96%
For k=7: Accuracy: 96.70%, F1: 24.18%, Precision: 13.81%, Recall: 96.70%
For k=8: Accuracy: 97.25%, F1: 21.61%, Precision: 12.16%, Recall: 97.25%
For k=9: Accuracy: 97.80%, F1: 19.56%, Precision: 10.87%, Recall: 97.80%
For k=10: Accuracy: 97.80%, F1: 17.78%, Precision: 9.78%, Recall: 97.80%


Tag1样本比例:
P0: 0.29765980133565295  (2个标签)
P1: 0.17990908580728435  (1个标签)

#根据标签单独评估，但是训练数据源于所有标签，只是评估时候用P0和P1单独评估
distilbert-based-uncased-fr-tag1模型, 原始tokenizer, P0标签
For k=1: Accuracy: 6.98%, F1: 6.98%, Precision: 6.98%, Recall: 6.98%
For k=2: Accuracy: 19.23%, F1: 12.82%, Precision: 9.62%, Recall: 19.23%
For k=3: Accuracy: 44.68%, F1: 22.34%, Precision: 14.89%, Recall: 44.68%
For k=4: Accuracy: 71.87%, F1: 28.75%, Precision: 17.97%, Recall: 71.87% (F1最好)
For k=5: Accuracy: 77.75%, F1: 25.92%, Precision: 15.55%, Recall: 77.75%
For k=6: Accuracy: 78.54%, F1: 22.44%, Precision: 13.09%, Recall: 78.54%
For k=7: Accuracy: 87.07%, F1: 21.77%, Precision: 12.44%, Recall: 87.07%
For k=8: Accuracy: 100.00%, F1: 22.22%, Precision: 12.50%, Recall: 100.00%
For k=9: Accuracy: 100.00%, F1: 20.00%, Precision: 11.11%, Recall: 100.00%
For k=10: Accuracy: 100.00%, F1: 18.18%, Precision: 10.00%, Recall: 100.00%

distilbert-based-uncased-fr-tag1模型, 原始tokenizer, P1标签
For k=1: Accuracy: 41.67%, F1: 41.67%, Precision: 41.67%, Recall: 41.67%
For k=2: Accuracy: 74.86%, F1: 49.91%, Precision: 37.43%, Recall: 74.86% (F1最好)
For k=3: Accuracy: 92.76%, F1: 46.38%, Precision: 30.92%, Recall: 92.76%
For k=4: Accuracy: 98.44%, F1: 39.38%, Precision: 24.61%, Recall: 98.44%
For k=5: Accuracy: 100.00%, F1: 33.33%, Precision: 20.00%, Recall: 100.00%
For k=6: Accuracy: 100.00%, F1: 28.57%, Precision: 16.67%, Recall: 100.00%
For k=7: Accuracy: 100.00%, F1: 25.00%, Precision: 14.29%, Recall: 100.00%
For k=8: Accuracy: 100.00%, F1: 22.22%, Precision: 12.50%, Recall: 100.00%
For k=9: Accuracy: 100.00%, F1: 20.00%, Precision: 11.11%, Recall: 100.00%
For k=10: Accuracy: 100.00%, F1: 18.18%, Precision: 10.00%, Recall: 100.00%

Tag2样本比例:
P0: 0.4467866278712014   (14个标签)
P1: 0.1380110584657038   (1个标签)

distilbert-based-uncased-fr-tag2模型, 原始tokenizer, P0标签（标签数量少）
For k=1: Accuracy: 1.59%, F1: 1.59%, Precision: 1.59%, Recall: 1.59%
For k=2: Accuracy: 3.54%, F1: 2.36%, Precision: 1.77%, Recall: 3.54%
For k=3: Accuracy: 8.97%, F1: 4.49%, Precision: 2.99%, Recall: 8.97%
For k=4: Accuracy: 17.16%, F1: 6.86%, Precision: 4.29%, Recall: 17.16%
For k=5: Accuracy: 27.46%, F1: 9.15%, Precision: 5.49%, Recall: 27.46%
For k=6: Accuracy: 34.49%, F1: 9.85%, Precision: 5.75%, Recall: 34.49%
For k=7: Accuracy: 43.08%, F1: 10.77%, Precision: 6.15%, Recall: 43.08%
For k=8: Accuracy: 49.07%, F1: 10.90%, Precision: 6.13%, Recall: 49.07%
For k=9: Accuracy: 55.50%, F1: 11.10%, Precision: 6.17%, Recall: 55.50%  (F1最好)
For k=10: Accuracy: 59.98%, F1: 10.91%, Precision: 6.00%, Recall: 59.98%

distilbert-based-uncased-fr-tag2模型, 原始tokenizer, P1标签 (标签只存在一个，意义不大,且数量不多)
For k=1: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=2: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=3: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=4: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=5: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=6: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=7: Accuracy: 0.06%, F1: 0.02%, Precision: 0.01%, Recall: 0.06%
For k=8: Accuracy: 0.06%, F1: 0.01%, Precision: 0.01%, Recall: 0.06%
For k=9: Accuracy: 0.06%, F1: 0.01%, Precision: 0.01%, Recall: 0.06%
For k=10: Accuracy: 0.25%, F1: 0.05%, Precision: 0.03%, Recall: 0.25%


Tag3:
P0: 0.13193537298040564  (6个标签)
P1: 0.3737366792712272  (12个标签)

distilbert-based-uncased-fr-tag3模型, 原始tokenizer, P0标签  (数据太少)
For k=1: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=2: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=3: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=4: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=5: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=6: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=7: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=8: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=9: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%
For k=10: Accuracy: 0.00%, F1: 0.00%, Precision: 0.00%, Recall: 0.00%


distilbert-based-uncased-fr-tag3模型, 原始tokenizer, P1标签
For k=1: Accuracy: 1.72%, F1: 1.72%, Precision: 1.72%, Recall: 1.72%
For k=2: Accuracy: 3.80%, F1: 2.53%, Precision: 1.90%, Recall: 3.80%
For k=3: Accuracy: 7.54%, F1: 3.77%, Precision: 2.51%, Recall: 7.54%
For k=4: Accuracy: 11.53%, F1: 4.61%, Precision: 2.88%, Recall: 11.53%
For k=5: Accuracy: 15.33%, F1: 5.11%, Precision: 3.07%, Recall: 15.33%
For k=6: Accuracy: 20.29%, F1: 5.80%, Precision: 3.38%, Recall: 20.29%
For k=7: Accuracy: 26.00%, F1: 6.50%, Precision: 3.71%, Recall: 26.00%
For k=8: Accuracy: 29.68%, F1: 6.59%, Precision: 3.71%, Recall: 29.68%
For k=9: Accuracy: 33.05%, F1: 6.61%, Precision: 3.67%, Recall: 33.05% (F1最好)
For k=10: Accuracy: 35.99%, F1: 6.54%, Precision: 3.60%, Recall: 35.99%


下面是根据p0和p1的数据专门微调的版本
distilbert-based-uncased-fr-tag1-p0模型, 原始tokenizer, P0标签
For k=1: Accuracy: 77.83%, F1: 77.83%, Precision: 77.83%, Recall: 77.83%  (F1最高)
For k=2: Accuracy: 100.00%, F1: 66.67%, Precision: 50.00%, Recall: 100.00%
For k=3: Accuracy: 100.00%, F1: 50.00%, Precision: 33.33%, Recall: 100.00%
For k=4: Accuracy: 100.00%, F1: 40.00%, Precision: 25.00%, Recall: 100.00%
For k=5: Accuracy: 100.00%, F1: 33.33%, Precision: 20.00%, Recall: 100.00%
For k=6: Accuracy: 100.00%, F1: 28.57%, Precision: 16.67%, Recall: 100.00%
For k=7: Accuracy: 100.00%, F1: 25.00%, Precision: 14.29%, Recall: 100.00%
For k=8: Accuracy: 100.00%, F1: 22.22%, Precision: 12.50%, Recall: 100.00%
For k=9: Accuracy: 100.00%, F1: 20.00%, Precision: 11.11%, Recall: 100.00%
For k=10: Accuracy: 100.00%, F1: 18.18%, Precision: 10.00%, Recall: 100.00%

distilbert-based-uncased-fr-tag2-p0模型, 原始tokenizer, P0标签
For k=1: Accuracy: 34.35%, F1: 34.35%, Precision: 34.35%, Recall: 34.35%
For k=2: Accuracy: 60.63%, F1: 40.42%, Precision: 30.32%, Recall: 60.63%
For k=3: Accuracy: 82.71%, F1: 41.35%, Precision: 27.57%, Recall: 82.71% (F1最高)
For k=4: Accuracy: 91.07%, F1: 36.43%, Precision: 22.77%, Recall: 91.07%
For k=5: Accuracy: 95.36%, F1: 31.79%, Precision: 19.07%, Recall: 95.36%
For k=6: Accuracy: 97.15%, F1: 27.76%, Precision: 16.19%, Recall: 97.15%
For k=7: Accuracy: 98.19%, F1: 24.55%, Precision: 14.03%, Recall: 98.19%
For k=8: Accuracy: 98.86%, F1: 21.97%, Precision: 12.36%, Recall: 98.86%
For k=9: Accuracy: 99.21%, F1: 19.84%, Precision: 11.02%, Recall: 99.21%
For k=10: Accuracy: 99.55%, F1: 18.10%, Precision: 9.95%, Recall: 99.55%

distilbert-based-uncased-fr-tag3-p0模型, 原始tokenizer, P0标签
For k=1: Accuracy: 99.48%, F1: 99.48%, Precision: 99.48%, Recall: 99.48%  (最高)
For k=2: Accuracy: 100.00%, F1: 66.67%, Precision: 50.00%, Recall: 100.00%
For k=3: Accuracy: 100.00%, F1: 50.00%, Precision: 33.33%, Recall: 100.00%
For k=4: Accuracy: 100.00%, F1: 40.00%, Precision: 25.00%, Recall: 100.00%
For k=5: Accuracy: 100.00%, F1: 33.33%, Precision: 20.00%, Recall: 100.00%
For k=6: Accuracy: 100.00%, F1: 28.57%, Precision: 16.67%, Recall: 100.00%
For k=7: Accuracy: 100.00%, F1: 25.00%, Precision: 14.29%, Recall: 100.00%
For k=8: Accuracy: 100.00%, F1: 22.22%, Precision: 12.50%, Recall: 100.00%
For k=9: Accuracy: 100.00%, F1: 20.00%, Precision: 11.11%, Recall: 100.00%
For k=10: Accuracy: 100.00%, F1: 18.18%, Precision: 10.00%, Recall: 100.00%

distilbert-based-uncased-fr-tag3-p1模型, 原始tokenizer, P1标签
For k=1: Accuracy: 71.80%, F1: 71.80%, Precision: 71.80%, Recall: 71.80% (F1最高)
For k=2: Accuracy: 85.90%, F1: 57.27%, Precision: 42.95%, Recall: 85.90%
For k=3: Accuracy: 91.66%, F1: 45.83%, Precision: 30.55%, Recall: 91.66%
For k=4: Accuracy: 94.91%, F1: 37.96%, Precision: 23.73%, Recall: 94.91%
For k=5: Accuracy: 96.81%, F1: 32.27%, Precision: 19.36%, Recall: 96.81%
For k=6: Accuracy: 98.16%, F1: 28.05%, Precision: 16.36%, Recall: 98.16%
For k=7: Accuracy: 99.08%, F1: 24.77%, Precision: 14.15%, Recall: 99.08%
For k=8: Accuracy: 99.63%, F1: 22.14%, Precision: 12.45%, Recall: 99.63%
For k=9: Accuracy: 99.82%, F1: 19.96%, Precision: 11.09%, Recall: 99.82%
For k=10: Accuracy: 99.94%, F1: 18.17%, Precision: 9.99%, Recall: 99.94%
```