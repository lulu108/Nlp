# P3 文本分类实验说明

## 1. 实验任务说明

本实验为实验三：文本分类实验。实验目标是基于自行收集并整理的中文新闻文本数据，完成文本分类模型的构建、训练、测试与结果分析。

本实验数据包含 4 个类别：

- 体育
- 科技
- 财经
- 教育

每个类别整理 200 条文本，共 800 条样本。每条样本只对应一个主类别标签，因此本实验属于多类别单标签文本分类任务。

实验主要流程包括：

1. 数据整理与格式统一
2. 数据清洗与质量检查
3. 中文分词、停用词过滤与 TF-IDF 特征构建
4. 训练集、开发集、测试集三划分
5. 使用多种传统机器学习模型进行分类实验
6. 对不同模型的分类结果进行对比分析

---

## 2. 项目目录说明

```text
P3/
├── 01process.py                         # 阶段1：数据清洗与质量检查
├── 02_preprocess_split.py               # 阶段2：分词、三划分与 TF-IDF 特征构建
├── 03_train_linear_svm.py               # 单独训练 Linear SVM 的实验脚本
├── 04_train_classical_models.py         # 多模型对比实验主脚本
├── 05_train_multilabel.py               # 阶段5：多标签文本分类扩展实验
├── extract_thucnews_4class.py           # 数据整理脚本
├── read.md                              # 实验运行说明
└── data/
    ├── thucnews_4class_200.csv          # 整理后的原始数据
    ├── thucnews_4class_200_clean.csv    # 清洗后的数据
    ├── train.csv                        # 训练集
    ├── dev.csv                          # 开发集
    ├── test.csv                         # 测试集
    ├── X_train_tfidf.npz                # 训练集 TF-IDF 特征
    ├── X_dev_tfidf.npz                  # 开发集 TF-IDF 特征
    ├── X_test_tfidf.npz                 # 测试集 TF-IDF 特征
    ├── y_train.npy                      # 训练集标签
    ├── y_dev.npy                        # 开发集标签
    ├── y_test.npy                       # 测试集标签
    ├── tfidf_vectorizer.joblib          # TF-IDF 向量器
    ├── multilabel_news_200.csv          # 多标签扩展实验数据
    ├── thucnews_4class_200_stage1_report.txt
    ├── thucnews_4class_200_stage2_report.txt
    ├── classical_model_outputs/         # 多模型实验输出结果
    └── multilabel_outputs/              # 多标签扩展实验输出结果
```

---

## 3. 阶段0：数据整理

本实验数据由本人自行收集并整理，包含体育、科技、财经、教育四类中文新闻文本，每类 200 条，共 800 条样本。数据统一整理为 CSV 格式，主要字段包括：

- `label`：类别编号
- `category`：类别名称
- `text`：文本内容
- `source_file`：样本来源文件名或样本编号

运行命令：

```bash
python P3/extract_thucnews_4class.py --per-class 200
```

输出文件：

```text
P3/data/thucnews_4class_200.csv
```

---

## 4. 阶段1：数据清洗与质量检查

该阶段主要对整理后的文本数据进行检查和清洗，包括：

1. 检查空值
2. 检查重复文本
3. 检查类别标签映射是否一致
4. 检查文本长度分布
5. 删除异常样本
6. 输出清洗后的数据和检查报告

运行命令：

```bash
python P3/01process.py --input-csv P3/data/thucnews_4class_200.csv --output-csv P3/data/thucnews_4class_200_clean.csv --report P3/data/thucnews_4class_200_stage1_report.txt
```

输出文件：

```text
P3/data/thucnews_4class_200_clean.csv
P3/data/thucnews_4class_200_stage1_report.txt
```

阶段1报告主要用于说明数据质量情况，例如样本总数、各类别分布、空值数量、重复文本数量和清洗后样本数。

---

## 5. 阶段2：文本预处理、三划分与 TF-IDF 特征构建

该阶段主要完成模型训练前的数据处理，包括：

1. 读取清洗后的 CSV 数据
2. 保留 `label / category / text` 三列
3. 按类别进行分层划分，得到训练集、开发集和测试集
4. 对文本进行中文分词
5. 去除停用词
6. 生成 `text_processed` 字段
7. 仅使用训练集拟合 TF-IDF 向量器
8. 分别转换训练集、开发集和测试集特征
9. 保存特征矩阵、标签文件和阶段报告

运行命令：

```bash
python P3/02_preprocess_split.py --input-csv P3/data/thucnews_4class_200_clean.csv --output-dir P3/data --report P3/data/thucnews_4class_200_stage2_report.txt
```

输出文件：

```text
P3/data/train.csv
P3/data/dev.csv
P3/data/test.csv
P3/data/X_train_tfidf.npz
P3/data/X_dev_tfidf.npz
P3/data/X_test_tfidf.npz
P3/data/y_train.npy
P3/data/y_dev.npy
P3/data/y_test.npy
P3/data/tfidf_vectorizer.joblib
P3/data/thucnews_4class_200_stage2_report.txt
```

说明：

本阶段采用 train/dev/test 三划分方式。训练集用于模型训练，开发集用于参数选择，测试集只用于最终性能评估。TF-IDF 向量器只在训练集上拟合，以避免测试集信息泄漏。

---

## 6. 阶段3：Linear SVM 单模型实验

该脚本用于单独训练和分析 Linear SVM 模型。实验流程包括：

1. 读取阶段2生成的 train/dev/test 特征
2. 在训练集上训练模型
3. 在开发集上选择最优参数 `C`
4. 使用 train + dev 重训最终模型
5. 在测试集上进行最终评估
6. 保存模型、预测结果、分类报告和混淆矩阵图

运行命令：

```bash
python P3/03_train_linear_svm.py
```

输出文件：

```text
P3/data/linear_svm_best_model.joblib
P3/data/linear_svm_3way_report.txt
P3/data/linear_svm_test_predictions.csv
P3/data/linear_svm_test_confusion_matrix.png
```

---

## 7. 阶段4：多模型对比实验

该阶段是本实验的主实验部分，使用多种传统机器学习模型进行文本分类，并比较不同模型的性能。

使用的模型包括：

1. Multinomial Naive Bayes，朴素贝叶斯
2. Logistic Regression，逻辑回归
3. Linear SVM，线性支持向量机

实验流程：

1. 读取 train/dev/test 的 TF-IDF 特征和标签
2. 在训练集上训练模型
3. 在开发集上进行参数选择
4. 选择开发集 Macro-F1 最优的参数
5. 使用 train + dev 重新训练最终模型
6. 在测试集上进行最终评估
7. 输出不同模型的 Accuracy、Precision、Recall、Macro-F1 和 Weighted-F1
8. 保存预测结果、分类报告、混淆矩阵图和结果汇总表

运行命令：

```bash
python P3/04_train_classical_models.py
```

输出目录：

```text
P3/data/classical_model_outputs/
```

主要输出文件包括：

```text
P3/data/classical_model_outputs/classical_models_report.txt
P3/data/classical_model_outputs/classical_models_summary.csv
P3/data/classical_model_outputs/dev_search_details.csv

P3/data/classical_model_outputs/nb_best_model.joblib
P3/data/classical_model_outputs/nb_test_predictions.csv
P3/data/classical_model_outputs/nb_test_confusion_matrix.png
P3/data/classical_model_outputs/nb_classification_report.txt

P3/data/classical_model_outputs/lr_best_model.joblib
P3/data/classical_model_outputs/lr_test_predictions.csv
P3/data/classical_model_outputs/lr_test_confusion_matrix.png
P3/data/classical_model_outputs/lr_classification_report.txt

P3/data/classical_model_outputs/svm_best_model.joblib
P3/data/classical_model_outputs/svm_test_predictions.csv
P3/data/classical_model_outputs/svm_test_confusion_matrix.png
P3/data/classical_model_outputs/svm_classification_report.txt
```

---

## 8. 阶段5：多标签文本分类扩展实验

本阶段用于补充验证严格意义上的多标签分类任务。数据文件为：

```text
P3/data/multilabel_news_200.csv
```

每条文本可同时对应体育、科技、财经、教育中的一个或多个标签，标签采用 multi-hot 形式表示。

运行命令：

```bash
python P3/05_train_multilabel.py
```

输出目录：

```text
P3/data/multilabel_outputs/
```

主要输出文件包括：

```text
P3/data/multilabel_outputs/multilabel_report.txt
P3/data/multilabel_outputs/multilabel_predictions.csv
P3/data/multilabel_outputs/multilabel_model.joblib
P3/data/multilabel_outputs/multilabel_tfidf_vectorizer.joblib
```

---

## 9. 实验结果说明

多模型对比实验的最终结果保存在：

```text
P3/data/classical_model_outputs/classical_models_report.txt
```

模型汇总结果保存在：

```text
P3/data/classical_model_outputs/classical_models_summary.csv
```

开发集参数搜索结果保存在：

```text
P3/data/classical_model_outputs/dev_search_details.csv
```

其中：

- `classical_models_report.txt` 用于查看完整实验结果、分类报告和混淆矩阵；
- `classical_models_summary.csv` 用于比较不同模型在测试集上的总体性能；
- `dev_search_details.csv` 用于分析不同模型、不同参数在开发集上的表现；
- `*_test_predictions.csv` 用于查看每条测试样本的真实类别和预测类别；
- `*_test_confusion_matrix.png` 可用于实验报告中的可视化分析。

---

## 10. 推荐运行顺序

完整实验推荐按以下顺序运行：

```bash
python P3/extract_thucnews_4class.py --per-class 200

python P3/01process.py --input-csv P3/data/thucnews_4class_200.csv --output-csv P3/data/thucnews_4class_200_clean.csv --report P3/data/thucnews_4class_200_stage1_report.txt

python P3/02_preprocess_split.py --input-csv P3/data/thucnews_4class_200_clean.csv --output-dir P3/data --report P3/data/thucnews_4class_200_stage2_report.txt

python P3/04_train_classical_models.py

python P3/05_train_multilabel.py
```

如果只想单独查看 Linear SVM 模型效果，可以额外运行：

```bash
python P3/03_train_linear_svm.py
```

---

## 11. 实验报告撰写建议

实验报告中可以重点引用以下内容：

### 11.1 数据规模与类别分布

来自：

```text
P3/data/thucnews_4class_200_stage1_report.txt
```

可用于说明：

- 原始样本数
- 清洗后样本数
- 各类别样本数量
- 标签映射情况
- 空值和重复文本检查结果

### 11.2 数据划分、分词和 TF-IDF 特征维度

来自：

```text
P3/data/thucnews_4class_200_stage2_report.txt
```

可用于说明：

- 训练集、开发集、测试集划分比例
- 各数据集类别分布
- 分词和停用词过滤情况
- TF-IDF 特征维度

### 11.3 多模型实验结果

来自：

```text
P3/data/classical_model_outputs/classical_models_report.txt
P3/data/classical_model_outputs/classical_models_summary.csv
```

可用于说明：

- 朴素贝叶斯、逻辑回归、线性 SVM 的测试集表现
- 不同模型的 Accuracy、Macro-F1、Weighted-F1
- 最优模型及其原因

### 11.4 可视化结果

可以使用：

```text
P3/data/classical_model_outputs/svm_test_confusion_matrix.png
P3/data/classical_model_outputs/nb_test_confusion_matrix.png
P3/data/classical_model_outputs/lr_test_confusion_matrix.png
```

报告中建议重点分析：

- 不同模型在测试集上的 Accuracy 和 Macro-F1；
- 每一类文本的 Precision、Recall 和 F1-score；
- 混淆矩阵中容易被误分类的类别；
- 朴素贝叶斯、逻辑回归和线性 SVM 的性能差异；
- TF-IDF 特征在中文新闻文本分类任务中的优势与局限。

---

## 12. 注意事项

1. 本实验数据应表述为本人自行收集并整理的中文新闻文本数据。
2. 本实验是体育、科技、财经、教育四类文本分类任务。
3. 主实验为多类别单标签分类；扩展实验为严格意义上的多标签文本分类。
4. 训练过程中使用开发集选择参数，测试集只用于最终评估。
5. TF-IDF 向量器只在训练集上拟合，避免测试集信息泄漏。
6. 实验报告中不要将测试集用于参数选择，测试集只用于最终模型评估。
