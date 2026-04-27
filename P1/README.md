# P1 项目检查与文档整理

本文档基于当前 P1 目录脚本与数据文件的逐项检查结果整理，仅做现状说明与后续 TODO 规划，不涉及任何业务代码修改。

## 1. 目标与当前状态

### 1.1 项目目标

- 目标 A：将原始中文新闻文本清洗为可训练句子集合。
- 目标 B：基于 HanLP 分词构建 BMES 标注数据。
- 目标 C：训练并评估 HMM 中文分词标注模型（BMES）。
- 目标 D：补充 BiLSTM-CRF 深度模型训练与评估。
- 目标 E：在测试集上进行实体识别结果导出（HanLP NER）。

### 1.2 当前完成度（估计）

- 已完成约 80%（基础链路完整，含 HMM 与 BiLSTM-CRF 两条训练线）。
- 主要缺口：
  - 缺统一实验报告模板与可复现实验对照表。
  - 缺 NLP4J 版本实现与对齐评测（当前目录未发现相关脚本）。
  - 缺自动化测试与一键评估汇总脚本。

## 2. 目录与脚本检查结论

## 2.1 主线脚本（P1 根目录）

- 01preprocess.py：完成文本清洗、分句、去噪、去重，输出 intermediate/news_clean.txt。
- 02build_bmes_dataset.py：完成 HanLP 分词、规则合并、BMES 生成、训练/测试划分。
- 03train_hmm.py：完成 HMM 训练、Viterbi 解码、准确率、混淆矩阵、P/R/F1 与示例输出。
- 04ner_hanlp.py：完成 HanLP NER（人名/地名/机构名）抽取与文本导出。

结论：主线实验链路可跑通，属于可交付基础版本。

### 2.2 备份/阶段版本

- \_081 目录：保留了较早版本主流程，实现与主线基本同构，可用于历史对比。
- test2 目录：包含增强迭代版流程，新增词典回灌脚本 04update_dict.py，可基于错误词反馈持续改进分词词典。

结论：存在多版本并行，便于迭代，但也带来结果口径分散问题。

### 2.3 深度学习版本（BiLSTMCRF）

- 01build_bmes_from_raw.py：完成从原始文本到 data_bmes.txt 的完整构建（含清洗、规则合并、BMES）。
- 02_prepare_data.py：完成 train/dev/test 三分。
- 03shrink_fasttext_vec.py：完成大词向量裁剪，生成字符子集向量。
- train_bilstm_crf.py：完成 BiLSTM-CRF 训练、评估、早停、混淆矩阵导出，并包含多种子训练与多数投票集成评估。
- 83 子目录：存在一套相对旧版 BiLSTM-CRF 脚本，可做回退或对照。

结论：深度模型链路完整，具备较好的扩展性和实验对照价值。

## 3. 数据文件检查结论

### 3.1 主线与中间产物

- input/data.txt：存在原始文本。
- intermediate/news_clean.txt：存在清洗后句子。
- datasets/auto/train.txt, test.txt, dev.txt：存在自动构建数据集。
- datasets/manual/train_manual.txt, test_manual.txt：存在人工审阅数据。
- output/ner_hanlp.txt：存在 NER 输出结果。
- logs/\*.txt：存在训练与整理日志。

### 3.2 test2 子链路

- test2/input 与 test2/intermediate：存在输入与中间文件。
- test2/datasets/auto：存在 train/test/dev 与 final 文件。
- test2/output：存在错误词与误差样例输出，支持词典迭代。

### 3.3 BiLSTMCRF 数据与产物

- BiLSTMCRF/data/data.txt、data_bmes.txt、train/dev/test：齐全。
- BiLSTMCRF/data/cc.zh.300.vec/cc.zh.300.vec：超大原始向量文件（约 200 万词，300 维）。
- BiLSTMCRF/data/cc.zh.300.char_subset.vec：裁剪后的子集向量文件，已存在。
- BiLSTMCRF/output：存在混淆矩阵等结果产物。

结论：数据资产完整度较高，满足课程实验与扩展实验需求。

## 4. 可复现实验流程（建议）

### 4.1 主线 HMM（P1）

1. 预处理：python P1/01preprocess.py
2. 构建 BMES：python P1/02build_bmes_dataset.py
3. 训练评估 HMM：python P1/03train_hmm.py
4. NER 导出：python P1/04ner_hanlp.py

### 4.2 迭代链路（P1/test2）

1. python P1/test2/01preprocess.py
2. python P1/test2/02build_bmes_dataset.py
3. python P1/test2/03train_hmm.py
4. 词典回灌（可选）：python P1/test2/04update_dict.py
5. 回灌后重复步骤 2-3 对比指标变化。

### 4.3 深度模型链路（P1/BiLSTMCRF）

1. python P1/BiLSTMCRF/01build_bmes_from_raw.py
2. python P1/BiLSTMCRF/02_prepare_data.py
3. （可选）python P1/BiLSTMCRF/03shrink_fasttext_vec.py
4. python P1/BiLSTMCRF/train_bilstm_crf.py

## 5. TODO 清单（只整理，不改代码）

### 5.1 已完成能力

- 文本清洗、分句、去噪、去重。
- HanLP 分词 + BMES 构建。
- HMM 训练与 Viterbi 预测。
- HMM 指标评估：Accuracy、混淆矩阵、按标签 P/R/F1。
- HanLP NER 导出。
- BiLSTM-CRF 训练与评估（含多种子集成）。

### 5.2 缺失或不完整能力

- 缺统一实验结果汇总表（主线、test2、BiLSTMCRF 口径未统一）。
- 缺自动化回归评测脚本（当前依赖人工运行与观察输出）。
- 缺 NLP4J 实现与对照实验（目录中未发现 NLP4J 代码与配置）。
- 缺固定版本依赖锁定说明（例如 HanLP 词典加载行为在不同环境可能差异）。

### 5.3 后续建议新增/修改文件（规划项）

- 建议新增：P1/reports/experiment_summary.md
  - 汇总三条链路的核心指标与样例分析。
- 建议新增：P1/scripts/run_all_experiments.ps1
  - 一键执行主线、test2、BiLSTMCRF 并保存日志。
- 建议新增：P1/scripts/collect_metrics.py
  - 统一解析输出并生成对照表。
- 建议新增：P1/nlp4j_baseline/
  - 放置 NLP4J 版本实现及运行说明。
- 建议新增：P1/docs/reproducibility.md
  - 固化环境、依赖版本、随机种子与复现实验步骤。

说明：以上均为规划建议，当前未做任何代码变更。

## 6. 实验要求对应关系

| 实验要求                                       | 当前实现情况               | 对应文件                                                                                            |
| ---------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------- |
| 文本预处理（清洗、分句、去噪）                 | 已完成                     | P1/01preprocess.py, P1/test2/01preprocess.py                                                        |
| 构建 BMES 标注语料                             | 已完成                     | P1/02build_bmes_dataset.py, P1/test2/02build_bmes_dataset.py, P1/BiLSTMCRF/01build_bmes_from_raw.py |
| HMM 训练与解码（Viterbi）                      | 已完成                     | P1/03train_hmm.py, P1/test2/03train_hmm.py                                                          |
| 评估指标（Accuracy、P/R/F1、Confusion Matrix） | 主线已完成，test2 为简化版 | P1/03train_hmm.py, P1/BiLSTMCRF/train_bilstm_crf.py                                                 |
| NER 结果展示                                   | 已完成                     | P1/04ner_hanlp.py, P1/output/ner_hanlp.txt                                                          |
| 错误驱动迭代（词典更新）                       | 已完成（test2）            | P1/test2/04update_dict.py, P1/test2/output/top_error_words.txt                                      |
| 深度学习对照（BiLSTM-CRF）                     | 已完成                     | P1/BiLSTMCRF/train_bilstm_crf.py                                                                    |
| NLP4J 对照实现                                 | 未完成（未发现）           | 待新增目录与脚本                                                                                    |

## 7. 结论

P1 当前已具备可运行、可评估、可扩展的中文分词实验框架。主线 HMM 与 BiLSTM-CRF 均可作为课程实验的核心实现，test2 提供了错误驱动的迭代能力。后续优先事项建议聚焦在统一评测口径与补齐 NLP4J 对照实现，以提升实验完整性与报告可复现性。
