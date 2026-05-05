# P1 项目检查与文档整理

本文档基于当前 P1 目录脚本与数据文件的逐项检查结果整理，仅做现状说明与后续 TODO 规划，不涉及任何业务代码修改。

## 1. 目标与当前状态

### 1.1 项目目标

- 目标 A：将原始中文新闻文本清洗为可训练句子集合。
- 目标 B：基于 HanLP 分词构建 BMES 标注数据。
- 目标 C：训练并评估 HMM 中文分词标注模型（BMES）。
- 目标 D：补充 BiLSTM-CRF 深度模型训练与评估。
- 目标 E：在测试集上进行实体识别结果导出（HanLP NER）。

### 1.2 当前完成度

- 主体功能已基本完成，剩余主要是可复现性文档、依赖锁定和一键运行脚本。
- NLP4J 对照实验已补到 `P1/nlp4j_baseline/`，当前为基于 `org.nlp4j:nlp4j-core` 的规则型中文序列标注 baseline，训练式 NLP4J 中文模型未接入。

## 2. 目录与脚本检查结论

## 2.1 主线脚本（P1 根目录）

- 01preprocess.py：完成文本清洗、分句、去噪、去重，输出 intermediate/news_clean.txt。
- 02build_bmes_dataset.py：完成 HanLP 分词、规则合并、BMES 生成、训练/测试划分。
- 03train_hmm.py：完成 HMM 训练、Viterbi 解码、准确率、混淆矩阵、P/R/F1 与示例输出。
- 04ner_hanlp.py：完成 HanLP NER（人名/地名/机构名）抽取与文本导出。

结论：主线实验链路可跑通，属于可交付基础版本。

### 2.2 备份/阶段版本

- \_081 目录：保留了较早版本主流程，实现与主线基本同构，可用于历史对比。
- test2 目录：早期迭代探索版本（legacy exploration），包含词典回灌与错误分析流程，用于探索与备份，不作为最终主报告链路。

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
- test2/output：已生成结构化评估输出（metrics.json、metrics_summary.txt、confusion_matrix.tsv/png、label_report.tsv）与错误词分析结果。

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

### 4.2 迭代链路（P1/test2，可选）

1. python P1/test2/01preprocess.py
2. python P1/test2/02build_bmes_dataset.py
3. python P1/test2/03train_hmm.py
4. 词典回灌（可选）：python P1/test2/04update_dict.py
5. 回灌后重复步骤 2-3 对比指标变化。

说明：test2 已完成结构化评估输出，包括 metrics.json、metrics_summary.txt、confusion_matrix.tsv/png、label_report.tsv，但仅作为探索/备份版本展示。

### 4.3 深度模型链路（P1/BiLSTMCRF）

1. python P1/BiLSTMCRF/01build_bmes_from_raw.py
2. python P1/BiLSTMCRF/02_prepare_data.py
3. （可选）python P1/BiLSTMCRF/03shrink_fasttext_vec.py
4. python P1/BiLSTMCRF/train_bilstm_crf.py

### 4.4 统一评测汇总（新增）

在完成上述任意实验链路后，可执行统一汇总脚本：

1. 运行命令：python P1/scripts/collect_metrics.py
2. 生成文件：

- P1/reports/experiment_summary.md
- P1/reports/metrics_summary.csv

如果希望汇总结果优先反映你刚运行的主线 HMM（03train_hmm.py），建议先保存最新日志：

- PowerShell：python P1/03train_hmm.py | Out-File -FilePath P1/logs/hmm_main_latest.txt -Encoding utf8
- CMD：python P1/03train_hmm.py > P1\\logs\\hmm_main_latest.txt

然后再执行：python P1/scripts/collect_metrics.py

3. 汇总策略：

- 优先解析现有日志/输出文件，不重跑训练。
- 对格式不一致日志做兼容解析（含 UTF-8/UTF-16）。
- 无法解析的指标统一标记为“待补充”，不做虚构填值。

### 4.5 NLP4J 对照实验（新增）

P1 已在现有 `P1/nlp4j_baseline/` 目录上补充规则型 NLP4J 对照实验，用于课程要求中的序列标注对照。当前状态如下：

- 已完成 `sample_input.txt`、`sample_output.txt`、`convert_nlp4j_output.py`。
- 已补充 `pom.xml` 与 `src/main/java/Nlp4jSequenceLabelingDemo.java`，并实现基于 `nlp4j-core` 的规则型中文序列标注 baseline。
- 当前项目形态是 **Java/Maven 命令行程序**，不是 Spring Boot。
- JDK 目标版本为 `21`。

该 baseline 规则与词典来源：

- 词典来源：jieba 默认词典、THUOCL 开放词库、P1 自动抽词、HanLP NER 输出
- 可输出：token / pos / entity
- 说明：这不是训练式 NLP4J 中文 tokenizer/POS/NER 模型，Accuracy/F1 不与 HMM、BiLSTM-CRF 同口径比较

当前运行方式：

1. `python P1/nlp4j_baseline/scripts/build_external_dicts.py`
2. `cd P1/nlp4j_baseline`
3. `mvn compile`
4. `mvn exec:java`
5. `python convert_nlp4j_output.py --input output/nlp4j_result.tsv`
6. `cd ../../`
7. `python P1/scripts/collect_metrics.py`

单句输入示例：

`mvn exec:java -Dexec.args="张三在北京参加清华大学举办的自然语言处理会议。"`

转换脚本仍保持可用：

1. 如果你已有真实 NLP4J 输出，可保存为 `P1/nlp4j_baseline/real_output.txt`。
2. 执行 `python P1/nlp4j_baseline/convert_nlp4j_output.py --input P1/nlp4j_baseline/real_output.txt`
3. 若不传 `--input`，默认读取 `P1/nlp4j_baseline/sample_output.txt`

统一汇总脚本 `python P1/scripts/collect_metrics.py` 会统计 NLP4J token 数与 PER/LOC/ORG 数，但 Accuracy / Precision / Recall / F1 仍应视为“待补充”。

## 5. TODO 清单（只整理，不改代码）

### 5.1 已完成能力

- 文本清洗、分句、去噪、去重。
- HanLP 分词 + BMES 构建。
- HMM 训练与 Viterbi 预测。
- HMM 指标评估：Accuracy、混淆矩阵、按标签 P/R/F1。
- HanLP NER 导出。
- BiLSTM-CRF 训练与评估（含多种子集成）。

### 5.2 缺失或不完整能力

- 已新增统一实验结果汇总表 (P1/reports/)，主表聚焦主线 HMM/BiLSTM-CRF/NLP4J，对 test2 仅做附录展示。
- 缺自动化回归评测脚本（当前依赖人工运行与观察输出）。
- 已完成规则型 NLP4J baseline（`nlp4j-core` + 词典/规则），但训练式 NLP4J 中文 tokenizer/POS/NER 模型仍未接入。
- 缺固定版本依赖锁定说明（例如 HanLP 词典加载行为在不同环境可能差异）。

### 5.3 后续建议新增/修改文件（规划项）

- 建议新增：P1/docs/reproducibility.md

## 6. 实验要求对应关系

| 实验要求 | 当前实现情况 | 对应文件 |
| 文本预处理（清洗、分句、去噪） | 已完成 | P1/01preprocess.py, P1/test2/01preprocess.py |
| HMM 训练与解码（Viterbi） | 已完成 | P1/03train_hmm.py, P1/test2/03train_hmm.py |
| NER 结果展示 | 已完成 | P1/04ner_hanlp.py, P1/output/ner_hanlp.txt |
| 深度学习对照（BiLSTM-CRF） | 已完成 | P1/BiLSTMCRF/train_bilstm_crf.py |
| NLP4J 对照实验 | 部分完成：已实现基于 nlp4j-core 的规则型中文序列标注 baseline，训练式 NLP4J 中文模型未接入。 | P1/nlp4j_baseline/ |

## 7. 结论

### 7.1 已完成的基础设施

- ✅ P1/reports/experiment_summary.md：汇总三条链路的核心指标与样例分析。
- ✅ P1/scripts/collect_metrics.py：统一解析输出并生成对照表（现已纳入 NLP4J 占位行）。
- ✅ P1/nlp4j_baseline/：已实现基于 `nlp4j-core` 的规则型中文序列标注 baseline，并补充外部词典构建脚本与运行说明。

### 7.2 仍待补充的能力

- test2 结果定位调整：test2 作为词典回灌与错误分析的探索版本保留，不作为最终主结果链路。
- 训练式 NLP4J 模型适配：仍需探索并接入训练式中文 tokenizer/POS/NER 模型（如有可用模块）。
- 自动化一键运行脚本：建议新增 P1/scripts/run_all_experiments.ps1 实现一键执行主线、test2、BiLSTMCRF 并保存日志。
- 可复现性文档：建议新增 P1/docs/reproducibility.md 固化环境、依赖版本、随机种子与复现实验步骤。
- 固定版本依赖锁定说明：HanLP 词典加载行为在不同环境可能差异，需要明确说明。
