# 统一实验结果汇总

## 1. 实验链路说明
- P1 主线 HMM：标准预处理、BMES 构建、HMM 训练与评估链路。
- P1/test2 迭代版 HMM：面向词典回灌和误差修复的迭代链路。
- P1/BiLSTMCRF 深度学习版本：BiLSTM-CRF 训练评估链路。

## 2. 统一指标总表
| 实验链路 | 标签准确率 | 词级Precision | 词级Recall | 词级F1 | token数 | PER实体数 | LOC实体数 | ORG实体数 | 交叉验证Mean | 交叉验证Std | 训练样本数 | 测试样本数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 主线 HMM | 0.8148 | 0.7789 | 0.7108 | 0.7213 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 310 | 77 |
| P1/test2 迭代版 HMM | 0.7661 | 0.7554 | 0.6889 | 0.6958 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 260 | 64 |
| P1/BiLSTMCRF 深度学习版本 | 0.8373 | 0.8248 | 0.7959 | 0.8057 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 402 | 51 |
| P1/nlp4j_baseline NLP4J 对照实验 | 待补充 | 待补充 | 待补充 | 待补充 | 10 | 1 | 1 | 1 | 待补充 | 待补充 | 待补充 | 待补充 |

## 3. HMM 主线结果
- 标签准确率: 0.8148
- 词级 Precision/Recall/F1: 0.7789 / 0.7108 / 0.7213
- 交叉验证 Mean/Std: 待补充 / 待补充
- 指标来源: P1/logs/hmm_main_latest.txt
- 备注: 优先解析最新主线 HMM 控制台日志

## 4. test2 迭代链路结果
- 标签准确率: 0.7661
- 词级 Precision/Recall/F1: 0.7554 / 0.6889 / 0.6958
- 训练/测试样本数: 260 / 64
- 混淆矩阵: P1\test2\output\confusion_matrix.tsv
- 标签报告: P1\test2\output\label_report.tsv
- 指标来源: D:\Project\project_nlp\Nlp\P1\test2\output\metrics.json; P1\test2\output\confusion_matrix.tsv; P1\test2\output\label_report.tsv
- 备注: 已生成 test2 结构化评估输出 (metrics.json / confusion_matrix / label_report)

## 5. BiLSTM-CRF 结果
- 标签准确率: 0.8373
- 词级 Precision/Recall/F1: 0.8248 / 0.7959 / 0.8057
- 训练/测试样本数: 402 / 51
- 指标来源: P1/BiLSTMCRF/output/confusion_matrix.tsv
- 备注: 由混淆矩阵反推宏平均 P/R/F1，非训练日志直接打印值

## 6. NLP4J 对照实验
- 当前状态: 词典规则型 NLP4J baseline，非训练式模型指标；Accuracy/F1 仍待补充
- 转换结果: 10 个 token, PER 1 个, LOC 1 个, ORG 1 个
- Accuracy/Precision/Recall/F1: 待补充 / 待补充 / 待补充 / 待补充
- 指标来源: P1/nlp4j_baseline/output/nlp4j_result.tsv
- 说明: 当前仅做转换与占位汇总，不把 sample_output.txt 当作真实实验指标。

## 7. NER 结果样例
### 样例 1
- 句子1: 毋庸讳言，涉企谣言如同寄生在市场肌体上的“毒瘤”，不仅误导消费者，更损害企业商誉，让企业蒙受不白之冤，可谓不胜其烦、不堪其苦。
- 人名: 无
- 地名: 无
- 机构名: 无

### 样例 2
- 句子2: AI红包大战之热，令人不禁想起11年前那个改写移动支付格局的除夕之夜。
- 人名: 无
- 地名: 无
- 机构名: 无

### 样例 3
- 句子3: 他表示，目前我国的超大规模市场优势还没有充分发挥出来，扩大消费特别是服务消费潜力巨大，城市更新、传统基础设施改造升级、新型基础设施建设等领域都有巨大的投资空间。
- 人名: 无
- 地名: 无
- 机构名: 无

## 8. 当前不足
- test2 已生成结构化评测输出（metrics.json/混淆矩阵/标签报告）。
- BiLSTM-CRF 当前主要从混淆矩阵反推指标，缺少统一文本日志沉淀。
- 三条链路的指标命名和落盘格式仍未完全统一。

## 9. 下一步需要补 NLP4J
- 若后续接入真实 NLP4J 运行环境，可将 Accuracy/F1 纳入同一份 metrics_summary.csv。
- 也可补充标准答案文件，将转换结果与真实指标一起评估。
