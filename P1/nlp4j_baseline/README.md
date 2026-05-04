# NLP4J 对照实验占位目录

## 1. 当前定位

`P1/nlp4j_baseline` 当前是一个 **Java 17 + Maven** 的命令行实验目录，不是 Spring Boot 项目。

- JDK 版本要求：`17`
- Maven 编译参数：`maven.compiler.release=17`
- 运行方式：命令行
- 当前用途：为 P1 中的 NLP4J 对照实验预留输入、输出、转换脚本和 Java 接入骨架

## 2. 现有文件

- `sample_input.txt`
  - 示例输入，每行一条句子。
- `sample_output.txt`
  - 只是 **TSV 格式示例**，不是一次真实的 NLP4J 运行结果。
- `convert_nlp4j_output.py`
  - 将 NLP4J 风格输出转换为统一的 `output/nlp4j_result.tsv` 与 `output/nlp4j_entities.tsv`。
- `output/`
  - 转换脚本输出目录。
- `pom.xml`
  - Maven 工程配置。
- `src/main/java/Nlp4jSequenceLabelingDemo.java`
  - Java 命令行入口程序。

## 3. Maven 与依赖说明

当前 `pom.xml` 没有强行写入未经确认的 NLP4J Maven 坐标，因此现在可以稳定完成工程编译，但还没有真实接入 NLP4J API。

如果后续你已经确认了可用的 NLP4J jar 或 Maven 坐标，可以按下面两种方式接入：

1. 优先方案：补充真实 Maven 依赖坐标到 `pom.xml`。
2. 本地 jar 方案：将相关 jar 放入 `P1/nlp4j_baseline/lib/`，再按实际 API 适配 Java 代码。

当前 Java 程序会检测 `lib/` 目录和 jar 情况；如果没有真实依赖，会直接给出 fail-fast 提示，不会用伪规则结果冒充 NLP4J。

## 4. 运行方式

在项目根目录下执行：

```bash
cd P1/nlp4j_baseline
mvn dependency:tree
mvn compile
mvn exec:java
python convert_nlp4j_output.py --input output/nlp4j_result.tsv
```

默认行为：

- 优先读取 `input/sample_input.txt`
- 如果不存在，则回退读取当前目录下已有的 `sample_input.txt`
- 目标输出文件为 `output/nlp4j_result.tsv`

注意：

- 当前版本仅提供可编译的 Java/Maven 骨架。
- 如果没有真实 NLP4J jar 或尚未完成 API 适配，`mvn exec:java` 会给出明确错误提示。
- 因此当前阶段 **不会产生真实 NLP4J 指标**。

## 5. 输出格式

目标输出文件：`output/nlp4j_result.tsv`

字段为：

```text
sentence_id    token    pos    entity
```

其中：

- `sentence_id`：句子编号，从 1 开始
- `token`：NLP4J 分词结果
- `pos`：词性
- `entity`：`PER / LOC / ORG / O`

## 6. 转换脚本

如果你已经拿到了真实 NLP4J 输出，可以执行：

```bash
python P1/nlp4j_baseline/convert_nlp4j_output.py --input P1/nlp4j_baseline/real_output.txt
```

如果不传 `--input`，脚本默认读取：

```bash
P1/nlp4j_baseline/sample_output.txt
```

转换结果会写到：

- `P1/nlp4j_baseline/output/nlp4j_result.tsv`
- `P1/nlp4j_baseline/output/nlp4j_entities.tsv`

## 7. 基于 nlp4j-core 的词典规则型 baseline

当前以 `nlp4j-core` 的框架能力为基础，补充自定义词典和规则，完成中文序列标注演示：

- 词典路径：`P1/nlp4j_baseline/dict/`
- 规则：最长匹配切分 + 词典实体映射 + 简单 POS 规则
- 说明：这不是训练好的 NLP4J NER 模型，结果不作为 HMM/BiLSTM-CRF 同口径 Accuracy/F1

输出文件：

- `P1/nlp4j_baseline/output/nlp4j_result.tsv`

字段：

```text
sentence_id    token    pos    entity
```

## 8. 当前状态结论

- 第一版 `sample_input.txt`、`sample_output.txt`、转换脚本、汇总占位已经存在。
- 现在已补齐 Java/Maven 命令行工程骨架。
- 当前还没有真实调用 NLP4J。
- 下一步需要补充真实 NLP4J jar 或已确认可用的 Maven 坐标，并在 `Nlp4jSequenceLabelingDemo.java` 中完成 API 适配。

## 9. NLP4J 依赖接入尝试

- 原先尝试 `edu.emory.mathcs.nlp:nlp4j-api:1.1.2`，但当前 Maven 仓库无法解析该依赖。
- 当前改为尝试 `org.nlp4j:nlp4j-core:1.3.7.19`。
- `nlp4j-core` 可能只是核心模块，不一定直接提供中文分词/NER。
- 当前主要目标是完成公开 NLP4J 依赖接入验证。
- 是否已经真实调用 API：否
- 是否支持中文：未确认（需查官方文档与示例）
- Accuracy/F1：待补充

## 10. nlp4j-stanford 可行性探索

目标：评估 `org.nlp4j:nlp4j-stanford:1.3.5.0` 是否能作为中文 token/POS/NER 扩展方案。

执行方式：

```bash
cd P1/nlp4j_baseline
mvn dependency:get "-Dartifact=org.nlp4j:nlp4j-stanford:1.3.5.0" -Dtransitive=false
```

结论要点：

- 该依赖可以从公开 Maven 仓库下载。
- jar 内部包含 `StanfordPosAnnotator` 等 POS 相关类，但未看到 NER/NamedEntity 相关类。
- 其 POM 依赖中 `stanford-corenlp` 与 `stanford-corenlp:models` 为 `provided`，需要额外提供。
- 是否包含中文模型仍未确认，若要中文 NER 通常需要额外 Stanford 中文模型包。
- 因此当前阶段不建议直接将 `nlp4j-stanford` 写入 `pom.xml` 作为中文序列标注方案。
