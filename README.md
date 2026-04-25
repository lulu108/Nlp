# P4 - 自然语言处理应用程序设计

## 1. 项目简介

本项目为 NLP 实验四：自然语言处理应用程序设计。

项目目标是开发一个基于网页的可交互自然语言处理应用系统，实现以下功能：

- 中文分词
- 命名实体识别
- 文本分类
- 文本聚类与可视化展示
- 系统测试
- 实验结果分析支撑

系统面向课程实验演示，强调功能完整、交互清晰、结果直观、便于实验报告撰写。

---

## 2. 实验目标

通过本项目，完成以下实验目标：

1. 掌握自然语言应用程序的设计方法
2. 实现可交互的文本处理系统
3. 实现处理结果可视化
4. 完成系统测试与实验结果分析

---

## 3. 功能模块

### 3.1 中文分词

用户输入一段中文文本后，系统对文本进行分词处理，并展示分词结果。

### 3.2 命名实体识别

系统识别文本中的实体信息，例如人名、地名、机构名等，并进行结构化展示。

### 3.3 文本分类

系统对输入文本进行类别预测，并返回预测标签及置信度。

### 3.4 文本聚类与可视化

用户输入或上传多篇文本后，系统对文本进行向量化与聚类分析，并通过二维散点图展示聚类结果。

---

## 4. 项目结构

推荐目录结构如下：

```text
P4/
├─ AGENTS.md
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ docs/
│  └─ API_SPEC.md
├─ data/
│  ├─ demo/
│  │  ├─ single_text_examples.json
│  │  └─ cluster_examples.json
│  └─ train/
│     └─ classify_train.csv
├─ models/
│  ├─ classifier/
│  │  ├─ tfidf_vectorizer.pkl
│  │  └─ svm_model.pkl
│  └─ cache/
├─ algorithms/
│  ├─ __init__.py
│  ├─ tokenizer.py
│  ├─ ner.py
│  ├─ classifier.py
│  ├─ cluster.py
│  └─ preprocess.py
├─ backend/
│  ├─ __init__.py
│  ├─ app.py
│  ├─ config.py
│  ├─ routes/
│  │  ├─ __init__.py
│  │  ├─ tokenize.py
│  │  ├─ ner.py
│  │  ├─ classify.py
│  │  └─ cluster.py
│  ├─ services/
│  │  ├─ __init__.py
│  │  ├─ tokenize_service.py
│  │  ├─ ner_service.py
│  │  ├─ classify_service.py
│  │  └─ cluster_service.py
│  └─ utils/
│     ├─ __init__.py
│     ├─ response.py
│     └─ validators.py
├─ frontend/
│  ├─ package.json
│  ├─ src/
│  │  ├─ main.js
│  │  ├─ App.vue
│  │  ├─ api/
│  │  │  └─ nlp.js
│  │  ├─ pages/
│  │  │  ├─ SingleTextPage.vue
│  │  │  └─ ClusterPage.vue
│  │  ├─ components/
│  │  │  ├─ TextInputPanel.vue
│  │  │  ├─ TokenResultCard.vue
│  │  │  ├─ NerResultCard.vue
│  │  │  ├─ ClassifyResultCard.vue
│  │  │  └─ ClusterChart.vue
│  │  └─ styles/
│  │     └─ main.css
├─ tests/
│  ├─ test_tokenize_api.py
│  ├─ test_ner_api.py
│  ├─ test_classify_api.py
│  ├─ test_cluster_api.py
│  └─ test_demo_flow.py
└─ .codex/
   └─ skills/
      ├─ p4-web-module-builder/
      │  └─ SKILL.md
      ├─ p4-nlp-api-integrator/
      │  └─ SKILL.md
      └─ p4-experiment-checker/
         └─ SKILL.md
```

目录说明：

- `frontend/`：前端网页代码
- `backend/`：后端接口与服务
- `algorithms/`：NLP 算法逻辑
- `models/`：模型文件或缓存文件
- `data/`：示例数据、训练数据、测试数据
- `tests/`：系统测试代码
- `docs/`：接口文档、设计说明
- `.codex/skills/`：Codex 技能说明

维护说明：如目录结构发生调整，请同步更新本节树状结构，避免文档与实现不一致。

---

## 5. 系统设计思路

本项目采用“前后端分离 + 算法模块解耦”的设计思路。

### 5.1 前端层

负责：

- 文本输入
- 结果展示
- 聚类图表可视化
- 用户交互反馈

### 5.2 后端层

负责：

- 接收前端请求
- 调用 NLP 服务
- 返回统一格式的 JSON 数据

### 5.3 算法层

负责：

- 分词处理
- 命名实体识别
- 文本分类
- 文本聚类与降维可视化数据生成

这种分层设计能够提高系统的清晰性、可维护性和可扩展性。

---

## 6. 接口说明

后端提供以下接口：

- `POST /api/tokenize`
- `POST /api/ner`
- `POST /api/classify`
- `POST /api/cluster`

详细接口说明见：

- `docs/API_SPEC.md`

---

## 7. 环境要求

- Python 3.10+
- Node.js 18+
- pip
- npm
- HanLP 依赖会随 `requirements.txt` 一并安装，用于中文 NER 基线
- 当前 NER 的 HanLP 路径采用“分词器 + NER”串联推理（先分词，再做 NER），不再使用逐字符输入
- 当 HanLP 模型不可用或推理异常时，系统仍会自动回退到规则识别路径，保证接口稳定可用
- NER 标签采用受控、可读的类别集合：`PER`、`LOC`、`ORG`、`GPE`、`FAC`、`COMPANY`、`INSTITUTION`
- 为保持课程项目可解释性，未纳入受控集合的标签会收敛到 `ORG`，避免输出过多零散类别
- 可通过 `GET /api/meta` 查看 `ner_status.last_used_path`：`hanlp` 表示主路径，`fallback` 表示已回退规则识别

---

## 8. 启动方式（快速运行）

按以下顺序执行：

1. 安装后端依赖

```bash
pip install -r requirements.txt
```

说明：

- `requirements.txt` 已包含 `hanlp`；如果你之前已经安装过旧版本依赖，重新执行一次 `pip install -r requirements.txt` 即可
- 首次调用 NER 模块时，HanLP 可能会自动下载预训练模型文件到本地缓存，因此第一次加载会比后续请求更慢

2. 启动 Flask 后端

```bash
python backend/app.py
```

说明：

- 默认监听 `http://127.0.0.1:5000`
- 默认以单进程模式启动，不开启自动重载，便于排查“5000 端口命中了旧 Flask 进程”的问题
- 如需显式开启调试模式，可在启动前设置 `FLASK_DEBUG=1`
- 如需切换端口，可设置 `PORT=5001` 之类的环境变量

3. 安装前端依赖

```bash
cd frontend
npm install
```

4. 启动前端开发服务

```bash
npm run dev
```

5. 浏览器访问前端地址

- `http://localhost:5173`

6. 后端接口默认地址

- `http://localhost:5000`

7. 启动后自检当前命中的后端

```powershell
Invoke-RestMethod http://localhost:5000/api/meta | ConvertTo-Json -Depth 10
```

返回结果中会包含：

- 当前进程 `pid`
- 当前 Python 解释器路径
- 当前项目根目录

如果这里返回的不是你正在开发的 `P4` 路径，说明 `5000` 端口上仍然挂着旧服务。

---

## 9. 端口冲突排查

当 `test_client` 返回正确结果，但浏览器访问 `http://localhost:5000/api/ner` 却不对时，通常说明浏览器打到的不是当前这份后端。

推荐按下面顺序排查：

1. 查看 5000 端口当前由谁占用

```powershell
netstat -ano | findstr :5000
```

2. 根据最后一列 PID 查看进程

```powershell
tasklist | findstr <PID>
```

3. 确认是否为旧的 Python/Flask 进程；如果是，结束它

```powershell
taskkill /PID <PID> /F
```

4. 重新启动当前仓库的后端，再访问 `/api/meta` 确认命中的服务身份

前端也支持通过环境变量切换后端地址，例如：

```bash
VITE_API_BASE_URL=http://localhost:5001
```

这样即使 `5000` 已被其他旧服务占用，也可以临时改用新的端口进行联调。

---

## 10. 示例输入

### 9.1 单文本示例

```json
{
  "text": "今天天气很好，我想去北京旅游。"
}
```

### 9.2 多文本聚类示例

```json
{
  "documents": [
    { "title": "文本1", "text": "苹果公司发布了新产品。" },
    { "title": "文本2", "text": "人工智能推动了科技发展。" },
    { "title": "文本3", "text": "某球队赢得了比赛冠军。" }
  ]
}
```

---

## 11. 测试说明

建议至少包含以下测试内容：

- 分词接口测试
- NER 接口测试
- 分类接口测试
- 聚类接口测试
- 前后端联调测试
- 非法输入测试

测试代码放在 `tests/` 目录下。

建议执行：

```bash
pytest
```

---

## 12. 演示与截图建议

建议按以下顺序演示：

- 打开首页
- 输入一段文本
- 展示分词结果
- 展示命名实体识别结果
- 展示分类结果
- 切换到聚类页面
- 输入多篇文本并展示聚类散点图

建议截图清单：

- 首页截图
- 分词结果截图
- NER 结果截图
- 分类结果截图
- 聚类可视化截图

该顺序也适合实验报告截图整理。

---

## 13. 当前实现状态

请按实际进度勾选：

- [ ] 分词完成
- [ ] NER 完成
- [ ] 分类完成
- [ ] 聚类完成
- [ ] 测试完成

---

## 14. 可扩展方向

在完成基础实验要求后，可考虑增加：

- 文本摘要
- 情感分析
- 文件上传
- 聚类图交互增强
- 模型切换选项

---

## 15. 项目说明

本项目以课程实验为目标，重点在于：

- 满足实验要求
- 构建完整可运行系统
- 提高界面清晰度和结果可视化效果
- 便于实验过程说明与报告撰写

## 分类器训练说明

使用 `POST /api/classify` 之前，需要先在 `P4/` 目录下训练并生成分类模型文件：

```bash
python scripts/train_classifier.py --data-path data/train/classify_train.csv
```

该命令会生成接口运行所需的两个文件：

- `models/classifier/tfidf_vectorizer.pkl`
- `models/classifier/svm_model.pkl`

如果旧说明中出现 `classifier_model.pkl`，请以当前实际使用的 `svm_model.pkl` 为准。

## 聚类模块说明

`POST /api/cluster` 当前使用真实文本聚类流程：

1. 复用中文文本预处理逻辑
2. 使用 TF-IDF 提取文本特征
3. 使用 KMeans 进行聚类
4. 使用 PCA 将向量降到二维，便于前端可视化

接口返回结构保持不变：

```json
{
  "success": true,
  "points": [{ "title": "doc1", "x": 0.12, "y": -0.45, "cluster": 0 }]
}
```

输入要求：

- 至少提供两篇文档
- 每篇文档必须包含非空的 `title` 和 `text`
- 如果文档中包含 `label` 等额外字段，聚类模块会忽略这些字段
- `cluster_count` 为可选参数，默认值为 `2`
- 如果传入 `cluster_count`，必须满足 `2 <= cluster_count <= 有效文档数量`

演示时也可以从 `data/train/classify_train.csv` 中抽取若干行，将其中的文本内容作为聚类输入。

## 聚类离线评估示例

如果希望基于带标签数据离线评估当前聚类流程，可以在 `P4/` 目录下运行：

```bash
python scripts/evaluate_cluster.py --data-path data/train/classify_train.csv
```

也可以显式指定聚类数量：

```bash
python scripts/evaluate_cluster.py --data-path data/train/classify_train.csv --cluster-count 4
```

如果希望比较多组 TF-IDF + KMeans 参数，可以运行网格搜索：

```bash
python scripts/evaluate_cluster.py --data-path data/train/classify_train.csv --grid-search
```

也可以缩小搜索范围，例如：

```bash
python scripts/evaluate_cluster.py --data-path data/train/classify_train.csv --grid-search --cluster-counts 3,4 --max-features-list 2000,5000 --ngram-ranges 1-1,1-2 --top-n 5
```

如果希望比较基线方案与 SVD 降维后的聚类效果，可以运行：

```bash
python scripts/evaluate_cluster.py --data-path data/train/classify_train.csv --compare-svd
```

单次评估中启用 SVD 的示例：

```bash
python scripts/evaluate_cluster.py --data-path data/train/classify_train.csv --use-svd --svd-components 100
```

如果希望只在 `体育 / 教育 / 财经` 三类子集上做聚类实验，可以先生成子集：

```bash
python scripts/prepare_cluster_subset.py
```

然后运行：

```bash
python scripts/evaluate_cluster.py --data-path data/train/cluster_train_sports_edu_fin.csv --cluster-count 3
```
