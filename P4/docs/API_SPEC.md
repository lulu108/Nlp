# P4 API Specification

## 1. 文档说明

本文档定义 P4 项目的前后端接口规范。

所有接口均采用 JSON 数据格式进行交互。
接口目标是支持以下功能：

- 中文分词
- 命名实体识别
- 文本分类
- 文本聚类可视化

统一约定：

- 请求方式：`POST`
- 数据格式：`application/json`
- 所有接口返回 JSON
- 返回结果中必须包含 `success` 字段
- HTTP 状态码策略：参数错误使用 `400`，服务端错误使用 `500`

---

## 2. 通用返回格式

### 2.1 成功返回示例

```json
{
  "success": true,
  "message": "ok"
}
```

### 2.2 失败返回示例

```json
{
  "success": false,
  "message": "text field is required"
}
```

## 3. 中文分词接口

### 3.1 接口地址

`POST /api/tokenize`

### 3.2 功能说明

对输入文本进行中文分词，并返回分词结果列表。

### 3.3 请求参数

```json
{
  "text": "今天天气很好，我想去北京旅游。"
}
```

### 3.4 参数说明

- `text`：字符串类型，必填，表示待处理文本

### 3.5 成功返回示例

```json
{
  "success": true,
  "tokens": [
    "今天",
    "天气",
    "很",
    "好",
    "，",
    "我",
    "想",
    "去",
    "北京",
    "旅游",
    "。"
  ]
}
```

### 3.6 失败返回示例

```json
{
  "success": false,
  "message": "text cannot be empty"
}
```

## 4. 命名实体识别接口

### 4.1 接口地址

`POST /api/ner`

### 4.2 功能说明

识别输入文本中的命名实体，并返回实体列表。

### 4.3 请求参数

```json
{
  "text": "小明在北京大学学习。"
}
```

### 4.4 成功返回示例

```json
{
  "success": true,
  "entities": [
    {
      "text": "小明",
      "label": "PER",
      "start": 0,
      "end": 2
    },
    {
      "text": "北京大学",
      "label": "ORG",
      "start": 3,
      "end": 7
    }
  ]
}
```

### 4.5 字段说明

- `text`：实体文本
- `label`：实体类别，采用受控集合：`PER`、`LOC`、`ORG`、`MISC`、`GPE`、`FAC`、`COMPANY`、`INSTITUTION`
- `start`：实体起始下标
- `end`：实体结束下标

说明：

- 接口字段结构保持不变（`text`、`label`、`start`、`end`）
- 系统优先保持标签可读和稳定；不在受控集合内的细分类别会归并为 `MISC`

### 4.6 失败返回示例

```json
{
  "success": false,
  "message": "text cannot be empty"
}
```

## 5. 文本分类接口

### 5.1 接口地址

`POST /api/classify`

### 5.2 功能说明

对输入文本进行类别预测，返回分类标签与置信度。

### 5.3 请求参数

```json
{
  "text": "人工智能技术正在推动科技行业快速发展。"
}
```

### 5.4 成功返回示例

```json
{
  "success": true,
  "label": "科技",
  "confidence": 0.95
}
```

### 5.5 字段说明

- `label`：预测类别
- `confidence`：预测置信度，范围一般在 0 到 1 之间
- `confidence` 建议统一保留 4 位小数（如 `0.9532`）

### 5.6 失败返回示例

```json
{
  "success": false,
  "message": "classification model is not available"
}
```

## 6. 文本聚类接口

### 6.1 接口地址

`POST /api/cluster`

### 6.2 功能说明

对多篇文本进行向量化、聚类与降维，并返回可视化所需的二维坐标点。

### 6.3 请求参数

```json
{
  "cluster_count": 2,
  "documents": [
    {
      "title": "文本1",
      "text": "苹果公司发布了新手机产品。"
    },
    {
      "title": "文本2",
      "text": "人工智能和大数据促进科技发展。"
    },
    {
      "title": "文本3",
      "text": "这支球队最终赢得了联赛冠军。"
    }
  ]
}
```

### 6.4 参数说明

- `documents`：列表类型，必填
- 每个元素包含：
  - `title`：文本标题
  - `text`：文本内容
- `cluster_count`：整数类型，可选，表示聚类数量（建议 >= 2）

### 6.5 成功返回示例

```json
{
  "success": true,
  "points": [
    {
      "title": "文本1",
      "x": 1.24,
      "y": -0.82,
      "cluster": 0
    },
    {
      "title": "文本2",
      "x": 0.77,
      "y": -0.45,
      "cluster": 0
    },
    {
      "title": "文本3",
      "x": -1.13,
      "y": 1.05,
      "cluster": 1
    }
  ]
}
```

### 6.6 字段说明

- `title`：文本标题
- `x`、`y`：降维后的二维坐标
- `cluster`：聚类类别编号

### 6.7 失败返回示例

```json
{
  "success": false,
  "message": "at least two documents are required"
}
```

## 7. 输入校验建议

空文本处理规则：空字符串输入直接报错，不做自动忽略。

建议后端对以下情况进行校验：

- `text` 字段缺失
- `text` 为空字符串
- `documents` 为空
- 文档数量不足
- 某篇文档缺少 `title` 或 `text`

## 8. 接口设计原则

- 接口命名清晰统一
- 返回结构稳定
- 成功与失败均返回 JSON
- 错误信息尽量简洁明确
- 尽量保证前后端字段一致，减少联调成本

## 9. 前端调用建议

前端调用时建议统一封装为：

- `tokenizeText(text)`
- `recognizeEntities(text)`
- `classifyText(text)`
- `clusterDocuments(documents)`

这样更利于维护和复用。

## 聚类实现说明

`POST /api/cluster` 返回由真实聚类流程生成的结果，处理步骤如下：

1. 中文文本预处理
2. TF-IDF 向量化
3. KMeans 聚类
4. PCA 二维降维

输入规则：

- `documents` 至少包含两篇文档
- 每篇文档必须包含非空的 `title` 和 `text`
- 如果存在 `label` 等额外字段，聚类模块会忽略
- `cluster_count` 为可选参数，默认值为 `2`
- 如果传入 `cluster_count`，必须满足 `2 <= cluster_count <= 有效文档数量`
