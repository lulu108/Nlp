# 1) 嵌套采样，默认生成3类数据集（440 / 600 / 800）

```bash
python P2/sample_thucnews_to_data500.py --seed 42
```

默认类别：

- 体育
- 科技
- 教育
- 房产

默认规模：

- 440 = 4 x 110
- 600 = 4 x 150
- 800 = 4 x 200

# 2) 三组预处理

如果已经有 `data440.txt`、`data600.txt`、`data800.txt`，直接运行：

```bash
python P2/01process_data500.py --input P2/data/data440.txt --tag 440
python P2/01process_data500.py --input P2/data/data600.txt --tag 600
python P2/01process_data500.py --input P2/data/data800.txt --tag 800
```

# 3) 三组聚类（默认 4 类，支持 preset）

```bash
python P2/02vectorize_cluster.py --input P2/data/news_for_cluster_440.csv --tag 440 --preset auto --seed 42 --summary-csv P2/data/cluster_metrics_summary.csv
python P2/02vectorize_cluster.py --input P2/data/news_for_cluster_600.csv --tag 600 --preset auto --seed 42 --summary-csv P2/data/cluster_metrics_summary.csv
python P2/02vectorize_cluster.py --input P2/data/news_for_cluster_800.csv --tag 800 --preset auto --seed 42 --summary-csv P2/data/cluster_metrics_summary.csv
```

说明：

- `--preset auto` 会按 `--tag` 自动套用推荐参数：
- 440 -> 字符级（char）+ 2~4gram（推荐 seed=52）
- 600/800 -> 字符级（char）+ 2~4gram

# 3.2) 一键小范围网格搜索（自动产出 best ACC 配置）

针对 440 建议先跑：

```bash
python P2/02vectorize_cluster.py --input P2/data/news_for_cluster_440.csv --tag 440_grid --preset auto --grid-search true --seed 42
```

会自动输出：

- `P2/data/grid_search_kmeans_440_grid.csv`（全部候选结果）
- `P2/data/grid_search_best_440_grid.json`（best 配置）
- 同时用 best 配置再跑一次完整聚类并输出 `cluster_metrics_kmeans_440_grid.csv`

# 3.1) 手工切换特征模式（可选）

保留原有词级模式（默认）：

```bash
python P2/02vectorize_cluster.py --input P2/data/news_for_cluster_600.csv --tag 600_word --preset none --feature-mode word --ngram-min 1 --ngram-max 1 --seed 42
```

使用字符级模式：

```bash
python P2/02vectorize_cluster.py --input P2/data/news_for_cluster_600.csv --tag 600_char --preset none --feature-mode char --ngram-min 2 --ngram-max 4 --max-df 0.95 --min-df 2 --max-features 30000 --n-init 100 --seed 42
```

# 4) 主要输出文件

聚类指标：

- `P2/data/cluster_metrics_kmeans_440.csv`
- `P2/data/cluster_metrics_kmeans_600.csv`
- `P2/data/cluster_metrics_kmeans_800.csv`

聚类明细：

- `P2/data/cluster_result_kmeans_440.csv`
- `P2/data/cluster_result_kmeans_600.csv`
- `P2/data/cluster_result_kmeans_800.csv`

汇总结果：

- `P2/data/cluster_metrics_summary.csv`

# 5) 其他聚类模型对比实验

为满足实验要求中“用其他模型进行实验，并与 K-means 进行比较分析”的要求，本实验在 KMeans 主实验基础上进一步加入 MiniBatchKMeans 和 Agglomerative Clustering 两种聚类方法，形成三模型对比：

- KMeans：主实验模型，使用全量样本迭代更新聚类中心；
- MiniBatchKMeans：KMeans 的小批量近似版本，适合更大规模文本聚类；
- Agglomerative Clustering：层次聚类方法，用于比较不同聚类思想。

当前主对比建议使用 800 条数据集，因为该组数据满足“不少于 500 条”的实验要求，且样本规模更适合观察模型差异。

## 5.1 三模型对比实验运行命令

推荐运行：

```bash
python P2/03compare_cluster_models.py --input P2/data/news_for_cluster_800.csv --tag 800 --feature-mode char --max-df 0.95 --min-df 2 --ngram-min 2 --ngram-max 4 --max-features 40000 --sublinear-tf true --seed 42 --mbk-batch-sizes 64,128,256,512,800 --mbk-n-inits 20,50 --mbk-max-iters 300,500 --svd-dim 100
```

# 6) 聚类模型对比结果可视化

为便于实验报告展示，本项目提供 `04plot_cluster_comparison.py` 对三模型对比结果进行可视化。该脚本读取第 5 步生成的指标表、聚类结果表和 MiniBatchKMeans 参数搜索表，并输出多张可直接放入实验报告的图片。

## 6.1 可视化脚本运行命令

```bash
python P2/04plot_cluster_comparison.py --tag 800 --input P2/data/news_for_cluster_800.csv --metrics P2/data/cluster_model_comparison_800.csv --result P2/data/cluster_result_compare_800.csv --mbk-grid P2/data/minibatch_kmeans_grid_800.csv
```
