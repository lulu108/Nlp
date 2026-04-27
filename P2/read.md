# 1) 嵌套采样，默认生成 4 类数据集（440 / 600 / 800）

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
