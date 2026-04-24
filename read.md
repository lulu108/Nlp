## 1.文本聚类：三类样本数

体育：200
教育：200
财经：200
评估这个子集的命令：

```bash
cd P4
python scripts/evaluate_cluster.py --data-path data/train/cluster_train_sports_edu_fin.csv --cluster-count 3
```

如果想重新从原始训练集生成这个子集，运行：

```bash
cd P4
python scripts/prepare_cluster_subset.py
```

### 最优版本的三类文本聚类验证

```bash
python scripts/evaluate_cluster.py --data-path data/train/cluster_train_sports_edu_fin.csv --cluster-count 3 --max-features 5000 --ngram-range 1,2 --min-df 1 --max-df 0.8 --sublinear-tf
```

- 运行结果：

```bash
===== Cluster Evaluation Summary =====
Documents: 600
Unique labels: 3
Config: pipeline=tfidf_kmeans, cluster_count=3, max_features=5000, ngram_range=1-2, min_df=1, max_df=0.8, sublinear_tf=True, svd_components=None
Vocabulary size: 5000
ARI: 0.8475
NMI: 0.8016
Silhouette Score: 0.0114
Purity: 0.9467
Per-cluster dominant labels:
  cluster=0 | size=219 | dominant_label=体育 | dominant_count=198 | dominant_ratio=0.9041
  cluster=1 | size=203 | dominant_label=财经 | dominant_count=195 | dominant_ratio=0.9606
  cluster=2 | size=178 | dominant_label=教育 | dominant_count=175 | dominant_ratio=0.9831
```
