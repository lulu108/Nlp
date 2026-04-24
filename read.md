## 1.三类样本数：

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
