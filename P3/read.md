## 1.一次每个标签提取200条数据

```bash
python P3/extract_thucnews_4class.py --per-class 200
```

## 2.阶段1清洗检查

```bash
python P3/01process.py --input-csv P3/data/thucnews_4class_200.csv --output-csv P3/data/thucnews_4class_200_clean.csv --report P3/data/thucnews_4class_200_stage1_report.txt
```

## 3.阶段二预处理

```bash
python P3/02_preprocess_split.py --input-csv P3/data/thucnews_4class_200_clean.csv --output-dir P3/data --report P3/data/thucnews_4class_200_stage2_report.txt
```
