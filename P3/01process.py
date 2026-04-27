"""
阶段1：数据检查与清洗（面向 THUCNews 四分类 CSV）

功能覆盖：
1. 读取 CSV
2. 检查空值
3. 检查重复文本
4. 检查 label-category 映射一致性
5. 检查文本长度分布
6. 清理异常样本并输出清洗结果

默认输入：P3/data/thucnews_4class_150.csv
默认输出：
- P3/data/thucnews_4class_150_clean.csv
- P3/data/thucnews_4class_150_stage1_report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = BASE_DIR / "data" / "thucnews_4class_150.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "data" / "thucnews_4class_150_clean.csv"
DEFAULT_REPORT_TXT = BASE_DIR / "data" / "thucnews_4class_150_stage1_report.txt"

# 约定映射：用于一致性检查
EXPECTED_LABEL_MAP: Dict[str, int] = {
	"体育": 0,
	"科技": 1,
	"财经": 2,
	"教育": 3,
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="阶段1：数据检查与清洗")
	parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="输入 CSV 路径")
	parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="清洗后 CSV 路径")
	parser.add_argument("--report", default=str(DEFAULT_REPORT_TXT), help="检查报告路径")
	parser.add_argument(
		"--min-text-len",
		type=int,
		default=30,
		help="最小文本长度（小于该值的样本将删除）",
	)
	parser.add_argument(
		"--drop-duplicated-text",
		action="store_true",
		help="是否删除重复文本（默认关闭，仅报告）",
	)
	return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
	# 兼容 utf-8-sig 与 utf-8 两种常见导出
	for enc in ("utf-8-sig", "utf-8"):
		try:
			return pd.read_csv(path, encoding=enc)
		except Exception:
			continue
	return pd.read_csv(path)


def ensure_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise ValueError(f"缺少必要列: {missing}，当前列: {list(df.columns)}")


def length_distribution_info(series: pd.Series) -> Dict[str, float]:
	s = series.astype(str).str.len()
	return {
		"count": float(s.shape[0]),
		"min": float(s.min()) if not s.empty else 0.0,
		"max": float(s.max()) if not s.empty else 0.0,
		"mean": float(s.mean()) if not s.empty else 0.0,
		"median": float(s.median()) if not s.empty else 0.0,
		"p10": float(s.quantile(0.10)) if not s.empty else 0.0,
		"p25": float(s.quantile(0.25)) if not s.empty else 0.0,
		"p75": float(s.quantile(0.75)) if not s.empty else 0.0,
		"p90": float(s.quantile(0.90)) if not s.empty else 0.0,
	}


def format_stats(stats: Dict[str, float]) -> str:
	return (
		f"count={int(stats['count'])}, min={int(stats['min'])}, max={int(stats['max'])}, "
		f"mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
		f"p10={stats['p10']:.2f}, p25={stats['p25']:.2f}, p75={stats['p75']:.2f}, p90={stats['p90']:.2f}"
	)


def main() -> None:
	args = parse_args()

	input_csv = Path(args.input_csv)
	output_csv = Path(args.output_csv)
	report_path = Path(args.report)

	if not input_csv.exists():
		raise FileNotFoundError(f"输入文件不存在: {input_csv}")

	print("===== 阶段1：数据检查开始 =====")
	print(f"读取文件: {input_csv}")

	df = safe_read_csv(input_csv)
	original_n = len(df)
	print(f"原始样本数: {original_n}")

	ensure_required_columns(df, required_cols=["label", "category", "text"])

	# 统一类型，避免 NaN/数值带来的判断歧义
	df["category"] = df["category"].astype(str).str.strip()
	df["text"] = df["text"].astype(str).str.strip()
	df["label"] = pd.to_numeric(df["label"], errors="coerce")

	report_lines: List[str] = []
	report_lines.append("===== 阶段1：数据检查与清洗报告 =====")
	report_lines.append(f"输入文件: {input_csv}")
	report_lines.append(f"原始样本数: {original_n}")
	report_lines.append("")

	# 1) 空值检查
	null_label = int(df["label"].isna().sum())
	null_category = int(df["category"].eq("").sum())
	null_text = int(df["text"].eq("").sum())
	report_lines.append("[1] 空值检查")
	report_lines.append(f"label 空值: {null_label}")
	report_lines.append(f"category 空值(空串): {null_category}")
	report_lines.append(f"text 空值(空串): {null_text}")
	report_lines.append("")

	# 2) 重复文本检查
	duplicated_text_count = int(df["text"].duplicated().sum())
	report_lines.append("[2] 重复文本检查")
	report_lines.append(f"重复文本条数(保留首条, 后续重复计数): {duplicated_text_count}")
	report_lines.append("")

	# 3) label-category 映射检查
	report_lines.append("[3] 标签映射检查")
	invalid_map_mask = []
	for idx, row in df[["label", "category"]].iterrows():
		cate = row["category"]
		lb = row["label"]
		exp = EXPECTED_LABEL_MAP.get(cate)
		if exp is None:
			invalid_map_mask.append(idx)
			continue
		if pd.isna(lb) or int(lb) != exp:
			invalid_map_mask.append(idx)

	report_lines.append(f"映射异常条数: {len(invalid_map_mask)}")
	report_lines.append(f"期望映射: {EXPECTED_LABEL_MAP}")
	report_lines.append("")

	# 4) 文本长度分布
	report_lines.append("[4] 文本长度分布")
	before_len_stats = length_distribution_info(df["text"])
	report_lines.append("清洗前长度统计: " + format_stats(before_len_stats))
	report_lines.append("")

	# 5) 清理异常样本
	report_lines.append("[5] 异常清理规则")
	report_lines.append("- 删除 label 为空")
	report_lines.append("- 删除 category 为空")
	report_lines.append("- 删除 text 为空")
	report_lines.append(f"- 删除 text 长度 < {args.min_text_len}")
	report_lines.append("- 删除 category 不在预期集合中的样本")
	report_lines.append("- 删除 label 与 category 映射不一致的样本")
	if args.drop_duplicated_text:
		report_lines.append("- 删除重复 text（仅保留首条）")
	report_lines.append("")

	clean_df = df.copy()
	clean_df = clean_df[clean_df["label"].notna()]
	clean_df = clean_df[clean_df["category"] != ""]
	clean_df = clean_df[clean_df["text"] != ""]

	clean_df["label"] = clean_df["label"].astype(int)
	clean_df = clean_df[clean_df["category"].isin(EXPECTED_LABEL_MAP.keys())]

	# 映射一致性过滤
	clean_df = clean_df[
		clean_df.apply(lambda x: EXPECTED_LABEL_MAP[x["category"]] == int(x["label"]), axis=1)
	]

	# 长度过滤
	clean_df = clean_df[clean_df["text"].str.len() >= args.min_text_len]

	# 去重（可选）
	if args.drop_duplicated_text:
		clean_df = clean_df.drop_duplicates(subset=["text"], keep="first")

	clean_df = clean_df.reset_index(drop=True)

	removed_n = original_n - len(clean_df)
	report_lines.append(f"清洗后样本数: {len(clean_df)}")
	report_lines.append(f"删除样本数: {removed_n}")
	report_lines.append("")

	# 6) 清洗后统计
	report_lines.append("[6] 清洗后统计")
	after_len_stats = length_distribution_info(clean_df["text"])
	report_lines.append("清洗后长度统计: " + format_stats(after_len_stats))

	cat_dist = clean_df["category"].value_counts().to_dict()
	label_dist = clean_df["label"].value_counts().sort_index().to_dict()
	report_lines.append(f"category 分布: {cat_dist}")
	report_lines.append(f"label 分布: {label_dist}")
	report_lines.append("")

	# 输出
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	report_path.parent.mkdir(parents=True, exist_ok=True)
	clean_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
	report_path.write_text("\n".join(report_lines), encoding="utf-8")

	print("\n===== 阶段1：数据检查完成 =====")
	print(f"清洗后样本数: {len(clean_df)}")
	print(f"清洗 CSV: {output_csv}")
	print(f"报告文件: {report_path}")


if __name__ == "__main__":
	main()
