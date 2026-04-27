from __future__ import annotations
# python scripts/evaluate_cluster.py --data-path data/train/classify_train.csv --grid-search --use-svd --svd-components 100
import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.classifier_preprocessing import build_classifier_text
from algorithms.cluster import DEFAULT_MAX_FEATURES, DEFAULT_NGRAM_RANGE, RANDOM_STATE

DEFAULT_DATA_PATH = ROOT_DIR / "data" / "train" / "classify_train.csv"
DEFAULT_GRID_CLUSTER_COUNTS = [3, 4, 5, 6]
DEFAULT_GRID_MAX_FEATURES = [2000, 3000, 5000]
DEFAULT_GRID_NGRAM_RANGES = [(1, 1), (1, 2)]
DEFAULT_GRID_MIN_DF = [1, 2]
DEFAULT_GRID_MAX_DF = [0.8, 1.0]
DEFAULT_GRID_SUBLINEAR_TF = [False, True]
DEFAULT_SVD_COMPONENTS = 100
DEFAULT_COMPARE_SVD_COMPONENTS = [50, 100, 200]


@dataclass(frozen=True)
class ClusterConfig:
    cluster_count: int
    max_features: int | None
    ngram_range: tuple[int, int]
    min_df: int | float
    max_df: int | float
    sublinear_tf: bool
    use_svd: bool = False
    svd_components: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P4 text clustering on labeled CSV data")
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="CSV path with at least text,label columns",
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=None,
        help="Single-run cluster count. Defaults to the number of unique labels.",
    )
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--ngram-range", default=f"{DEFAULT_NGRAM_RANGE[0]},{DEFAULT_NGRAM_RANGE[1]}")
    parser.add_argument("--min-df", default="1")
    parser.add_argument("--max-df", default="1.0")
    parser.add_argument("--sublinear-tf", action="store_true")
    parser.add_argument("--use-svd", action="store_true", help="Use TF-IDF -> TruncatedSVD -> KMeans")
    parser.add_argument("--svd-components", type=int, default=DEFAULT_SVD_COMPONENTS)
    parser.add_argument(
        "--compare-svd",
        action="store_true",
        help="Compare baseline against several SVD component settings",
    )
    parser.add_argument(
        "--compare-svd-components",
        default="50,100,200",
        help="Comma list used by --compare-svd, e.g. 50,100,200",
    )

    parser.add_argument("--grid-search", action="store_true", help="Run parameter search instead of one config")
    parser.add_argument("--cluster-counts", default=None, help="Comma list, e.g. 3,4,5,6")
    parser.add_argument("--max-features-list", default=None, help="Comma list, e.g. 2000,3000,5000")
    parser.add_argument("--ngram-ranges", default=None, help="Comma list, e.g. 1-1,1-2")
    parser.add_argument("--min-df-list", default=None, help="Comma list, e.g. 1,2")
    parser.add_argument("--max-df-list", default=None, help="Comma list, e.g. 0.8,1.0")
    parser.add_argument("--sublinear-tf-list", default=None, help="Comma list, e.g. false,true")
    parser.add_argument("--top-n", type=int, default=10, help="Rows to show in the ranked grid summary")
    return parser.parse_args()


def parse_df_value(value: str) -> int | float:
    raw = value.strip()
    if not raw:
        raise ValueError("df value cannot be empty")

    if "." in raw:
        parsed = float(raw)
        if not 0.0 < parsed <= 1.0:
            raise ValueError("float df values must satisfy 0 < value <= 1")
        return parsed

    parsed_int = int(raw)
    if parsed_int < 1:
        raise ValueError("integer df values must be >= 1")
    return parsed_int


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return default
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("integer list cannot be empty")
    return items


def parse_df_list(value: str | None, default: list[int | float]) -> list[int | float]:
    if value is None:
        return default
    items = [parse_df_value(item) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("df list cannot be empty")
    return items


def parse_bool_list(value: str | None, default: list[bool]) -> list[bool]:
    if value is None:
        return default

    mapping = {
        "1": True,
        "true": True,
        "yes": True,
        "y": True,
        "0": False,
        "false": False,
        "no": False,
        "n": False,
    }
    items: list[bool] = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"invalid boolean value: {item}")
        items.append(mapping[key])

    if not items:
        raise ValueError("boolean list cannot be empty")
    return items


def parse_ngram_range(value: str) -> tuple[int, int]:
    parts = value.replace("-", ",").split(",")
    if len(parts) != 2:
        raise ValueError("ngram range must look like 1,2 or 1-2")

    start = int(parts[0].strip())
    end = int(parts[1].strip())
    if start < 1 or end < start:
        raise ValueError("ngram range must satisfy 1 <= start <= end")
    return (start, end)


def parse_ngram_ranges(value: str | None, default: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if value is None:
        return default
    items = [parse_ngram_range(item) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("ngram range list cannot be empty")
    return items


def load_csv_auto(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as err:  # noqa: BLE001
            last_err = err
    raise ValueError(f"failed to read CSV: {path}, last error: {last_err}")


def load_evaluation_dataframe(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"evaluation data not found: {data_path}")

    df = load_csv_auto(data_path)
    if "text" not in df.columns:
        raise ValueError("evaluation data must include 'text' column")
    if "label" not in df.columns:
        raise ValueError("evaluation data must include 'label' column")

    text_series = df["text"].fillna("").astype(str)
    if "title" in df.columns:
        title_series = df["title"].fillna("").astype(str)
        processed_texts = [build_classifier_text(title, text) for title, text in zip(title_series, text_series)]
    else:
        title_series = pd.Series([""] * len(df))
        processed_texts = text_series.map(build_classifier_text)

    out = pd.DataFrame(
        {
            "title": title_series.astype(str).str.strip(),
            "text": text_series.astype(str).str.strip(),
            "label": df["label"].astype(str).str.strip(),
            "processed_text": processed_texts,
        }
    )

    out = out[(out["label"] != "") & (out["processed_text"] != "")].reset_index(drop=True)
    if out.empty:
        raise ValueError("evaluation data is empty after cleaning")
    if len(out) < 2:
        raise ValueError("at least two valid documents are required for cluster evaluation")

    label_count = int(out["label"].nunique())
    if label_count < 2:
        raise ValueError("at least two unique labels are required for cluster evaluation")

    return out


def resolve_cluster_count(dataframe: pd.DataFrame, cluster_count: int | None) -> int:
    resolved = int(dataframe["label"].nunique()) if cluster_count is None else int(cluster_count)
    validate_cluster_count(resolved, len(dataframe))
    return resolved


def validate_cluster_count(cluster_count: int, document_count: int) -> None:
    if cluster_count < 2:
        raise ValueError("cluster_count must be >= 2")
    if cluster_count > document_count:
        raise ValueError("cluster_count must be <= document count")


def resolve_svd_components(requested: int, features_shape: tuple[int, int]) -> int:
    n_samples, n_features = features_shape
    max_allowed = min(n_samples - 1, n_features - 1)

    if requested < 2:
        raise ValueError("svd_components must be >= 2")
    if max_allowed < 2:
        raise ValueError("not enough features or samples to apply TruncatedSVD")
    if requested > max_allowed:
        raise ValueError(f"svd_components must be <= {max_allowed} for the current data")

    return requested


def calculate_purity(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    contingency = pd.crosstab(
        pd.Series(predicted_labels, name="cluster"),
        pd.Series(true_labels, name="label"),
    )
    return float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())


def summarize_clusters(true_labels: np.ndarray, predicted_labels: np.ndarray) -> list[dict[str, object]]:
    contingency = pd.crosstab(
        pd.Series(predicted_labels, name="cluster"),
        pd.Series(true_labels, name="label"),
    )
    summary: list[dict[str, object]] = []

    for cluster_id, row in contingency.iterrows():
        cluster_size = int(row.sum())
        dominant_label = str(row.idxmax())
        dominant_count = int(row.max())
        summary.append(
            {
                "cluster": int(cluster_id),
                "size": cluster_size,
                "dominant_label": dominant_label,
                "dominant_count": dominant_count,
                "dominant_ratio": dominant_count / cluster_size if cluster_size > 0 else 0.0,
            }
        )

    return summary


def evaluate_clustering(dataframe: pd.DataFrame, config: ClusterConfig) -> dict[str, object]:
    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        token_pattern=r"(?u)\b\w+\b",
    )
    tfidf_features = vectorizer.fit_transform(dataframe["processed_text"])

    kmeans_features = tfidf_features
    silhouette_features = tfidf_features
    effective_svd_components: int | None = None

    if config.use_svd:
        effective_svd_components = resolve_svd_components(
            int(config.svd_components or DEFAULT_SVD_COMPONENTS),
            tfidf_features.shape,
        )
        svd = TruncatedSVD(n_components=effective_svd_components, random_state=RANDOM_STATE)
        svd_features = svd.fit_transform(tfidf_features)
        kmeans_features = svd_features
        silhouette_features = svd_features

    model = KMeans(n_clusters=config.cluster_count, n_init=10, random_state=RANDOM_STATE)
    predicted_labels = model.fit_predict(kmeans_features)
    true_labels = dataframe["label"].to_numpy()

    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    unique_clusters = np.unique(predicted_labels)
    if 1 < len(unique_clusters) < len(dataframe):
        silhouette = float(silhouette_score(silhouette_features, predicted_labels))
    else:
        silhouette = float("nan")

    purity = calculate_purity(true_labels, predicted_labels)
    cluster_summary = summarize_clusters(true_labels, predicted_labels)

    return {
        "pipeline": "tfidf_svd_kmeans" if config.use_svd else "tfidf_kmeans",
        "n_documents": int(len(dataframe)),
        "n_labels": int(dataframe["label"].nunique()),
        "cluster_count": int(config.cluster_count),
        "max_features": config.max_features,
        "ngram_range": config.ngram_range,
        "min_df": config.min_df,
        "max_df": config.max_df,
        "sublinear_tf": config.sublinear_tf,
        "use_svd": config.use_svd,
        "svd_components": effective_svd_components,
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "ari": float(ari),
        "nmi": float(nmi),
        "silhouette": silhouette,
        "purity": float(purity),
        "cluster_summary": cluster_summary,
    }


def build_single_config(dataframe: pd.DataFrame, args: argparse.Namespace) -> ClusterConfig:
    return ClusterConfig(
        cluster_count=resolve_cluster_count(dataframe, args.cluster_count),
        max_features=args.max_features,
        ngram_range=parse_ngram_range(args.ngram_range),
        min_df=parse_df_value(args.min_df),
        max_df=parse_df_value(args.max_df),
        sublinear_tf=bool(args.sublinear_tf),
        use_svd=bool(args.use_svd),
        svd_components=int(args.svd_components) if args.use_svd else None,
    )


def build_grid_configs(dataframe: pd.DataFrame, args: argparse.Namespace) -> list[ClusterConfig]:
    cluster_counts = parse_int_list(args.cluster_counts, DEFAULT_GRID_CLUSTER_COUNTS)
    max_features_values = parse_int_list(args.max_features_list, DEFAULT_GRID_MAX_FEATURES)
    ngram_ranges = parse_ngram_ranges(args.ngram_ranges, DEFAULT_GRID_NGRAM_RANGES)
    min_df_values = parse_df_list(args.min_df_list, DEFAULT_GRID_MIN_DF)
    max_df_values = parse_df_list(args.max_df_list, DEFAULT_GRID_MAX_DF)
    sublinear_tf_values = parse_bool_list(args.sublinear_tf_list, DEFAULT_GRID_SUBLINEAR_TF)

    configs: list[ClusterConfig] = []
    for cluster_count, max_features, ngram_range, min_df, max_df, sublinear_tf in itertools.product(
        cluster_counts,
        max_features_values,
        ngram_ranges,
        min_df_values,
        max_df_values,
        sublinear_tf_values,
    ):
        validate_cluster_count(cluster_count, len(dataframe))
        configs.append(
            ClusterConfig(
                cluster_count=cluster_count,
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=sublinear_tf,
                use_svd=bool(args.use_svd),
                svd_components=int(args.svd_components) if args.use_svd else None,
            )
        )

    return configs


def build_compare_configs(dataframe: pd.DataFrame, args: argparse.Namespace) -> list[ClusterConfig]:
    base = build_single_config(dataframe, args)
    svd_components_list = parse_int_list(args.compare_svd_components, DEFAULT_COMPARE_SVD_COMPONENTS)

    configs = [
        ClusterConfig(
            cluster_count=base.cluster_count,
            max_features=base.max_features,
            ngram_range=base.ngram_range,
            min_df=base.min_df,
            max_df=base.max_df,
            sublinear_tf=base.sublinear_tf,
            use_svd=False,
            svd_components=None,
        )
    ]

    for components in svd_components_list:
        configs.append(
            ClusterConfig(
                cluster_count=base.cluster_count,
                max_features=base.max_features,
                ngram_range=base.ngram_range,
                min_df=base.min_df,
                max_df=base.max_df,
                sublinear_tf=base.sublinear_tf,
                use_svd=True,
                svd_components=components,
            )
        )

    return configs


def sort_key(metrics: dict[str, object]) -> tuple[float, float, float]:
    silhouette = float(metrics["silhouette"])
    if np.isnan(silhouette):
        silhouette = -1.0
    return (float(metrics["ari"]), float(metrics["nmi"]), silhouette)


def print_summary(metrics: dict[str, object]) -> None:
    print("===== Cluster Evaluation Summary =====")
    print(f"Documents: {metrics['n_documents']}")
    print(f"Unique labels: {metrics['n_labels']}")
    print_config(metrics)
    print(f"Vocabulary size: {metrics['vocabulary_size']}")
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")

    silhouette = float(metrics["silhouette"])
    if np.isnan(silhouette):
        print("Silhouette Score: nan")
    else:
        print(f"Silhouette Score: {silhouette:.4f}")

    print(f"Purity: {metrics['purity']:.4f}")
    print("Per-cluster dominant labels:")

    for item in metrics["cluster_summary"]:
        print(
            "  "
            f"cluster={item['cluster']} | "
            f"size={item['size']} | "
            f"dominant_label={item['dominant_label']} | "
            f"dominant_count={item['dominant_count']} | "
            f"dominant_ratio={item['dominant_ratio']:.4f}"
        )


def print_config(metrics: dict[str, object]) -> None:
    ngram_range = metrics["ngram_range"]
    print(
        "Config: "
        f"pipeline={metrics['pipeline']}, "
        f"cluster_count={metrics['cluster_count']}, "
        f"max_features={metrics['max_features']}, "
        f"ngram_range={ngram_range[0]}-{ngram_range[1]}, "
        f"min_df={metrics['min_df']}, "
        f"max_df={metrics['max_df']}, "
        f"sublinear_tf={metrics['sublinear_tf']}, "
        f"svd_components={metrics['svd_components']}"
    )


def print_grid_results(results: list[dict[str, object]], top_n: int) -> None:
    ranked = sorted(results, key=sort_key, reverse=True)
    shown = ranked[: max(1, top_n)]

    print("===== Cluster Grid Search Summary =====")
    print(f"Total configs: {len(results)}")
    print(f"Showing top {len(shown)} configs sorted by ARI, then NMI, then Silhouette")
    print(
        "rank | ari    | nmi    | sil    | purity | pipeline        | k | max_features | "
        "ngram | min_df | max_df | sublinear_tf | svd | vocab"
    )

    for rank, item in enumerate(shown, start=1):
        ngram_range = item["ngram_range"]
        silhouette = float(item["silhouette"])
        silhouette_text = "nan" if np.isnan(silhouette) else f"{silhouette:.4f}"
        svd_text = "-" if item["svd_components"] is None else str(item["svd_components"])
        print(
            f"{rank:>4} | "
            f"{item['ari']:.4f} | "
            f"{item['nmi']:.4f} | "
            f"{silhouette_text:>6} | "
            f"{item['purity']:.4f} | "
            f"{item['pipeline']:<15} | "
            f"{item['cluster_count']} | "
            f"{item['max_features']} | "
            f"{ngram_range[0]}-{ngram_range[1]} | "
            f"{item['min_df']} | "
            f"{item['max_df']} | "
            f"{item['sublinear_tf']} | "
            f"{svd_text} | "
            f"{item['vocabulary_size']}"
        )

    print()
    print("===== Best Configuration =====")
    print_summary(ranked[0])


def print_svd_comparison(results: list[dict[str, object]]) -> None:
    print("===== Baseline vs SVD Comparison =====")
    print("pipeline        | svd_components | ari    | nmi    | sil    | purity | vocab")

    for item in results:
        silhouette = float(item["silhouette"])
        silhouette_text = "nan" if np.isnan(silhouette) else f"{silhouette:.4f}"
        svd_text = "-" if item["svd_components"] is None else str(item["svd_components"])
        print(
            f"{item['pipeline']:<15} | "
            f"{svd_text:>14} | "
            f"{item['ari']:.4f} | "
            f"{item['nmi']:.4f} | "
            f"{silhouette_text:>6} | "
            f"{item['purity']:.4f} | "
            f"{item['vocabulary_size']}"
        )

    ranked = sorted(results, key=sort_key, reverse=True)
    print()
    print("===== Best Comparison Result =====")
    print_summary(ranked[0])


def run_grid_search(dataframe: pd.DataFrame, args: argparse.Namespace) -> None:
    configs = build_grid_configs(dataframe, args)
    results: list[dict[str, object]] = []

    for idx, config in enumerate(configs, start=1):
        print(f"Evaluating config {idx}/{len(configs)}...", flush=True)
        results.append(evaluate_clustering(dataframe, config))

    print_grid_results(results, args.top_n)


def run_svd_comparison(dataframe: pd.DataFrame, args: argparse.Namespace) -> None:
    configs = build_compare_configs(dataframe, args)
    results = [evaluate_clustering(dataframe, config) for config in configs]
    print_svd_comparison(results)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    dataframe = load_evaluation_dataframe(data_path)

    if args.grid_search:
        run_grid_search(dataframe, args)
        return

    if args.compare_svd:
        run_svd_comparison(dataframe, args)
        return

    config = build_single_config(dataframe, args)
    metrics = evaluate_clustering(dataframe, config)
    print_summary(metrics)


if __name__ == "__main__":
    main()
