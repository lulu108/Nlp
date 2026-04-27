#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查并修复 THUCNews 目录编码问题。

支持功能：
1) 检查 zip 内类别目录是否完整
2) 检查已解压目录的类别与样本数
3) 尝试重命名已解压目录中的乱码目录名
4) 从 zip 按正确目录名重新解压（推荐）

示例：
python P2/check_fix_thucnews.py --check-only
python P2/check_fix_thucnews.py --fix-rename
python P2/check_fix_thucnews.py --extract-fixed
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP = BASE_DIR / "THUCNews.zip"
DEFAULT_EXTRACTED_ROOT = BASE_DIR / "THUCNews" / "THUCNews"
DEFAULT_FIXED_ROOT = BASE_DIR / "THUCNews_fixed" / "THUCNews"

EXPECTED_CATEGORIES = {
    "体育", "娱乐", "家居", "彩票", "房产", "教育", "时尚", "时政",
    "星座", "游戏", "社会", "科技", "股票", "财经",
}


def maybe_decode_segment(seg: str) -> str:
    """
    尝试将乱码目录名恢复为中文。
    主要针对 zip 中 cp437/utf-8 误解码导致的乱码。
    """
    candidates = [seg]

    # 常见 zip 乱码恢复：cp437 -> utf-8
    try:
        candidates.append(seg.encode("cp437").decode("utf-8"))
    except Exception:
        pass

    # 已经落盘后的中文乱码（如 浣撹偛）：gbk -> utf-8
    try:
        candidates.append(seg.encode("gbk").decode("utf-8"))
    except Exception:
        pass

    def score(s: str) -> int:
        cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
        printable = sum(1 for ch in s if ch.isprintable())
        return cjk * 10 + printable

    best = max(candidates, key=score)
    return best


def normalize_zip_path(name: str) -> str:
    parts = name.split("/")
    fixed_parts = [maybe_decode_segment(p) for p in parts]
    return "/".join(fixed_parts)


def inspect_zip_categories(zip_path: Path):
    if not zip_path.exists():
        raise FileNotFoundError("zip 不存在: {}".format(zip_path))

    categories = set()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for n in zf.namelist():
            if not n or n.startswith("__MACOSX/"):
                continue
            fixed = normalize_zip_path(n)
            if not fixed.startswith("THUCNews/"):
                continue
            parts = fixed.split("/")
            if len(parts) >= 2 and parts[1]:
                categories.add(parts[1])

    # 过滤掉意外项
    categories = {c for c in categories if c not in {"THUCNews", ""}}
    return categories


def inspect_extracted_categories(root: Path):
    if not root.exists():
        return {}
    stats = {}
    for d in root.iterdir():
        if d.is_dir():
            stats[d.name] = sum(1 for _ in d.glob("*.txt"))
    return stats


def print_check_report(zip_categories, extracted_stats):
    print("===== THUCNews 检查报告 =====")
    print("zip 类别数: {}".format(len(zip_categories)))
    print("zip 类别: {}".format(sorted(zip_categories)))

    missing = sorted(EXPECTED_CATEGORIES - zip_categories)
    extra = sorted(zip_categories - EXPECTED_CATEGORIES)

    print("缺失标准类别: {}".format(missing if missing else "无"))
    print("额外类别: {}".format(extra if extra else "无"))

    print("\n已解压目录统计:")
    if not extracted_stats:
        print("- 未找到已解压目录")
    else:
        for k in sorted(extracted_stats):
            print("- {}: {}".format(k, extracted_stats[k]))


def fix_rename_extracted(root: Path):
    if not root.exists():
        raise FileNotFoundError("已解压目录不存在: {}".format(root))

    renamed = []
    for d in list(root.iterdir()):
        if not d.is_dir():
            continue
        fixed_name = maybe_decode_segment(d.name)
        if fixed_name != d.name:
            target = d.parent / fixed_name
            if target.exists():
                print("跳过重命名（目标已存在）: {} -> {}".format(d.name, fixed_name))
                continue
            d.rename(target)
            renamed.append((d.name, fixed_name))

    print("重命名完成，共 {} 个目录。".format(len(renamed)))
    for old, new in renamed:
        print("- {} -> {}".format(old, new))


def extract_fixed_from_zip(zip_path: Path, output_root: Path):
    if not zip_path.exists():
        raise FileNotFoundError("zip 不存在: {}".format(zip_path))

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    file_count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name or name.endswith("/"):
                continue
            if name.startswith("__MACOSX/"):
                continue

            fixed = normalize_zip_path(name)
            if not fixed.startswith("THUCNews/"):
                continue

            # 只保留 THUCNews 下的真实文件
            rel = fixed[len("THUCNews/"):]
            if not rel:
                continue

            dst = output_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(name, "r") as src, open(dst, "wb") as out:
                out.write(src.read())
            file_count += 1

    print("重新解压完成: {}".format(output_root))
    print("写入文件数: {}".format(file_count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=str(DEFAULT_ZIP), help="THUCNews.zip 路径")
    parser.add_argument("--root", default=str(DEFAULT_EXTRACTED_ROOT), help="已解压 THUCNews 根目录")
    parser.add_argument("--fixed-root", default=str(DEFAULT_FIXED_ROOT), help="修复后输出目录")
    parser.add_argument("--check-only", action="store_true", help="仅检查")
    parser.add_argument("--fix-rename", action="store_true", help="尝试修复已解压目录名")
    parser.add_argument("--extract-fixed", action="store_true", help="从 zip 重新按正确编码解压")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    root = Path(args.root)
    fixed_root = Path(args.fixed_root)

    zip_categories = inspect_zip_categories(zip_path)
    extracted_stats = inspect_extracted_categories(root)
    print_check_report(zip_categories, extracted_stats)

    if args.fix_rename:
        print("\n===== 修复已解压目录名 =====")
        fix_rename_extracted(root)
        extracted_stats = inspect_extracted_categories(root)
        print("修复后目录统计:")
        for k in sorted(extracted_stats):
            print("- {}: {}".format(k, extracted_stats[k]))

    if args.extract_fixed:
        print("\n===== 从 zip 重新正确解压 =====")
        extract_fixed_from_zip(zip_path, fixed_root)
        fixed_stats = inspect_extracted_categories(fixed_root)
        print("修复解压后目录统计:")
        for k in sorted(fixed_stats):
            print("- {}: {}".format(k, fixed_stats[k]))


if __name__ == "__main__":
    main()
