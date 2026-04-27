# -*- coding: utf-8 -*-
"""
功能：
1. 读取 output/top_error_words.txt（03train_hmm.py 导出的 Top 错词）
2. 自动将新词追加到 02build_bmes_dataset.py 的 USER_TERMS 和 MERGE_RULES
3. 对已存在的词跳过，避免重复

用法：
    python P1/test2/04update_dict.py
"""

import re
from pathlib import Path

TOP_ERRORS_PATH = Path("./P1/test2/output/top_error_words.txt")
DATASET_SCRIPT = Path("./P1/test2/02build_bmes_dataset.py")


def load_top_errors():
    if not TOP_ERRORS_PATH.exists():
        print(f"未找到 Top 错词文件: {TOP_ERRORS_PATH}")
        print("请先运行 03train_hmm.py 生成该文件。")
        return []
    words = [w.strip() for w in TOP_ERRORS_PATH.read_text(encoding="utf-8").splitlines() if w.strip()]
    print(f"读取 Top 错词 {len(words)} 个: {words}")
    return words


def extract_existing_user_terms(source: str):
    """从源码中提取已有的 USER_TERMS 词条"""
    matches = re.findall(r'USER_TERMS\s*=\s*\[(.*?)\]', source, re.DOTALL)
    if not matches:
        return set()
    existing = set(re.findall(r'"([^"]+)"', matches[0]))
    existing |= set(re.findall(r"'([^']+)'", matches[0]))
    return existing


def extract_existing_merge_rules(source: str):
    """从源码中提取已有的 MERGE_RULES 目标词"""
    matches = re.findall(r'MERGE_RULES\s*=\s*\[(.*?)\]', source, re.DOTALL)
    if not matches:
        return set()
    # 每条规则末尾的目标词: ("...", "...", ...), "TARGET"
    targets = set(re.findall(r'\),\s*"([^"]+)"\s*\)', matches[0]))
    return targets


def split_word_to_merge_rule(word: str):
    """
    将词拆成最简单的 2-gram 合并规则。
    例如 "党中央" -> (("党中", "央"), "党中央")
    对于 2 字词直接拆成两个单字。
    """
    if len(word) == 2:
        tokens = tuple(word)
    else:
        # 前 n-1 字 + 最后 1 字
        tokens = (word[:-1], word[-1])
    return tokens, word


def append_new_words(words):
    source = DATASET_SCRIPT.read_text(encoding="utf-8")

    existing_terms = extract_existing_user_terms(source)
    existing_rules = extract_existing_merge_rules(source)

    new_terms = [w for w in words if w not in existing_terms]
    new_rules = [w for w in words if w not in existing_rules and len(w) >= 2]

    if not new_terms and not new_rules:
        print("所有 Top 错词已存在于词典中，无需更新。")
        return

    # 追加到 USER_TERMS
    if new_terms:
        terms_block = "\n".join(f'    "{w}",' for w in new_terms)
        terms_comment = "    # === 自动回灌 Top 错词 ==="
        insert_marker = re.search(r'(USER_TERMS\s*=\s*\[.*?)(]\s*\n)', source, re.DOTALL)
        if insert_marker:
            insert_pos = insert_marker.end(1)
            source = source[:insert_pos] + f"\n{terms_comment}\n{terms_block}\n" + source[insert_pos:]
            print(f"USER_TERMS 新增 {len(new_terms)} 词: {new_terms}")

    # 追加到 MERGE_RULES
    if new_rules:
        rules_lines = []
        for w in new_rules:
            tokens, target = split_word_to_merge_rule(w)
            token_str = ", ".join(f'"{t}"' for t in tokens)
            rules_lines.append(f'    (({token_str}), "{target}"),')
        rules_block = "\n".join(rules_lines)
        rules_comment = "    # === 自动回灌 Top 错词 ==="
        insert_marker = re.search(r'(MERGE_RULES\s*=\s*\[.*?)(]\s*\n)', source, re.DOTALL)
        if insert_marker:
            insert_pos = insert_marker.end(1)
            source = source[:insert_pos] + f"\n{rules_comment}\n{rules_block}\n" + source[insert_pos:]
            print(f"MERGE_RULES 新增 {len(new_rules)} 条规则: {new_rules}")

    DATASET_SCRIPT.write_text(source, encoding="utf-8")
    print(f"已更新: {DATASET_SCRIPT}")
    print("请重新运行 02build_bmes_dataset.py 重建数据集，再运行 03train_hmm.py 训练。")


def main():
    words = load_top_errors()
    if words:
        append_new_words(words)


if __name__ == "__main__":
    main()
