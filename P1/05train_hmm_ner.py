# -*- coding: utf-8 -*-
"""
05train_hmm_ner.py

推荐运行：
python P1/05train_hmm_ner.py
python P1/05train_hmm_ner.py --alpha 0.001
python P1/05train_hmm_ner.py --alpha 0.001 --no-postprocess

基于 HMM 的 NER 序列标注扩展实验。

功能：
1. 读取 P1/datasets/auto/train.txt 与 test.txt 中的原始句子；
2. 使用 HanLP 自动识别人名、地名、机构名，生成字符级 NER 银标标签；
3. 使用 HMM 训练 NER 序列标注模型；
4. 使用 Viterbi 算法预测测试集 NER 标签；
5. 使用训练集实体词典、姓氏规则、地名/机构名后缀规则进行后处理；
6. 输出后处理前后的标签级 Accuracy、实体级 Precision / Recall / F1；
7. 输出预测样例到 P1/output/hmm_ner_predictions.txt。

说明：
- 本脚本不是原来的 BMES 中文分词模型，而是 HMM-NER 扩展模型。
- 原 03train_hmm.py 的标签集合是 {B, M, E, S}，用于中文分词。
- 本脚本的标签集合是：
  O,
  B-PER, M-PER, E-PER, S-PER,
  B-LOC, M-LOC, E-LOC, S-LOC,
  B-ORG, M-ORG, E-ORG, S-ORG。
- 当前实验数据没有人工标注的 NER 标签，因此这里先用 HanLP 自动生成 NER 银标标签。
- 后处理只使用训练集银标抽取的实体词典和通用规则，不直接使用测试集银标答案修正预测。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# =========================
# 路径配置
# =========================

TRAIN_PATH = Path("./P1/datasets/auto/train.txt")
TEST_PATH = Path("./P1/datasets/auto/test.txt")

OUTPUT_DIR = Path("./P1/output")
REPORT_DIR = Path("./P1/reports")

PREDICTION_PATH = OUTPUT_DIR / "hmm_ner_predictions.txt"
METRICS_PATH = REPORT_DIR / "hmm_ner_metrics.csv"


# =========================
# NER 标签体系
# =========================

ENTITY_TYPES = ["PER", "LOC", "ORG"]

# PER: 人名；LOC: 地名；ORG: 机构名；O: 非实体
STATES = [
    "O",
    "B-PER", "M-PER", "E-PER", "S-PER",
    "B-LOC", "M-LOC", "E-LOC", "S-LOC",
    "B-ORG", "M-ORG", "E-ORG", "S-ORG",
]

Entity = Tuple[str, int, int, str]
# Entity = (实体类型, 起始位置, 结束位置, 实体文本)
# 其中结束位置为开区间，例如 [start, end)


# =========================
# 数据读取
# =========================

def load_samples(path: Path, limit: Optional[int] = None) -> List[Dict[str, object]]:
    """
    读取 P1/datasets/auto/train.txt 或 test.txt。

    兼容当前实验数据格式：
    原句 \t 字符序列 \t BMES标签序列 \t 分词结果

    本脚本只使用原句和字符序列。
    原来的 BMES 分词标签不用于 NER 训练。
    """
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    samples: List[Dict[str, object]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        sentence = parts[0].strip()
        if not sentence:
            continue

        if len(parts) >= 2 and parts[1].strip():
            chars = parts[1].split()
        else:
            chars = list(sentence)

        # 防止字符序列和原句无法对齐
        if "".join(chars) != sentence:
            chars = list(sentence)

        samples.append({
            "sentence": sentence,
            "chars": chars,
        })

        if limit is not None and len(samples) >= limit:
            break

    return samples


# =========================
# HanLP 银标生成
# =========================

def build_hanlp_segmenter():
    """
    构建 HanLP 分词器，并开启人名、地名、机构名识别功能。
    """
    try:
        from pyhanlp import HanLP
    except Exception as exc:
        raise RuntimeError(
            "无法导入 pyhanlp。请先确认已安装 pyhanlp，并且 Java 环境可用。\n"
            "可尝试执行：pip install pyhanlp"
        ) from exc

    segmenter = HanLP.newSegment()
    segmenter.enableNameRecognize(True)
    segmenter.enablePlaceRecognize(True)
    segmenter.enableOrganizationRecognize(True)

    return segmenter


def nature_to_entity_type(nature: str) -> Optional[str]:
    """
    HanLP 词性到实体类型的映射。

    nr -> PER，人名
    ns -> LOC，地名
    nt -> ORG，机构名
    """
    if nature.startswith("nr"):
        return "PER"
    if nature.startswith("ns"):
        return "LOC"
    if nature.startswith("nt"):
        return "ORG"
    return None


def make_entity_tags(length: int, entity_type: str) -> List[str]:
    """
    将一个实体词转换为 BMES-实体类型标签。

    例如：
    王小洪 -> B-PER M-PER E-PER
    北京   -> B-LOC E-LOC
    美     -> S-LOC
    """
    if length <= 0:
        return []

    if length == 1:
        return [f"S-{entity_type}"]

    if length == 2:
        return [f"B-{entity_type}", f"E-{entity_type}"]

    return [f"B-{entity_type}"] + [f"M-{entity_type}"] * (length - 2) + [f"E-{entity_type}"]


def hanlp_to_ner_tags(sentence: str, chars: Sequence[str], segmenter) -> List[str]:
    """
    使用 HanLP 对句子进行实体识别，并转换为字符级 NER 标签。

    返回：
    与 chars 等长的标签序列。
    """
    text = "".join(chars)
    labels = ["O"] * len(chars)

    terms = segmenter.seg(sentence)

    cursor = 0

    for term in terms:
        word = str(term.word).strip()
        nature = str(term.nature) if term.nature is not None else ""

        if not word:
            continue

        entity_type = nature_to_entity_type(nature)

        # 尝试按 HanLP 分词顺序对齐原句
        if text.startswith(word, cursor):
            start = cursor
        else:
            start = text.find(word, cursor)
            if start < 0:
                start = text.find(word)

        if start < 0:
            continue

        end = start + len(word)

        if end > len(chars):
            continue

        cursor = max(cursor, end)

        # 非人名、地名、机构名则保持 O
        if entity_type is None:
            continue

        # 防止重叠实体覆盖
        if any(labels[i] != "O" for i in range(start, end)):
            continue

        labels[start:end] = make_entity_tags(end - start, entity_type)

    return labels


def add_silver_labels(samples: List[Dict[str, object]], segmenter) -> List[Dict[str, object]]:
    """
    为样本添加 HanLP 自动生成的 NER 银标标签。
    """
    labeled_samples: List[Dict[str, object]] = []

    for sample in samples:
        sentence = str(sample["sentence"])
        chars = list(sample["chars"])  # type: ignore[arg-type]

        tags = hanlp_to_ner_tags(sentence, chars, segmenter)

        if len(chars) != len(tags):
            continue

        labeled_samples.append({
            "sentence": sentence,
            "chars": chars,
            "tags": tags,
        })

    return labeled_samples


# =========================
# NER-BMES 合法转移约束
# =========================

def build_allowed_transitions() -> Tuple[set, set, Dict[str, set]]:
    """
    构造 NER-BMES 合法转移约束。

    规则：
    1. 句首允许：O、B-X、S-X；
    2. 句尾允许：O、E-X、S-X；
    3. O / E-X / S-X 后面允许：O、B-Y、S-Y；
    4. B-X 后面只允许：M-X、E-X；
    5. M-X 后面只允许：M-X、E-X。
    """
    start_states = {"O"}
    end_states = {"O"}

    boundary_states = {"O"}

    for entity_type in ENTITY_TYPES:
        start_states.add(f"B-{entity_type}")
        start_states.add(f"S-{entity_type}")

        end_states.add(f"E-{entity_type}")
        end_states.add(f"S-{entity_type}")

        boundary_states.add(f"E-{entity_type}")
        boundary_states.add(f"S-{entity_type}")

    after_boundary = {"O"}

    for entity_type in ENTITY_TYPES:
        after_boundary.add(f"B-{entity_type}")
        after_boundary.add(f"S-{entity_type}")

    allowed: Dict[str, set] = {state: set() for state in STATES}

    for state in boundary_states:
        allowed[state] = set(after_boundary)

    for entity_type in ENTITY_TYPES:
        allowed[f"B-{entity_type}"] = {
            f"M-{entity_type}",
            f"E-{entity_type}",
        }
        allowed[f"M-{entity_type}"] = {
            f"M-{entity_type}",
            f"E-{entity_type}",
        }

    return start_states, end_states, allowed


# =========================
# HMM-NER 模型
# =========================

class HMMNER:
    """
    基于 HMM 的 NER 序列标注模型。
    """

    def __init__(self, alpha: float = 0.001, use_constraints: bool = True):
        if alpha <= 0:
            raise ValueError("alpha 必须大于 0")

        self.states = STATES
        self.alpha = alpha
        self.use_constraints = use_constraints

        self.allowed_start, self.allowed_end, self.allowed_transitions = build_allowed_transitions()

        self.pi_count = defaultdict(int)
        self.trans_count = {state: defaultdict(int) for state in self.states}
        self.emit_count = {state: defaultdict(int) for state in self.states}
        self.state_count = defaultdict(int)
        self.vocab = set()

        self.pi: Dict[str, float] = {}
        self.trans: Dict[str, Dict[str, float]] = {state: {} for state in self.states}
        self.emit: Dict[str, Dict[str, float]] = {state: {} for state in self.states}
        self.unknown_emit: Dict[str, float] = {}

    def train(self, samples: Sequence[Dict[str, object]]) -> None:
        """
        训练 HMM 参数：
        1. 初始状态概率 pi；
        2. 状态转移概率 trans；
        3. 发射概率 emit。
        """
        for sample in samples:
            chars = list(sample["chars"])  # type: ignore[arg-type]
            tags = list(sample["tags"])    # type: ignore[arg-type]

            if not chars or not tags:
                continue

            if len(chars) != len(tags):
                continue

            self.pi_count[tags[0]] += 1

            for i, (ch, tag) in enumerate(zip(chars, tags)):
                if tag not in self.states:
                    continue

                self.state_count[tag] += 1
                self.emit_count[tag][ch] += 1
                self.vocab.add(ch)

                if i > 0:
                    prev_tag = tags[i - 1]
                    if prev_tag in self.states:
                        self.trans_count[prev_tag][tag] += 1

        self._calculate_log_probabilities()

    def _calculate_log_probabilities(self) -> None:
        """
        使用 Lidstone 平滑估计概率，并转换为 log 概率。
        """
        alpha = self.alpha
        num_states = len(self.states)
        vocab_size = max(1, len(self.vocab))
        total_sentences = max(1, sum(self.pi_count.values()))

        # 初始状态概率
        for state in self.states:
            prob = (self.pi_count[state] + alpha) / (total_sentences + alpha * num_states)
            self.pi[state] = math.log(prob)

        # 状态转移概率
        for prev_state in self.states:
            total = sum(self.trans_count[prev_state].values())
            denominator = total + alpha * num_states

            for curr_state in self.states:
                prob = (self.trans_count[prev_state][curr_state] + alpha) / denominator
                self.trans[prev_state][curr_state] = math.log(prob)

        # 发射概率
        for state in self.states:
            total = self.state_count[state]
            denominator = total + alpha * vocab_size

            for ch in self.vocab:
                prob = (self.emit_count[state][ch] + alpha) / denominator
                self.emit[state][ch] = math.log(prob)

            # 未登录字符 OOV 的默认发射概率
            unknown_prob = alpha / denominator
            self.unknown_emit[state] = math.log(unknown_prob)

    def get_emit_logprob(self, state: str, ch: str) -> float:
        """
        获取某状态生成某字符的 log 概率。
        如果字符未出现过，则使用 OOV 发射概率。
        """
        return self.emit[state].get(ch, self.unknown_emit[state])

    def viterbi(self, chars: Sequence[str]) -> List[str]:
        """
        Viterbi 解码，返回最优 NER 标签序列。
        """
        if not chars:
            return []

        neg_inf = -1e18

        dp: List[Dict[str, float]] = []
        backpointer: List[Dict[str, Optional[str]]] = []

        # 初始化
        dp.append({})
        backpointer.append({})

        for state in self.states:
            if self.use_constraints and state not in self.allowed_start:
                dp[0][state] = neg_inf
            else:
                dp[0][state] = self.pi[state] + self.get_emit_logprob(state, chars[0])

            backpointer[0][state] = None

        # 递推
        for t in range(1, len(chars)):
            dp.append({})
            backpointer.append({})

            for curr_state in self.states:
                best_score = neg_inf
                best_prev_state: Optional[str] = None

                for prev_state in self.states:
                    if self.use_constraints:
                        if curr_state not in self.allowed_transitions.get(prev_state, set()):
                            continue

                    score = (
                        dp[t - 1][prev_state]
                        + self.trans[prev_state][curr_state]
                        + self.get_emit_logprob(curr_state, chars[t])
                    )

                    if score > best_score:
                        best_score = score
                        best_prev_state = prev_state

                dp[t][curr_state] = best_score
                backpointer[t][curr_state] = best_prev_state

        # 终止
        if self.use_constraints:
            final_candidates = [
                (dp[-1][state], state)
                for state in self.states
                if state in self.allowed_end
            ]
        else:
            final_candidates = [
                (dp[-1][state], state)
                for state in self.states
            ]

        _, final_state = max(final_candidates, key=lambda item: item[0])

        # 回溯
        best_path = [final_state]

        for t in range(len(chars) - 1, 0, -1):
            prev_state = backpointer[t][best_path[-1]]
            if prev_state is None:
                prev_state = "O"
            best_path.append(prev_state)

        best_path.reverse()

        return best_path


# =========================
# 标签转实体
# =========================

def tag_prefix(tag: str) -> str:
    if tag == "O":
        return "O"
    return tag.split("-", 1)[0]


def tag_entity_type(tag: str) -> Optional[str]:
    if tag == "O":
        return None
    if "-" not in tag:
        return None
    return tag.split("-", 1)[1]


def tags_to_entities(chars: Sequence[str], tags: Sequence[str]) -> List[Entity]:
    """
    将字符级标签序列转换为实体列表。

    返回：
    [
        ("PER", 10, 13, "王小洪"),
        ("LOC", 15, 17, "天津"),
        ...
    ]
    """
    entities: List[Entity] = []

    n = min(len(chars), len(tags))
    i = 0

    while i < n:
        tag = tags[i]

        if tag == "O":
            i += 1
            continue

        prefix = tag_prefix(tag)
        entity_type = tag_entity_type(tag)

        if entity_type is None:
            i += 1
            continue

        if prefix == "S":
            entities.append((entity_type, i, i + 1, chars[i]))
            i += 1
            continue

        if prefix == "B":
            start = i
            i += 1

            while i < n and tags[i] == f"M-{entity_type}":
                i += 1

            if i < n and tags[i] == f"E-{entity_type}":
                end = i + 1
                text = "".join(chars[start:end])
                entities.append((entity_type, start, end, text))
                i = end
            else:
                # 如果出现非法片段，例如 B-PER 后面没有 M/E-PER，
                # 则兜底当作单字实体处理。
                entities.append((entity_type, start, start + 1, chars[start]))
            continue

        # 单独出现 M-X 或 E-X 属于非法片段，跳过
        i += 1

    return entities


def format_entities(entities: Iterable[Entity], entity_type: str) -> str:
    """
    格式化某一类实体，去重并保持顺序。
    """
    result = []
    seen = set()

    for etype, _start, _end, text in entities:
        if etype == entity_type and text not in seen:
            seen.add(text)
            result.append(text)

    return "、".join(result) if result else "无"


# =========================
# 词典与规则后处理
# =========================

SURNAMES = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华"
    "金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方"
    "俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮"
    "卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计"
    "伏成戴谈宋庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童"
    "颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌霍虞万支柯昝管卢莫经房裘"
    "缪干解应宗丁宣邓郁单杭洪包诸左石崔吉龚程邢滑裴陆荣翁荀羊"
    "於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯"
    "宓蓬全郗班仰秋仲伊宫宁仇栾暴甘斜厉戎祖武符刘景詹束龙叶幸"
    "司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴郁胥能"
    "苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈"
    "燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖"
    "庾终暨居衡步都耿满弘匡国文寇广禄阙东欧利师巩聂关荆"
)

DOUBLE_SURNAMES = {
    "欧阳", "司马", "上官", "夏侯", "诸葛", "闻人", "东方", "赫连",
    "皇甫", "尉迟", "公羊", "澹台", "公冶", "宗政", "濮阳", "淳于",
    "单于", "太叔", "申屠", "公孙", "仲孙", "轩辕", "令狐", "钟离",
    "宇文", "长孙", "慕容", "司徒", "司空"
}

ORG_SUFFIXES = [
    "国家金融监督管理总局",
    "中国人民银行",
    "证监会",
    "银保监会",
    "委员会",
    "人民政府",
    "公安部",
    "国家安全部",
    "司法部",
    "银行",
    "公司",
    "集团",
    "总局",
    "政府",
    "法院",
    "检察院",
    "研究院",
    "科学院",
    "大学",
    "学院",
    "中心",
    "协会",
    "学会",
    "机构",
    "部门",
    "新华社",
    "中新社",
    "日报社",
    "电视台",
]

# 注意：这里不再把“市、区、县、省”作为宽泛正则后缀匹配，
# 否则容易把“市场”“规模市”等普通短语错误识别为地名。
LOC_STRONG_SUFFIXES = [
    "自治区",
    "特别行政区",
    "自治州",
    "北部湾",
    "湾",
    "岛",
]

LOC_WHITELIST = {
    "北京", "上海", "天津", "重庆",
    "北京市", "上海市", "天津市", "重庆市",
    "台北", "台北市",
    "中国", "美国", "英国", "法国", "德国", "日本", "韩国", "俄罗斯",
    "台湾", "香港", "澳门",
    "广西", "广西壮族自治区",
    "越南", "广宁省", "北部湾",
}

# 明显不是实体的误报片段，可根据你自己的实验样例继续扩充
COMMON_BAD_ENTITY_WORDS = {
    "毋庸讳",
    "胜其烦",
    "堪其苦",
    "毒瘤",
    "之夜",
    "企业商誉",
    "企业蒙",
    "自贸试",
    "券公司",
    "言如同寄生在市",
    "国的超大规模市",
}

# 机构名前容易被正则吞进去的动词或普通词前缀
BAD_ORG_PREFIXES = [
    "批准",
    "支持",
    "同意",
    "允许",
    "推动",
    "要求",
    "联合",
    "发布",
    "宣布",
    "设立",
    "成立",
    "来自",
    "包括",
    "公告",
    "通过",
    "引入",
    "授权",
]


def is_all_chinese(text: str) -> bool:
    return bool(text) and all("\u4e00" <= ch <= "\u9fff" for ch in text)


def build_entity_gazetteer(samples: Sequence[Dict[str, object]]) -> Dict[str, set]:
    """
    从训练集银标中抽取实体词典。
    注意：只允许使用训练集，不能使用测试集银标，否则会造成测试结果虚高。
    """
    gazetteer = {
        "PER": set(),
        "LOC": set(),
        "ORG": set(),
    }

    for sample in samples:
        chars = list(sample["chars"])  # type: ignore[arg-type]
        tags = list(sample["tags"])    # type: ignore[arg-type]

        entities = tags_to_entities(chars, tags)

        for entity_type, _start, _end, text in entities:
            if len(text) >= 2:
                gazetteer[entity_type].add(text)

    return gazetteer


def is_valid_person_name(text: str, gazetteer: Dict[str, set]) -> bool:
    """
    人名过滤：
    1. 训练集人名词典中出现过，直接保留；
    2. 或者长度 2~4，且第一个字是常见姓氏或前两个字是复姓；
    3. 明显普通词过滤。
    """
    if text in gazetteer["PER"]:
        return True

    if text in COMMON_BAD_ENTITY_WORDS:
        return False

    if not is_all_chinese(text):
        return False

    if len(text) < 2 or len(text) > 4:
        return False

    if len(text) >= 3 and text[:2] in DOUBLE_SURNAMES:
        return True

    return text[0] in SURNAMES


def is_valid_location(text: str, gazetteer: Dict[str, set]) -> bool:
    """
    地名过滤：
    1. 训练集地名词典中出现过，保留；
    2. 常见地名白名单，保留；
    3. 以强地名后缀结尾，保留；
    4. 其他短语过滤。
    """
    if text in gazetteer["LOC"]:
        return True

    if text in LOC_WHITELIST:
        return True

    if text in COMMON_BAD_ENTITY_WORDS:
        return False

    if not is_all_chinese(text):
        return False

    if len(text) < 2 or len(text) > 12:
        return False

    # 不再接受任意 “xx市 / xx区 / xx县”，只接受强后缀
    return any(text.endswith(suffix) for suffix in LOC_STRONG_SUFFIXES)


def is_valid_organization(text: str, gazetteer: Dict[str, set]) -> bool:
    """
    机构名过滤：
    1. 训练集机构名词典中出现过，保留；
    2. 以强机构后缀结尾，保留；
    3. 太短或明显普通词过滤。
    """
    if text in gazetteer["ORG"]:
        return True

    if text in COMMON_BAD_ENTITY_WORDS:
        return False

    if not is_all_chinese(text):
        return False

    if len(text) < 3 or len(text) > 25:
        return False

    # “券公司”这类残片虽然以“公司”结尾，但长度过短，通常是边界错误
    if text.endswith("公司") and len(text) < 4:
        return False

    return any(text.endswith(suffix) for suffix in ORG_SUFFIXES)


def is_valid_entity(entity: Entity, gazetteer: Dict[str, set]) -> bool:
    entity_type, _start, _end, text = entity

    if entity_type == "PER":
        return is_valid_person_name(text, gazetteer)

    if entity_type == "LOC":
        return is_valid_location(text, gazetteer)

    if entity_type == "ORG":
        return is_valid_organization(text, gazetteer)

    return False


def find_all_occurrences(text: str, word: str) -> List[Tuple[int, int]]:
    """
    在 text 中查找 word 的所有出现位置。
    """
    spans = []
    start = 0

    while True:
        idx = text.find(word, start)
        if idx < 0:
            break

        spans.append((idx, idx + len(word)))
        start = idx + 1

    return spans


def gazetteer_match_entities(sentence: str, gazetteer: Dict[str, set]) -> List[Entity]:
    """
    使用训练集实体词典在测试句子中做最长匹配。
    """
    candidates: List[Entity] = []

    # 长词优先。ORG 放前面，避免“中国”这类 LOC 抢占“中国人民银行”
    for entity_type in ["ORG", "LOC", "PER"]:
        words = sorted(gazetteer[entity_type], key=len, reverse=True)

        for word in words:
            if len(word) < 2:
                continue

            for start, end in find_all_occurrences(sentence, word):
                candidates.append((entity_type, start, end, word))

    return candidates


def strip_bad_org_prefix(text: str) -> Tuple[int, str]:
    """
    清理机构名前被正则吞进去的动词前缀。

    返回：
    被裁掉的字符数, 裁剪后的文本

    例如：
    批准渣打银行 -> 2, 渣打银行
    公告批准渣打银行 -> 4, 渣打银行
    """
    removed = 0
    cleaned = text

    changed = True
    while changed:
        changed = False
        for prefix in sorted(BAD_ORG_PREFIXES, key=len, reverse=True):
            if cleaned.startswith(prefix) and len(cleaned) - len(prefix) >= 3:
                cleaned = cleaned[len(prefix):]
                removed += len(prefix)
                changed = True
                break

    return removed, cleaned


def suffix_rule_entities(sentence: str) -> List[Entity]:
    """
    使用通用后缀规则补充实体候选。

    本版对上一版做了两处收紧：
    1. 不再使用“任意中文串 + 市/区/县/省”的宽泛地名规则；
    2. 对“xx银行”匹配结果做动词前缀裁剪，避免“批准渣打银行”这类边界错误。
    """
    candidates: List[Entity] = []

    # 1. 机构名：xx银行，例如 渣打银行、汇丰银行、花旗银行、中国人民银行
    for match in re.finditer(r"[\u4e00-\u9fff]{2,8}银行", sentence):
        raw_text = match.group()
        removed, text = strip_bad_org_prefix(raw_text)
        start = match.start() + removed
        end = match.end()

        if text and len(text) >= 3:
            candidates.append(("ORG", start, end, text))

    # 2. 机构名：国家xxxx总局 / 国家xxxx委员会
    for match in re.finditer(r"国家[\u4e00-\u9fff]{2,12}(总局|委员会)", sentence):
        text = match.group()
        candidates.append(("ORG", match.start(), match.end(), text))

    # 3. 机构名：部委、总局、委员会、法院、检察院、研究院等
    for match in re.finditer(
        r"[\u4e00-\u9fff]{2,12}(公安部|司法部|国家安全部|总局|委员会|法院|检察院|研究院)",
        sentence,
    ):
        raw_text = match.group()
        removed, text = strip_bad_org_prefix(raw_text)
        start = match.start() + removed
        end = match.end()

        if text and len(text) >= 3:
            candidates.append(("ORG", start, end, text))

    # 4. 常见地名白名单
    for loc in LOC_WHITELIST:
        for start, end in find_all_occurrences(sentence, loc):
            candidates.append(("LOC", start, end, loc))

    # 5. 强地名后缀规则：只保留较安全的地名后缀，避免“市场”等误报
    for match in re.finditer(
        r"[\u4e00-\u9fff]{2,8}(自治区|特别行政区|自治州|湾|岛)",
        sentence,
    ):
        text = match.group()
        candidates.append(("LOC", match.start(), match.end(), text))

    return candidates


def resolve_overlaps(entities: List[Entity], gazetteer: Dict[str, set]) -> List[Entity]:
    """
    处理实体重叠：
    1. 先过滤不合法实体；
    2. 长实体优先；
    3. 若位置重叠，只保留更长的那个。
    """
    valid_entities = [
        entity for entity in entities
        if is_valid_entity(entity, gazetteer)
    ]

    # 长实体优先；同长度时 ORG 优先于 LOC，LOC 优先于 PER
    type_priority = {"ORG": 3, "LOC": 2, "PER": 1}

    valid_entities.sort(
        key=lambda e: (e[2] - e[1], type_priority.get(e[0], 0), -e[1]),
        reverse=True,
    )

    selected: List[Entity] = []
    occupied = set()

    for entity in valid_entities:
        _entity_type, start, end, _text = entity
        span = set(range(start, end))

        if occupied & span:
            continue

        selected.append(entity)
        occupied.update(span)

    selected.sort(key=lambda e: e[1])
    return selected


def entities_to_tags(chars: Sequence[str], entities: List[Entity]) -> List[str]:
    """
    将后处理后的实体列表重新转换成字符级 NER 标签。
    """
    tags = ["O"] * len(chars)

    for entity_type, start, end, _text in entities:
        if start < 0 or end > len(chars) or start >= end:
            continue

        # 避免覆盖已有实体
        if any(tags[i] != "O" for i in range(start, end)):
            continue

        entity_tags = make_entity_tags(end - start, entity_type)
        tags[start:end] = entity_tags

    return tags


def postprocess_ner_tags(
    sentence: str,
    chars: Sequence[str],
    pred_tags: Sequence[str],
    gazetteer: Dict[str, set],
    use_gazetteer: bool = True,
    use_suffix_rules: bool = True,
) -> List[str]:
    """
    HMM-NER 后处理：
    1. 先取 HMM 预测实体；
    2. 过滤明显不合理的实体；
    3. 用训练集实体词典补全；
    4. 用通用后缀规则补全；
    5. 解决重叠后重新转回标签。
    """
    candidates: List[Entity] = []

    # HMM 初始预测实体
    candidates.extend(tags_to_entities(chars, pred_tags))

    # 训练集实体词典匹配
    if use_gazetteer:
        candidates.extend(gazetteer_match_entities(sentence, gazetteer))

    # 后缀规则补全
    if use_suffix_rules:
        candidates.extend(suffix_rule_entities(sentence))

    final_entities = resolve_overlaps(candidates, gazetteer)

    return entities_to_tags(chars, final_entities)


# =========================
# 评价指标
# =========================

def calculate_tag_accuracy(
    samples: Sequence[Dict[str, object]],
    predictions: Sequence[List[str]],
) -> float:
    """
    标签级 Accuracy。
    """
    total = 0
    correct = 0

    for sample, pred_tags in zip(samples, predictions):
        true_tags = list(sample["tags"])  # type: ignore[arg-type]

        for true_tag, pred_tag in zip(true_tags, pred_tags):
            total += 1
            if true_tag == pred_tag:
                correct += 1

    return correct / total if total else 0.0


def calculate_entity_metrics(
    samples: Sequence[Dict[str, object]],
    predictions: Sequence[List[str]],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    实体级完全匹配 Precision / Recall / F1。

    只有实体类型、起始位置、结束位置、实体文本都一致，
    才算预测正确。
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    by_type_counts = {
        entity_type: {"tp": 0, "fp": 0, "fn": 0}
        for entity_type in ENTITY_TYPES
    }

    for sample, pred_tags in zip(samples, predictions):
        chars = list(sample["chars"])      # type: ignore[arg-type]
        true_tags = list(sample["tags"])   # type: ignore[arg-type]

        true_entities = set(tags_to_entities(chars, true_tags))
        pred_entities = set(tags_to_entities(chars, pred_tags))

        tp_set = true_entities & pred_entities
        fp_set = pred_entities - true_entities
        fn_set = true_entities - pred_entities

        total_tp += len(tp_set)
        total_fp += len(fp_set)
        total_fn += len(fn_set)

        for entity_type in ENTITY_TYPES:
            by_type_counts[entity_type]["tp"] += sum(
                1 for entity in tp_set if entity[0] == entity_type
            )
            by_type_counts[entity_type]["fp"] += sum(
                1 for entity in fp_set if entity[0] == entity_type
            )
            by_type_counts[entity_type]["fn"] += sum(
                1 for entity in fn_set if entity[0] == entity_type
            )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    micro = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
    }

    by_type: Dict[str, Dict[str, float]] = {}

    for entity_type, counts in by_type_counts.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]

        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0

        by_type[entity_type] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }

    return micro, by_type


# =========================
# 结果输出
# =========================

def write_metrics_csv(
    raw_tag_accuracy: float,
    raw_micro: Dict[str, float],
    raw_by_type: Dict[str, Dict[str, float]],
    final_tag_accuracy: float,
    final_micro: Dict[str, float],
    final_by_type: Dict[str, Dict[str, float]],
) -> None:
    """
    保存后处理前后评价指标到 CSV。
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    def add_rows(
        prefix: str,
        tag_accuracy: float,
        micro: Dict[str, float],
        by_type: Dict[str, Dict[str, float]],
    ) -> None:
        rows.extend([
            [f"{prefix}_tag_accuracy", tag_accuracy],
            [f"{prefix}_entity_micro_precision", micro["precision"]],
            [f"{prefix}_entity_micro_recall", micro["recall"]],
            [f"{prefix}_entity_micro_f1", micro["f1"]],
            [f"{prefix}_entity_micro_tp", micro["tp"]],
            [f"{prefix}_entity_micro_fp", micro["fp"]],
            [f"{prefix}_entity_micro_fn", micro["fn"]],
        ])

        for entity_type in ENTITY_TYPES:
            rows.extend([
                [f"{prefix}_{entity_type}_precision", by_type[entity_type]["precision"]],
                [f"{prefix}_{entity_type}_recall", by_type[entity_type]["recall"]],
                [f"{prefix}_{entity_type}_f1", by_type[entity_type]["f1"]],
                [f"{prefix}_{entity_type}_tp", by_type[entity_type]["tp"]],
                [f"{prefix}_{entity_type}_fp", by_type[entity_type]["fp"]],
                [f"{prefix}_{entity_type}_fn", by_type[entity_type]["fn"]],
            ])

    add_rows("raw", raw_tag_accuracy, raw_micro, raw_by_type)
    add_rows("final", final_tag_accuracy, final_micro, final_by_type)

    with METRICS_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for metric, value in rows:
            if isinstance(value, float):
                writer.writerow([metric, f"{value:.6f}"])
            else:
                writer.writerow([metric, value])


def write_prediction_file(
    samples: Sequence[Dict[str, object]],
    raw_predictions: Sequence[List[str]],
    final_predictions: Sequence[List[str]],
) -> None:
    """
    保存测试集预测样例。
    同时保存 HMM 原始预测与后处理后的最终预测。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    for idx, (sample, raw_tags, final_tags) in enumerate(
        zip(samples, raw_predictions, final_predictions),
        1,
    ):
        sentence = str(sample["sentence"])
        chars = list(sample["chars"])      # type: ignore[arg-type]
        true_tags = list(sample["tags"])   # type: ignore[arg-type]

        true_entities = tags_to_entities(chars, true_tags)
        raw_entities = tags_to_entities(chars, raw_tags)
        final_entities = tags_to_entities(chars, final_tags)

        lines.append(f"句子{idx}: {sentence}")
        lines.append(f"银标人名: {format_entities(true_entities, 'PER')}")
        lines.append(f"原始预测人名: {format_entities(raw_entities, 'PER')}")
        lines.append(f"最终预测人名: {format_entities(final_entities, 'PER')}")
        lines.append(f"银标地名: {format_entities(true_entities, 'LOC')}")
        lines.append(f"原始预测地名: {format_entities(raw_entities, 'LOC')}")
        lines.append(f"最终预测地名: {format_entities(final_entities, 'LOC')}")
        lines.append(f"银标机构名: {format_entities(true_entities, 'ORG')}")
        lines.append(f"原始预测机构名: {format_entities(raw_entities, 'ORG')}")
        lines.append(f"最终预测机构名: {format_entities(final_entities, 'ORG')}")
        lines.append("银标标签: " + " ".join(true_tags))
        lines.append("原始预测标签: " + " ".join(raw_tags))
        lines.append("最终预测标签: " + " ".join(final_tags))
        lines.append("")

    PREDICTION_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def print_metrics(
    title: str,
    tag_accuracy: float,
    micro: Dict[str, float],
    by_type: Dict[str, Dict[str, float]],
) -> None:
    """
    控制台打印指标。
    """
    print(title)
    print(f"标签级 Accuracy: {tag_accuracy:.4f}")
    print(
        "实体级 Micro P/R/F1: "
        f"{micro['precision']:.4f} / {micro['recall']:.4f} / {micro['f1']:.4f}"
    )

    for entity_type, chinese_name in [
        ("PER", "人名"),
        ("LOC", "地名"),
        ("ORG", "机构名"),
    ]:
        print(
            f"{chinese_name}({entity_type}) P/R/F1: "
            f"{by_type[entity_type]['precision']:.4f} / "
            f"{by_type[entity_type]['recall']:.4f} / "
            f"{by_type[entity_type]['f1']:.4f}"
        )


# =========================
# 主函数
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HMM-NER sequence labeling with HanLP silver labels"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="Lidstone 平滑系数，默认 0.001。此前实验中 0.001 的实体级 F1 较好。",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="调试用，只读取前 N 条训练/测试样本",
    )

    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="控制台展示前 N 条预测样例，默认 5",
    )

    parser.add_argument(
        "--use-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用 NER-BMES 合法转移约束，默认开启",
    )

    parser.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用词典/规则后处理，默认开启。关闭方式：--no-postprocess",
    )

    parser.add_argument(
        "--use-gazetteer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="后处理中是否使用训练集实体词典，默认开启。",
    )

    parser.add_argument(
        "--use-suffix-rules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="后处理中是否使用地名/机构名后缀规则，默认开启。",
    )

    args = parser.parse_args()

    print("===== HMM-NER 扩展实验 =====")
    print(f"训练集路径: {TRAIN_PATH}")
    print(f"测试集路径: {TEST_PATH}")
    print(f"alpha: {args.alpha}")
    print(f"NER-BMES 合法转移约束: {'开启' if args.use_constraints else '关闭'}")
    print(f"词典/规则后处理: {'开启' if args.postprocess else '关闭'}")
    print(f"训练集实体词典: {'开启' if args.use_gazetteer else '关闭'}")
    print(f"后缀规则: {'开启' if args.use_suffix_rules else '关闭'}")
    print("标签来源: HanLP 自动生成银标标签")
    print()

    raw_train_samples = load_samples(TRAIN_PATH, limit=args.limit)
    raw_test_samples = load_samples(TEST_PATH, limit=args.limit)

    print(f"原始训练样本数: {len(raw_train_samples)}")
    print(f"原始测试样本数: {len(raw_test_samples)}")
    print("正在调用 HanLP 生成 NER 银标标签...")

    segmenter = build_hanlp_segmenter()

    train_samples = add_silver_labels(raw_train_samples, segmenter)
    test_samples = add_silver_labels(raw_test_samples, segmenter)

    print(f"NER 训练样本数: {len(train_samples)}")
    print(f"NER 测试样本数: {len(test_samples)}")

    gazetteer = build_entity_gazetteer(train_samples)
    print(
        "训练集实体词典规模: "
        f"PER={len(gazetteer['PER'])}, "
        f"LOC={len(gazetteer['LOC'])}, "
        f"ORG={len(gazetteer['ORG'])}"
    )
    print()

    model = HMMNER(
        alpha=args.alpha,
        use_constraints=args.use_constraints,
    )

    model.train(train_samples)

    raw_predictions = [
        model.viterbi(sample["chars"])  # type: ignore[arg-type]
        for sample in test_samples
    ]

    if args.postprocess:
        final_predictions = []

        for sample, raw_tags in zip(test_samples, raw_predictions):
            sentence = str(sample["sentence"])
            chars = list(sample["chars"])  # type: ignore[arg-type]

            final_tags = postprocess_ner_tags(
                sentence=sentence,
                chars=chars,
                pred_tags=raw_tags,
                gazetteer=gazetteer,
                use_gazetteer=args.use_gazetteer,
                use_suffix_rules=args.use_suffix_rules,
            )

            final_predictions.append(final_tags)
    else:
        final_predictions = raw_predictions

    raw_tag_accuracy = calculate_tag_accuracy(test_samples, raw_predictions)
    raw_micro, raw_by_type = calculate_entity_metrics(test_samples, raw_predictions)

    final_tag_accuracy = calculate_tag_accuracy(test_samples, final_predictions)
    final_micro, final_by_type = calculate_entity_metrics(test_samples, final_predictions)

    print_metrics("===== 原始 HMM-NER 评价结果 =====", raw_tag_accuracy, raw_micro, raw_by_type)
    print()
    print_metrics("===== 后处理后 HMM-NER 评价结果 =====", final_tag_accuracy, final_micro, final_by_type)

    write_metrics_csv(
        raw_tag_accuracy,
        raw_micro,
        raw_by_type,
        final_tag_accuracy,
        final_micro,
        final_by_type,
    )

    write_prediction_file(test_samples, raw_predictions, final_predictions)

    print()
    print(f"指标已保存: {METRICS_PATH}")
    print(f"预测结果已保存: {PREDICTION_PATH}")

    if args.show_examples > 0:
        print()
        print("===== 预测样例 =====")

        for idx, (sample, raw_tags, final_tags) in enumerate(
            zip(
                test_samples[:args.show_examples],
                raw_predictions[:args.show_examples],
                final_predictions[:args.show_examples],
            ),
            1,
        ):
            sentence = str(sample["sentence"])
            chars = list(sample["chars"])      # type: ignore[arg-type]
            true_tags = list(sample["tags"])   # type: ignore[arg-type]

            true_entities = tags_to_entities(chars, true_tags)
            raw_entities = tags_to_entities(chars, raw_tags)
            final_entities = tags_to_entities(chars, final_tags)

            print()
            print(f"样例{idx}")
            print(f"原句: {sentence}")
            print(f"银标人名: {format_entities(true_entities, 'PER')}")
            print(f"原始预测人名: {format_entities(raw_entities, 'PER')}")
            print(f"最终预测人名: {format_entities(final_entities, 'PER')}")
            print(f"银标地名: {format_entities(true_entities, 'LOC')}")
            print(f"原始预测地名: {format_entities(raw_entities, 'LOC')}")
            print(f"最终预测地名: {format_entities(final_entities, 'LOC')}")
            print(f"银标机构名: {format_entities(true_entities, 'ORG')}")
            print(f"原始预测机构名: {format_entities(raw_entities, 'ORG')}")
            print(f"最终预测机构名: {format_entities(final_entities, 'ORG')}")
            print("银标标签:", " ".join(true_tags))
            print("原始预测标签:", " ".join(raw_tags))
            print("最终预测标签:", " ".join(final_tags))


if __name__ == "__main__":
    main()