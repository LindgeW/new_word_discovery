from stop_words import stop_words
from collections import Counter
from functools import reduce
from operator import mul
import math
import re
import os


def cut_sentence(text):
    """
        分句
    :param sentence:str, like "大漠帝国"
    :return:list
    """
    re_sen = re.compile('[,，"“”<>《》{}【】:;!?。：；？！\n\r]')  # .不加是因为不确定.是小数还是英文句号(中文省略号......)
    sentences = re_sen.split(text)
    sen_cuts = []
    for sen in sentences:
        if sen and str(sen).strip():
            sen_cuts.append(sen)
    return sen_cuts


def get_ngrams(text, ns=None, len_max=7):
    """
        获取文本的ngram等特征
    :param text: str, like "大漠帝国"
    :param ns: list, like [1, 2, 3]
    :param len_max: int, like 6, 7
    :return: list<list> or list
    """
    if ns is None:
        ns = [1]
    if type(ns) != list:
        raise RuntimeError("ns of function get_ngram() must be list!")
    for n in ns:
        if n < 1:
            raise RuntimeError("enum of ns must '>1'!")
    len_text = len(text)
    ngrams = []
    for i in range(len_text):
        ngrams += [text[i: j + i] for j in range(1, min(len_max + 1, len_text - i + 1))]
    return ngrams


class WordDiscovery:
    def __init__(self):
        self.stop_words = stop_words
        self.total_words_len = {}
        self.total_words = 0
        self.freq_min = 3
        self.len_max = 7
        self.round = 6
        self.eps = 1e-9
        self.empty_words = [sw for sw in stop_words.values() if len(sw) == 1]  # 虚词

    def count_word(self, text, use_type="text"):
        """
            词频统计(句子/段落/文章)
        :param text: str, path or doc, like "大漠帝国。" or "/home/data/doc.txt"
        :param use_type: str,  "text" or "file", file of "utf-8" of "txt"
        :return: class<Counter>, word-freq
        """
        self.words_count = Counter()
        if use_type == "text":  # 输入为文本形式
            texts = cut_sentence(text=text)  # 切句子, 如中英文的逗号/句号/感叹号
            for text in texts:
                n_grams = get_ngrams(len_max=self.len_max, text=text)  # 获取一个句子的所有n-gram
                self.words_count.update(n_grams)
        elif use_type == "file":  # 输入为文件形式
            if not os.path.exists(text):
                raise RuntimeError("path of text must exist!")
            with open(text, "r", encoding="utf-8") as fr8:
                for text in fr8:
                    if text.strip():
                        texts = cut_sentence(text=text)  # 切句子, 如中英文的逗号/句号/感叹号
                        for text in texts:
                            n_grams = get_ngrams(len_max=self.len_max, text=text)  # 获取一个句子的所有n-gram
                            self.words_count.update(n_grams)
        else:
            raise RuntimeError("use_type must be 'text' or 'file'")

        self.total_words = sum(self.words_count.values())

    def calculate_entropy(self, boundary_type="left"):
        """
            计算左熵和右熵
        :param boundary_type: str, like "left" or "right"
        :return: None
        """
        # 获取成词的最左边和最右边的一个字
        one_collect = {}
        self.total_words_len = {}
        for k, v in self.words_count.items():
            len_k = len(k)
            if len_k >= 2:  # 词长度大于3
                if boundary_type == "right":
                    k_boundary = k[:-1]
                else:
                    k_boundary = k[1:]
                # 左右边, 保存为dict, 左右丰度
                if k_boundary in self.words_count:
                    if k_boundary not in one_collect:
                        one_collect[k_boundary] = [v]
                    else:
                        one_collect[k_boundary] = one_collect[k_boundary] + [v]
            # 计算n-gram的长度
            if len_k not in self.total_words_len:
                self.total_words_len[len_k] = [v]
            else:
                self.total_words_len[len_k] += [v]
        self.total_words_len = dict([(k, sum(v)) for k, v in self.total_words_len.items()])

        # 计算左右熵
        for k, v in self.words_select.items():
            # 从字典获取
            boundary_v = one_collect.get(k, None)
            # 计算候选词的左右凝固度, 取最小的那个
            if boundary_v:
                # 求和
                sum_boundary = sum(boundary_v)
                # 计算信息熵
                entropy_boundary = sum([-(enum_bo / sum_boundary) * math.log(enum_bo / sum_boundary, 2)
                                       for enum_bo in boundary_v])
            else:
                entropy_boundary = 0.0
            # 惩罚虚词开头或者结尾
            if k[0] in self.empty_words or k[-1] in self.empty_words:
                entropy_boundary = entropy_boundary / len(k)
            if boundary_type == "right":
                self.right_entropy[k] = round(entropy_boundary, self.round)
            else:
                self.left_entropy[k] = round(entropy_boundary, self.round)

    def compute_entropys(self):
        """
            计算左右熵
        :return: dict
        """
        # 提取大于最大频率的词语, 以及长度在3-len_max的词语
        self.words_select = {word: count for word, count in self.words_count.items()
                             if count >= self.freq_min and " " not in word
                             and 1 < len(word) <= self.len_max}
        # 计算凝固度, 左右两边
        self.right_entropy = {}
        self.left_entropy = {}
        self.calculate_entropy(boundary_type="left")
        self.calculate_entropy(boundary_type="right")

    def compute_aggregation(self):
        """
            计算凝固度PMI
        :return: None
        """
        twl_1 = self.total_words_len[1]  # ngram=1 的所有词频
        self.aggregation = {}
        for word, value in self.words_select.items():
            len_word = len(word)
            twl_n = self.total_words_len[len_word]  # ngram=n 的所有词频
            words_freq = [self.words_count.get(wd, 1) for wd in word]
            probability_word = value / twl_n
            probability_chars = reduce(mul, ([wf for wf in words_freq])) / (twl_1 ** (len(word)))
            pmi = math.log(probability_word / probability_chars, 2)
            # AMI=PMI/length_word. 惩罚虚词(避免"的", "得", "了"开头结尾的情况)
            if word[0] in self.empty_words or word[-1] in self.empty_words:
                word_aggregation = pmi / (len_word ** len_word)
            else:
                word_aggregation = pmi / len_word
            self.aggregation[word] = round(word_aggregation, self.round)

    def compute_score(self, word, value, a, r, l, rl, lambda_0, lambda_3):
        """
            计算最终得分
        :param word: str, word with prepare
        :param value: float, word freq
        :param a: float, aggregation of word
        :param r: float, right entropy of word
        :param l: float, left entropy of word
        :param rl: float, right_entropy * left_entropy
        :param lambda_0: lambda 0
        :param lambda_3: lambda 3
        :return:
        """
        self.new_words[word] = {}
        self.new_words[word]["a"] = a
        self.new_words[word]["r"] = r
        self.new_words[word]["l"] = l
        self.new_words[word]["f"] = value
        # word-liberalization
        m1 = lambda_0(r)
        m2 = lambda_0(l)
        m3 = lambda_0(a)
        score_ns = lambda_0((lambda_3(m1, m2) + lambda_3(m1, m3) + lambda_3(m2, m3)) / 3)
        self.new_words[word]["ns"] = round(score_ns, self.round)
        # 乘以词频word-freq, 连乘是为了防止出现较小项
        score_s = value * a * rl * score_ns
        self.new_words[word]["s"] = round(score_s, self.round)

    def find_word(self, text, use_type="text", freq_min=2, len_max=5, entropy_min=2.0, aggregation_min=3.2,
                  use_output=True, use_avg=False, use_filter=False):
        """
            新词发现与策略
        :param text: str, path or doc, like "大漠帝国。" or "/home/data/doc.txt"
        :param use_type: str,  输入格式, 即文件输入还是文本输入, "text" or "file", file of "utf-8" of "txt"
        :param use_output: bool,  输出模式, 即最后结果是否全部输出
        :param use_filter: bool,  新词过滤, 即是否过滤通用词和停用词
        :param freq_min: int, 最小词频, 大于1
        :param len_max: int, 最大成词长度, 一般为5, 6, 7
        :param entropy_min: int, 左右熵阈值, 低于则过滤
        :param aggregation_min: int, PMI(凝固度)-阈值, 低于则过滤
        :return:
        """
        self.aggregation_min = aggregation_min
        self.entropy_min = entropy_min
        self.freq_min = freq_min
        self.len_max = len_max
        self.count_word(text=text, use_type=use_type)
        self.compute_entropys()
        self.compute_aggregation()

        self.new_words = {}
        # 左右熵和凝固度的综合
        lambda_3 = lambda m1, m2: math.log((m1 * math.e ** m2 + m2 * math.e ** m1 + self.eps) / (abs(m1 - m2) + 1), 10)
        lambda_0 = lambda x: -self.eps * x + self.eps if x <= 0 else x
        for word, value in self.words_select.items():
            # # 过滤通用词
            # if use_filter and word in 通用词典:
            #     continue
            # 过滤停用词
            if word in self.stop_words:
                continue
            # {"aggregation":"a",
            # "right_entropy":"r",
            # "left_entropy":"l",
            # "frequency":"f",
            #  "word-liberalization":"ns",
            #  "score":"s"}
            a = self.aggregation[word]
            r = self.right_entropy[word]
            l = self.left_entropy[word]
            rl = (r + l) / 2 if use_avg else r * l
            if use_output or (use_avg and a > self.aggregation_min and rl > self.entropy_min) \
                    or (not use_avg and a > self.aggregation_min and r > self.entropy_min and l > self.entropy_min):
                self.compute_score(word, value, a, r, l, rl, lambda_0, lambda_3)

        self.new_words = [item[0] for item in sorted(self.new_words.items(), key=lambda x: x[1]['s'], reverse=True)]
        return self.new_words


if __name__ == '__main__':
    wd = WordDiscovery()
    res = wd.find_word(text='ZX.txt', use_type="file", use_avg=False, use_filter=True, use_output=True,
                       freq_min=2, len_max=5, entropy_min=2.0, aggregation_min=3.2)

    with open("new_words.txt", "w", encoding="utf-8") as f:
        for new_word in res:
            f.write(new_word + "\n")

    print('Done')
