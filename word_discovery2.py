from stop_words import stop_words
from collections import Counter
import math
from functools import reduce
from operator import mul
import re
import os
from tqdm import tqdm


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


def get_candidate_wordsinfo(texts, max_word_len=5):
    '''
    texts：表示输入的所有文本
    max_word_len：表示最长的词长
    '''
    # 四个词典均以单词为 key，分别以词频、候选新词词频、左字集合、右字集合为 value
    words_freq, candidate_words_freq, candidate_words_left_characters, candidate_words_right_characters = {}, {}, {}, {}
    word_num = 0  # 统计所有可能的字符串频次
    with tqdm(texts, total=len(texts)) as pbar:
        for text in pbar:  # 遍历每个文本
            # word_indexes 中存储了所有可能的词汇的切分下标 (i,j) ，i 表示词汇的起始下标，j 表示结束下标，注意这里有包括了所有的字
            # word_indexes 的生成需要两层循环，第一层循环，遍历所有可能的起始下标 i；第二层循环，在给定 i 的情况下，遍历所有可能的结束下标 j
            # 例：自、自然、自然语、自然语言、自然语言处、自然语言处理、然、然语、然语言、然语言处、然语言处理、然语言处理是
            word_indexes = [(i, j) for i in range(len(text)) for j in range(i + 1, i + 1 + max_word_len)]
            word_num += len(word_indexes)
            for index in word_indexes:  # 遍历所有词汇的下标
                word = text[index[0]:index[1]]  # 获取单词
                # 更新所有切分出的字符串的频次信息
                if word in words_freq:
                    words_freq[word] += 1
                else:
                    words_freq[word] = 1

                if len(word) >= 2:  # 长度大于等于 2 的词以及不是词典中的词作为候选新词
                    # 更新候选新词词频
                    if word in candidate_words_freq:
                        candidate_words_freq[word] += 1
                    else:
                        candidate_words_freq[word] = 1
                    # 更新候选新词左字集合
                    if index[0] != 0:  # 当为文本中首个单词时无左字
                        if word in candidate_words_left_characters:
                            candidate_words_left_characters[word].append(text[index[0] - 1])
                        else:
                            candidate_words_left_characters[word] = [text[index[0] - 1]]
                    else:
                        if word in candidate_words_left_characters:
                            candidate_words_left_characters[word].append(len(candidate_words_left_characters[word]))
                        else:
                            candidate_words_left_characters[word] = [0]
                    # 更新候选新词右字集合
                    if index[1] < len(text) - 1:  # 当为文本中末个单词时无右字
                        if word in candidate_words_right_characters:
                            candidate_words_right_characters[word].append(text[index[1]])  #
                        else:
                            candidate_words_right_characters[word] = [text[index[1]]]
                    else:
                        if word in candidate_words_right_characters:
                            candidate_words_right_characters[word].append(len(candidate_words_right_characters[word]))
                        else:
                            candidate_words_right_characters[word] = [0]

    return word_num, words_freq, candidate_words_freq, candidate_words_left_characters, candidate_words_right_characters


# 计算候选单词的 pmi 值
def compute_pmi(words_freq, candidate_words_freq, word_num):
    words_pmi = {}
    with tqdm(candidate_words_freq, total=len(candidate_words_freq), desc="Counting pmi") as pbar:
        for word in pbar:
            # 首先，将某个候选单词按照不同的切分位置切分成两项，比如“电影院”可切分为“电”和“影院”以及“电影”和“院”
            bi_grams = [(word[0:i], word[i:]) for i in range(1, len(word))]
            # 对所有切分情况计算 pmi 值，取最大值作为当前候选词的最终 pmi 值
            # words_freq[bi_gram[0]]，words_freq[bi_gram[1]] 分别表示一个候选新词的前后两部分的出现频次
            words_pmi[word] = max(map(lambda bi_gram: math.log(
                words_freq[word] / (words_freq[bi_gram[0]] * words_freq[bi_gram[1]] / word_num)), bi_grams))
    return words_pmi

# def compute_pmi(words_freq, candidate_words_freq):
#     words_pmi = {}
#     empty_words = [sw for sw in stop_words.values() if len(sw) == 1]  # 虚词
#     with tqdm(candidate_words_freq, total=len(candidate_words_freq), desc="Counting pmi") as pbar:
#         for word in pbar:
#             word_prob = words_freq[word] / len(words_freq)
#             char_freq = [words_freq.get(wd, 1) for wd in word]
#             char_prob = reduce(mul, ([wf for wf in char_freq])) / (len(words_freq) ** (len(word)))
#             pmi = math.log(word_prob / char_prob, 2)
#             # AMI=PMI/length_word. 惩罚虚词(避免"的", "得", "了"开头结尾的情况)
#             if word[0] in empty_words or word[-1] in empty_words:
#                 word_aggregation = pmi / (len(word) ** len(word))
#             else:
#                 word_aggregation = pmi / len(word)
#             words_pmi[word] = word_aggregation
#     return words_pmi


# 计算候选单词的邻字熵
def compute_entropy(candidate_words_characters):
    words_entropy = {}
    with tqdm(candidate_words_characters.items(), total=len(candidate_words_characters),
              desc="Counting entropy") as pbar:
        for word, characters in pbar:
            character_freq = Counter(characters)  # 统计邻字的出现分布
            # 根据出现分布计算邻字熵
            words_entropy[word] = sum(
                map(lambda x: - x / len(characters) * math.log(x / len(characters)), character_freq.values()))
    return words_entropy


# C-value
def c_value(cand_words, words_freq):

    def count_nested(w):
        c, t = 0, 0
        for x in cand_words:
            if w != x and w in x:
                c += 1
                t += words_freq[x]
        return c, t

    cv_dict = {}
    for w in cand_words:
        c, t = count_nested(w)
        f = words_freq[w]
        if c == 0:
            cv = math.log2(len(w)) * f
        else:
            cv = math.log2(len(w)) * (f - t / c)
        cv_dict[w] = cv
    return cv_dict


# 根据各指标阈值获取最终的新词结果
def get_newwords(candidate_words_freq,
                 words_pmi,
                 words_left_entropy,
                 words_right_entropy,
                 words_freq_limit=4,
                 pmi_limit=5.0,
                 entropy_limit=1.0):
    # 在每一项指标中根据阈值进行筛选
    candidate_words = [k for k, v in candidate_words_freq.items() if v >= words_freq_limit and k not in stop_words]
    # candidate_words_pmi = [k for k, v in words_pmi.items() if v >= pmi_limit]
    # candidate_words_left_entropy = [k for k, v in words_left_entropy.items() if v >= entropy_limit]
    # candidate_words_right_entropy = [k for k, v in words_right_entropy.items() if v >= entropy_limit]
    # 对筛选结果进行合并
    # return list(set(candidate_words).intersection(candidate_words_pmi, candidate_words_left_entropy, candidate_words_right_entropy))

    new_word_score = {}
    empty_words = [sw for sw in stop_words.values() if len(sw) == 1]  # 虚词
    word_liberalization = lambda le, re: math.log((le * 2 ** re + re * 2 ** le + 1e-5) / (abs(le - re) + 1), 1.5)
    for cw in candidate_words:
        le, re = words_left_entropy[cw], words_right_entropy[cw]
        if cw[0] in empty_words or cw[-1] in empty_words:  # 惩罚虚词开头或者结尾
            le, re = le / len(cw), re / len(cw)

        lre = word_liberalization(le, re)
        # if words_pmi[cw] >= pmi_limit and min(re, le) >= entropy_limit:
        if words_pmi[cw] >= pmi_limit and lre >= entropy_limit:
            # new_word_score[cw] = words_pmi[cw] + min(re, le)
            new_word_score[cw] = words_pmi[cw] + lre

    new_words = [item[0] for item in sorted(new_word_score.items(), key=lambda x: x[1], reverse=True)]
    return new_words


def find_new_words(text, use_type='file', freq_min=2, len_max=5, entropy_min=2.0, aggregation_min=3.2):
    if use_type == "text":  # 输入为文本形式
        texts = cut_sentence(text=text)  # 切句子, 如中英文的逗号/句号/感叹号
    elif use_type == "file":  # 输入为文件形式
        if not os.path.exists(text):
            raise RuntimeError("path of text must exist!")

        texts = []
        with open(text, "r", encoding="utf-8") as fr8:
            for txt in fr8:
                if txt.strip():
                    txts = cut_sentence(text=txt)  # 切句子, 如中英文的逗号/句号/感叹号
                    texts.extend(txts)
    else:
        raise RuntimeError("use_type must be 'text' or 'file'")

    word_num, words_freq, candidate_words_freq, candidate_words_left_characters, candidate_words_right_characters = \
        get_candidate_wordsinfo(texts=texts, max_word_len=len_max)

    words_pmi = compute_pmi(words_freq, candidate_words_freq, word_num)
    words_left_entropy = compute_entropy(candidate_words_left_characters)
    words_right_entropy = compute_entropy(candidate_words_right_characters)
    new_words = get_newwords(candidate_words_freq,
                             words_pmi,
                             words_left_entropy,
                             words_right_entropy,
                             words_freq_limit=freq_min,
                             pmi_limit=aggregation_min,
                             entropy_limit=entropy_min)

    new_words = list(filter(lambda x: not re.search("[^\u4e00-\u9fa5]", x), new_words))  # 是否为汉字
    new_words = list(filter(lambda x: not re.search("[了但里的和为是]", x), new_words))
    return new_words


if __name__ == '__main__':
    res = find_new_words(text='ZX.txt', use_type="file", freq_min=2, len_max=5, entropy_min=2.0, aggregation_min=3.2)
    with open("new_words.txt", "w", encoding="utf-8") as f:
        for new_word in res:
            f.write(new_word + "\n")

    print('Done')
