import math


def load_lexicon(dict_path):
    lexicon = dict()
    if dict_path is None:
        return lexicon

    with open(dict_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line != '':
                token = line.split()
                if len(token) == 1:
                    lexicon[token[0]] = lexicon.get(token[0], 0) + 1
                else:
                    lexicon[token[0]] = float(token[1])
    return lexicon


class Chunk:
    def __init__(self, words, chrs):
        self.words = words
        self.lens_list = map(lambda x: len(x), words)
        # 优先级从高到低
        # 规则一：Maximum matching
        self.length = sum(self.lens_list)
        # 规则二：Largest average word length
        self.mean = float(self.length) / len(words)
        # 规则三：Smallest variance of word lengths
        self.var = sum(map(lambda x: (x - self.mean) ** 2, self.lens_list)) / len(self.words)
        # 规则四：Largest sum of degree of morphemic freedom of one-character words
        self.entropy = sum([math.log(float(chrs[x])) for x in words if len(x) == 1 and x in chrs])

    def __lt__(self, other):
        return (self.length, self.mean, -self.var, self.entropy) < \
           (other.length, other.mean, -other.var, other.entropy)

    def __repr__(self):
        return str(self.__dict__)


class MMSeg:
    def __init__(self, char_path=None, wd_path=None):
        self.chrs_dic = load_lexicon(char_path)
        self.wds_dic = load_lexicon(wd_path)

    def get_start_words(self, sentence):
        match_wds = []
        n = len(sentence)
        for i in range(1, n+1):
            seg = sentence[:i]
            if seg in self.wds_dic:
                match_wds.append(seg)
        return match_wds

    def get_chunks(self, sentence):
        # 获取chunk，每个chunk中最多三个词
        ret = []

        def iter_chunk(sentence, num, tmp_seg_words):
            # 获取当前句子中最开头的那个字，以及由该字所组成的词，但是该词需要同时满足俩个条件：
            # ①出现在预加载的词典中，②出现在当前句子中
            match_words = self.get_start_words(sentence)
            # 因为每个chunk中最多只有三个词，所以当num由最先初始化的3降为0时，进入if中，然后运用mmseg的4条消岐规则进行评分，
            # 最后将每种的分词的评分加入到一个list中去，以便在最后的时候进行比较，从而选取最优分词结果。
            if (not match_words or num == 0) and tmp_seg_words:
                ret.append(Chunk(tmp_seg_words, self.chrs_dic))
            else:
                # 否则，遍历match_words，从中依次取词，在原句中去除该词进行递归查找，然后num-1以及将当前word加入到tmp_seg_words中。
                for word in match_words:
                    iter_chunk(sentence[len(word):], num - 1, tmp_seg_words + [word])

        iter_chunk(sentence, num=3, tmp_seg_words=[])
        return ret

    def cws(self, sentence):
        while sentence:
            chunks = self.get_chunks(sentence)  # 接收返回的chunks
            # 将之前每种的分词评分运用①~④的消歧规则的进行依次比较，选取出当前最优解，
            # 然后在最优解中选取第一个词作为已分好的词，剩下的词重新当成参数传入到get_chunks方法中
            if chunks:
                word = max(chunks).words[0]
                yield word
                sentence = sentence[len(word):]
            else:
                yield sentence[0]
                sentence = sentence[1:]


if __name__ == "__main__":
    # txt = "我们去马尔代夫结婚吧"
    # txt = "然后在最优解中选取第一个词作为已分好的词，剩下的词重新当成参数传入到方法中"
    txt = "然后在最优解中选取第一个词作为已分好的词，剩下的词重新当成参数传入到get_chunks方法中"
    mmseg = MMSeg('char_dic', 'new_lex.txt')
    print(list(mmseg.cws(txt)))
