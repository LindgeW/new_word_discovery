import math


# https://github.com/jason2506/mmseg.py/blob/master/lexicon.py
def _create_trie(words):  # 字典树
    trie = {}
    for word in words:
        ptr = trie
        for char in word:
            if char not in ptr:
                ptr[char] = {}
            ptr = ptr[char]
        ptr[''] = ''

    return trie


class Lexicon(object):
    def __init__(self, tf):
        self._trie = _create_trie(tf.keys())
        self._tf = tf

    def term_frequency(self, term):
        return self._tf.get(term, 0)

    def get_chunks(self, string, start=0, max_len=3):
        str_len = len(string)
        if max_len == 0 or start == str_len:
            yield tuple()
        else:
            ptr = self._trie
            for i in range(start, str_len):
                if string[i] not in ptr:
                    break
                ptr = ptr[string[i]]
                if '' in ptr:
                    for chunk in self.get_chunks(string, i + 1, max_len - 1):
                        yield (string[start:i + 1],) + chunk


def wrap(line):
    w, f = line.strip().split()
    f = math.log(float(f) + 1.)
    return w, f


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
        # lexicon = dict(map(wrap, fin))
    return lexicon


class Chunk:
    def __init__(self, words, chrs=None):
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
        if chrs is not None:
            self.entropy = sum([math.log(float(chrs[x])) for x in words if len(x) == 1 and x in chrs])
        else:
            self.entropy = 0.

    def __lt__(self, other):
        return (self.length, self.mean, -self.var, self.entropy) < \
           (other.length, other.mean, -other.var, other.entropy)


class MMSeg:
    def __init__(self, char_path=None, wd_path=None):
        self.chrs_dic = load_lexicon(char_path)
        self.wds_dic = Lexicon(load_lexicon(wd_path))

    def cws(self, sentence):
        while sentence:
            chunks_iter = self.wds_dic.get_chunks(sentence)  # 接收返回的chunks
            # 将之前每种的分词评分运用①~④的消歧规则的进行依次比较，选取出当前最优解，
            # 然后在最优解中选取第一个词作为已分好的词，剩下的词重新当成参数传入到get_chunks方法中
            chunks = []
            for cks in chunks_iter:
                chunks.append(cks)

            if chunks:
                word = max([Chunk(chk, self.chrs_dic) for chk in chunks]).words[0]
                yield word
                sentence = sentence[len(word):]
            else:
                yield sentence[0]
                sentence = sentence[1:]


if __name__ == "__main__":
    # txt = "我们去马尔代夫结婚吧"
    txt = "然后在最优解中选取第一个词作为已分好的词，剩下的词重新当成参数传入到get_chunks方法中"
    mmseg = MMSeg('char_dic', 'new_lex.txt')
    print(list(mmseg.cws(txt)))
