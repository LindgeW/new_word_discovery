def load_lexicon(dict_path):
    lexicon_set = set()
    with open(dict_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            token = line.strip()
            if token != '':
                lexicon_set.add(token)
    return lexicon_set


# 正向最大匹配: 从左到右
def forward_match(sent, lexicon, max_len=5):
    res = []
    k = 0
    cand = None
    n = len(sent)
    while k < n:
        for j in range(max_len, 0, -1):
            e = min(k+j, n)
            cand = sent[k: e]
            if cand in lexicon:
                res.append(cand)
                k += len(cand)
                break
        else:
            res.append(cand)
            k += 1
    return res


# 逆向最大匹配：从右到左
def backward_match(sent, lexicon, max_len=5):
    res = []
    k = len(sent)
    cand = None
    while k > 0:
        for j in range(max_len, 0, -1):
            s = max(0, k-j)
            cand = sent[s: k]
            if cand in lexicon:
                res.insert(0, cand)
                k -= len(cand)
                break
        else:
            res.insert(0, cand)
            k -= 1
    return res


# 双向最大匹配
def bidirect_match(sent, lex, max_len=5):
    # 分得的词尽量要少、非字典词和单字词数量要少
    res_fw = forward_match(sent, lex, max_len)
    res_bw = backward_match(sent, lex, max_len)
    if len(res_fw) > len(res_bw):
        return res_bw
    elif len(res_fw) < len(res_bw):
        return res_fw
    else:
        nb_fw_single = sum([len(seg) < 2 for seg in res_fw])
        nb_bw_single = sum([len(seg) < 2 for seg in res_bw])
        if nb_fw_single < nb_bw_single:
            return res_fw
        else:
            return res_bw


def test_():
    lex = load_lexicon('new_lex.txt')
    # sent = '研究生命的起源。'
    # sent = '我们在野生动物园玩、'
    # sent = '北京大学生前来应聘实习生岗位、'
    # sent = '我们去马尔代夫结婚吧'
    # sent = '这个年轻人和尚未结婚的金大小姐定婚了，请把手抬起来'
    # sent = '武汉市长江大桥将代表发展中国家和地区参加此次座谈会。'
    sent = "然后在最优解中选取第一个词作为已分好的词，剩下的词重新当成参数传入到get_chunks方法中"

    res = forward_match(sent, lex, 4)
    print(res)
    res = backward_match(sent, lex, 4)
    print(res)
    res = bidirect_match(sent, lex, 4)
    print(res)
