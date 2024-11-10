import numpy as np
from collections import Counter
'''
N-gram 的概念

    1-gram（Unigram）：单个词，如 "我"，"有"，"朋友"。
    2-gram（Bigram）：两个相邻词的组合，如 "我有"，"有一个"。
    3-gram（Trigram）：三个相邻词的组合，如 "我有一个"，"有一个朋友"。
    在基于 N-gram 的语言模型中，我们可以用前 N-1 个词预测第 N 个词的出现概率。
'''
def generate_ngrams(text,n):
    '''
    生成文本的N-grams
    text: 输入的文本字符串
    n: N-gram的长度
    '''
    tokens = text.split()
    ngrams = []  # 初始化一个空列表来存放生成的 N-grams
    for i in range(len(tokens) - n + 1):  # 遍历所有可能的 N-gram 起始位置
        ngram = tokens[i:i + n]  # 从当前索引 i 开始，取长度为 n 的词列表
        ngram_tuple = tuple(ngram)  # 将词列表转换成元组（不可变类型）
        ngrams.append(ngram_tuple)  # 将生成的 N-gram 添加到列表中
    return Counter(ngrams)

# 示例文本
with open ('Harry Potter and the Sorcerer‘s Stone.txt','r',encoding='utf-8') as f:
    text = f.read()
# 计算2-gram
# bigrams = generate_ngrams(text, 2)
# print("2-gram:", bigrams)
# 计算3-gram
trigrams = generate_ngrams(text, 3)
print("3-gram:", trigrams)