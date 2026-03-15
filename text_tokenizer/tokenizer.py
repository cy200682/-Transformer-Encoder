import re#正则表示库，用来文本清洗
from collections import Counter#统计词频的工具·

class Tokenizer:

    def __init__(self, max_vocab_size=20000):#词表限制最大20000，排除低频词
        self.max_vocab_size = max_vocab_size
        self.word2idx = {
            "<pad>": 0,
            "<unk>": 1
        }
#word->index,pad补全长度，数据通常是tensor，要求一个矩阵，但是句子长度经常不一样，所以需要补全长度,unk,不在词表的未知词
        self.idx2word = {
            0: "<pad>",
            1: "<unk>"
        }
#index->word
    def tokenize(self, text):
        text = text.lower()
#转小写，减少词表大小
        text = re.sub(r"[^a-zA-Z ]", "", text)
#删除非字母字符，（不能判断情绪，正话反说也不太行感觉）
#^表示非，r"[^a-zA-Z ]"意思非字母空格
        tokens = text.split()
        return tokens
    def build_vocab(self, texts):
        counter = Counter()
        #空词频
        for text in texts:
            tokens = self.tokenize(text)

            counter.update(tokens)

        most_common = counter.most_common(self.max_vocab_size-2)
#取最常见的20000个单词，-2是unk和pad，要不然会超
        for i, (word, _) in enumerate(most_common, start=2):
#enumerate自动编号，start从2开始，0是pad，1是unk
            self.word2idx[word] = i
            self.idx2word[i] = word
    def encode(self, text):
        tokens = self.tokenize(text)

        ids = []

        for token in tokens:

            ids.append(self.word2idx.get(token, 1))
            #get(key, default)字典函数，如果不存在返回default，这个就是返回1(unk)
        return ids

    def decode(self, ids):

        words = []

        for i in ids:
            words.append(self.idx2word.get(i, "<unk>"))

        return " ".join(words)