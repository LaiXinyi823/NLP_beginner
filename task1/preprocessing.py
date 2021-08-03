"""
processing phrase data using ngram & BOW model
"""

import numpy as np
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import  CountVectorizer
from icecream import ic

class BOW(object):
    """
    stopwords & n-gram & BOW model 
    """
    def __init__(self, filepath):
        self.corpus = pd.read_csv(filepath, sep='\t', index_col='PhraseId')
        self.phrase = self.corpus.loc[:,'Phrase']
        self.sentiment = self.corpus.loc[:,'Sentiment']
        self.stopwords = nltk.corpus.stopwords.words('english') # 载入英文的停用词表
        self.cut_model = nltk.WordPunctTokenizer() # 建立词分割模型
    
    def nomalize_corpus(self,data):
        """
        normalize text & drop english stopwords 
        """
        # 去除字符串中结尾的标点符号
        data = re.sub(r'[^a-zA-Z0-9\s]', '', string=data)
        # 是字符串变小写格式
        data = data.lower()
        # 去除字符串两边的空格
        data = data.strip()
        # 进行分词操作
        tokens = self.cut_model.tokenize(data)
        # 使用停止用词表去除停用词
        data = [token for token in tokens if token not in self.stopwords]
        # 将去除停用词后的字符串使用' '连接，为了接下来的词袋模型做准备
        data = ' '.join(data)

        return data

    def bigram_BOW(self):
        """
        using ngram word vector & BOW model, return BOW results
        """       
        Normalize_corpus = np.vectorize(self.nomalize_corpus) # 向量化文本正则化函数
        data_list = Normalize_corpus(self.phrase) # 正则化文本，去掉停用词等
        cv = CountVectorizer(ngram_range=(2, 2)) #  bigram
        cv.fit(data_list)
        vocs = cv.get_feature_names()
        corpus_array = cv.transform(data_list).toarray()
        corpus_norm_df = pd.DataFrame(corpus_array, columns=vocs)
        ic(corpus_norm_df.iloc[5,4:9])


train_data_process = BOW('./data/train.tsv')
train_data_process.bigram_BOW()
