# -*- coding: utf-8 -*-
import jieba.posseg as jp
import pyLDAvis.gensim
from gensim import corpora, models
from xunfei import audio_to_text

#记得修改path
stopwordpath = '/Users/liyufei/techx/hackthon/stopwords.txt'
datapath = '/Users/liyufei/techx/hackthon/train_data.txt'
keywordspath = '/Users/liyufei/techx/hackthon/keywords.txt'

audio_test_path = './test.wav'

text_test = ['儿子们过年才会回来，女儿倒是经常回来']


def get_documents(datapath):
    #打开文件
    with open(datapath, 'r') as file:
        list = file.readlines()
    for i in range(0, len(list)):
        list[i] = list[i].rstrip('\n')

    return list

#将每个字分出来
def get_text(texts):
    # 获取停用词
    stopword_path = stopwordpath
    with open(stopword_path, 'r') as file:
        stopword = file.readlines()
    print(stopword)
    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
    additional_stopword = ['有','也','干','是','没有','没','还','不',
                           '再','来','回来','机会','来','捡','算','只',
                           '去','人','都','太','又','会','才','就','最',
                           '能','要','到','能够','需要','满足','已经',
                           '实行','怕','决定','像','很','害','想','经','越来越','好好']
    stopwords = stopword + additional_stopword
    words_list = []
    for text in texts:
        words = [w.word for w in jp.cut(text) if w.flag in flags and w.word not in stopwords]
        words_list.append(words)
        words_list = [x for x in words_list if x != []]
    return words_list


#字典Dictionary(42 unique tokens: ['国', '复', '兴', '太', '极']...)
def get_dictionary(words_list):
    # 构造词典
    # Dictionary()方法遍历所有的文本，为每个不重复的单词分配一个单独的整数ID，同时收集该单词出现次数以及相关的统计信息
    dictionary = corpora.Dictionary(words_list)

    return dictionary

def get_corpus(words_list, dictionary):

    # 将dictionary转化为一个词袋
    # doc2bow()方法将dictionary转化为一个词袋。得到的结果corpus是一个向量的列表，向量的个数就是文档数。
    # 在每个文档向量中都包含一系列元组,元组的形式是（单词 ID，词频）
    corpus = [dictionary.doc2bow(words) for words in words_list]
    # print('输出每个文档的向量:/corpus')
    # print(corpus)  # 输出每个文档的向量

    return corpus

# 生成LDA模型
def LDA_model(corpus, dictionary, num_topics = 3):

    lda_model = models.ldamodel.LdaModel(corpus=corpus, num_topics = num_topics, id2word=dictionary, passes=10)

    return lda_model

def make_prediction(text_list):
    print('test的文本：')
    print(text_list)
    dictionary = get_dictionary(text_list)
    corpus = get_corpus(text_list, dictionary)
    unseen_doc = corpus[0]
    print('unseen_doc:')
    print(unseen_doc)
    predict = lda_model[unseen_doc]

    return predict

def show_predicted_topic(predict):
    max = 0
    topic = 0
    for i in range(len(predict)):
        p = predict[i][1]
        if p > max:
            max = p
            topic = predict[i][0]
    return topic


data_path = datapath

texts = list(get_documents(data_path))
print(texts)
texts_list = get_text(texts)
print(texts_list)

#建立LDA model
num_topics = 9

dictionary = get_dictionary(texts_list)
corpus = get_corpus(texts_list, dictionary)

lda_model = LDA_model(corpus, dictionary, num_topics)

topic_words = lda_model.print_topics(num_topics)
print('打印所有主题:')
print(topic_words)


#使用text test
text_list = get_text(text_test)
print('prediction:')
predict = make_prediction(text_list)
print(predict)
print(show_predicted_topic(predict))

#使用audio test
other_text = audio_to_text(audio_test_path)
#print(other_text)
print('你是想说这句话吗？如果是，请输入yes；如果不是，请修改错别字')
ans_modify = input()
if ans_modify == 'yes':
    pass
else:
    other_text = ans_modify
print('收到！正在帮你归类')
text_list = get_text(other_text)
print(make_prediction(text_list))
print(show_predicted_topic(predict))


#visualization
ldamodel = lda_model
graph = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.show(graph)

#防诈骗系统
key_words_path = keywordspath
key_words = get_documents(key_words_path)

for key_word in key_words:
    for text in text_list[0]:
        if key_word == text:
            print('注意自己财产安全！！')

# #真实test2
# prereal_text = '/Users/liyufei/techx/hackthon/test.txt'
# real_text = list(get_documents(prereal_text))
# realtext_list = get_text(real_text)
# print('real test text:')
# print(realtext_list)
# print('real test prediction:')
# print(make_prediction(realtext_list))
#
#
#

#

#
# #匹配相似文本
# for x in texts:
#     list_doc = [x[1] for x in topic_words]
#
# list_doc1 = [i[1] for i in topic_words[0]]
# list_doc2 = [a[1] for a in topic_words[1]]









