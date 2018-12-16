import numpy as np # linear algebra
from numpy import *
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import re
import nltk
import nltk.stem
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Dense, Activation, Flatten, Convolution2D, Dropout, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
# import nltk
# nltk.download('punkt')
from sklearn.tree import DecisionTreeClassifier
import csv

##----清洗数据-----##
def cleandata(comment):
    comment_list = []
    for text in comment:
        # 单词都为小写
        text = text.lower()
        # 删除非字母、非数字成分
        text = re.sub(r"[^a-z'&^1-9]", " ", text)

        # 将简写单词复原
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"\'m", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " will ", text)
        text = re.sub(r"ain\'t", " are not ", text)
        text = re.sub(r"aren't", " are not ", text)
        text = re.sub(r"couldn\'t", " can not ", text)
        text = re.sub(r"didn't", " do not ", text)
        text = re.sub(r"doesn't", " do not ", text)
        text = re.sub(r"don't", " do not ", text)
        text = re.sub(r"hadn't", " have not ", text)
        text = re.sub(r"hasn't", " have not ", text)
        text = re.sub(r"\'ll", " will ", text)
        #利用nlp工具包nltk进行词干提取
        core_text = ""
        s = nltk.stem.snowball.EnglishStemmer()
        for word in word_tokenize(text):
            core_text = core_text + " " + s.stem(word)
        # 放回去
        comment_list.append(core_text)
    return comment_list
##-------读文件-----------##
def read_data(file):
    train_data = csv.reader(open(file, encoding="utf-8"))
    lines = 0
    for r in train_data:
        lines += 1
    train_data_label = np.zeros([lines - 1, ])
    train_data_content = []
    train_data = csv.reader(open(file, encoding="utf-8"))
    i = 0
    for data in train_data:
        if data[0] == "Label" or data[0] == "SmsId":
            continue
        if data[0] == "ham":
            train_data_label[i] = 0
        if data[0] == "spam":
            train_data_label[i] = 1
        train_data_content.append(data[1])
        i += 1
    print(train_data_label.shape, len(train_data_content))
    return train_data_label,train_data_content
##-------sigmoid函数模拟---------##
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
##--------改进的随机梯度上升法-------##
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix) #获得训练集的维数m*n
    weights = ones(n)   #初始化weight为全1向量
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001   #alpha会随着每次迭代从而减小，这样可以减小随机梯度上升的回归系数波动问题，同时也
                                        #同时比普通梯度上升收敛更快，加入常数项避免alpha变成0
            randIndex = int(random.uniform(0,len(dataIndex)))#随机选一个值更新回归系数
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h#计算预测误差
            weights = weights + alpha * error * dataMatrix[randIndex] #更新weights值
            del (list(dataIndex)[randIndex])
    return weights
##--------读取数据-------##
train_y,train_data_content = read_data("train.csv")
_,test_data_content = read_data("test.csv")
train_data_content = cleandata(train_data_content)
test_data_content = cleandata(test_data_content)
print(train_y)
##-------TF-IDF 将字符串转换为向量--------##
all_comment_list = list(train_data_content) + list(test_data_content)
text_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',token_pattern=r'\w{1,}',
                              max_features=5000, ngram_range=(1, 1), analyzer='word')
text_vector.fit(all_comment_list)
train_x = text_vector.transform(train_data_content)
test_x = text_vector.transform(test_data_content)
train_x = train_x.toarray()
test_x = test_x.toarray()
print(train_x.shape,test_x.shape,type(train_x),type(test_x))
test_y=zeros((test_x.shape[0],len(train_y)))
test_y=np.array(test_y)
print(test_x.shape,test_y.shape)
##------计算权值---------##
trainWeights = stocGradAscent1(train_x,train_y, 2)
lxt1,lxt2 = shape(test_x)
##-----对测试集进行预测------##
for jj in range(test_x.shape[0]):
    test_y[int(jj),0]= sigmoid(sum(train_x[int(jj)] * trainWeights))  # 利用sigmoid函数生成预测值，若大与0.5则判断为1



##-----------输出结果、制成submission.csv-------##
print(test_y.shape)
answer = pd.read_csv(open("sampleSubmission.csv"))
for i in range(test_y.shape[0]):
    predit = test_y[i,0]
    if predit < 0.5:
        answer.loc[i,"Label"] = "spam"
    else:
        answer.loc[i,"Label"] = "ham"
answer.to_csv("submission.csv",index=False)  # 不要保存引索／
