import numpy as np
from math import sqrt
from collections import Counter
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

class KNNClassifier(object):
    '''定义一个KNN分类器，手动实现KNN算法'''
    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """加载训练集"""
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, x):
        """计算由knn算法得到的标签"""
        distances = [sqrt(np.sum((np.array(x_train) - np.array(x))**2)) for x_train in self._X_train]#欧氏距离
        nearest = np.argsort(distances)#排序
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def evalue(self, X_test, y_test):
        """模型测试"""
        y_pred=[]
        for xi, yi in zip(X_test, y_test):
            y_ = self.predict(xi)
            y_pred.append(y_)
        accuracy = accuracy_score(y_test,y_pred)   
        self.show(accuracy,y_pred,y_test) 
        return y_pred
    
    def show(self,accuracy,y_pred,y_test):
        '''展示精确度、分类报告、混淆矩阵数据'''
        print("accuracy:"+str(accuracy))
        print(classification_report(y_test, y_pred))  
        print(confusion_matrix(y_test, y_pred))  
        return self