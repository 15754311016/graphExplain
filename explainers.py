from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from abc import ABCMeta, abstractmethod
from future import standard_library
import numpy as np
import scipy as sp
from sklearn import linear_model
import sklearn.metrics.pairwise
from pyHSICLasso.api import HSICLasso

standard_library.install_aliases()

from sklearn.preprocessing import StandardScaler


###############################
## Random Explainer
###############################

class RandomExplainer:
    def __init__(self):
        pass

    def reset(self):
        pass

    def explain_instance(self,
                         instance_vector,
                         label,
                         classifier,
                         num_features,
                         dataset):
        # print("nonzero:",instance_vector.nonzero())
        instance_vector = instance_vector.reshape(1, len(instance_vector))
        nonzero = instance_vector.nonzero()[1]
        explanation = np.random.choice(nonzero, num_features)
        return [(x, 1) for x in explanation]

    def explain(self,
                train_vectors,
                train_labels,
                classifier,
                num_features,
                dataset):
        i = np.random.randint(0, train_vectors.shape[0])
        explanation = self.explain_instance(train_vectors[i], None, None,
                                            num_features, dataset)
        return i, explanation


###############################
## Standalone Explainers
###############################

def most_important_word(classifier, v, class_):
    # Returns the word w that moves P(Y) - P(Y|NOT w) the most for class Y.
    max_index = 0
    max_change = -1
    orig = classifier.predict_proba(v)[0][class_]
    for i in v.nonzero()[1]:
        val = v[0, i]
        v[0, i] = 0
        pred = classifier.predict_proba(v)[0][class_]
        change = orig - pred
        if change > max_change:
            max_change = change
            max_index = i
        v[0, i] = val
    if max_change < 0:
        return -1
    return max_index


def explain_greedy(instance_vector,
                   label,
                   classifier,
                   num_features,
                   dataset=None):
    explanation = []
    z = instance_vector.copy()
    while len(explanation) < num_features:
        i = most_important_word(classifier, z, label)
        if i == -1:
            break
        z[0, i] = 0
        explanation.append(i)
    return [(x, 1) for x in explanation]


def most_important_word_martens(fea_data, index, predict_fn, v, class_):
    # Returns the word w that moves P(Y) - P(Y|NOT w) the most for class Y.
    max_index = 0
    max_change = -1
    # orig = predict_fn[index,class_]
    orig = softmax(predict_fn.forward(np.array([index])).data.numpy())[:, class_]
    # print("v:",v,v.shape)
    for i in v.nonzero()[1]:
        val = v[0, i]
        v[0, i] = 0
        fea_data[index] = v
        predict_fn.enc1.features.weight = nn.Parameter(torch.FloatTensor(fea_data))
        # pred = predict_fn[index,class_]
        pred = softmax(predict_fn.forward(np.array([index])).data.numpy())[:, class_]
        change = orig - pred
        if change > max_change:
            max_change = change
            max_index = i
        v[0, i] = val
        fea_data[index] = v
    if max_change < 0:
        return -1, max_change
    return max_index, max_change


def explain_greedy_martens(index, fea_data, instance_vector,
                           label,
                           predict_fn,
                           num_features,
                           dataset=None):
    # if not hasattr(predict_fn, '__call__'):
    #     predict_fn = predict_fn.predict_proba
    explanation = []
    instance_vector = instance_vector.reshape(1, -1)
    z = instance_vector.copy()
    # print("predict_fn:",predict_fn,predict_fn.shape)
    # print(predict_fn[0])
    # cur_score = predict_fn[index,label]
    cur_score = softmax(predict_fn.forward(np.array([index])).data.numpy())[:, label]
    while len(explanation) < num_features:
        i, change = most_important_word_martens(fea_data, index, predict_fn, z, label)
        cur_score -= change
        if i == -1:
            break
        explanation.append(i)
        if cur_score < .5:
            break
        z[0, i] = 0
    return [(x, 1) for x in explanation]


import copy
import argparse
import collections
import torch
import torch.nn as nn
from graphsage.model import load_cora
from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
from graphsage.model import SupervisedGraphSage


def define_graphsage_model(feat_data, labels, adj_lists):
    # feat_data, labels, adj_lists = load_cora()
    # print(adj_lists)
    features = nn.Embedding(2708, 1443)  # what does it mean
    # print("features:", feat_data[0])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # print("features.data:",np.array(nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False).data[0]))
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1443, 128, adj_lists, agg1, gcn=True, cuda=False)
    # print("aaaaaa:",enc1([1,2,3]))
    # print("start")
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    # print("enc1.embed_dim", enc1.embed_dim)
    enc1.num_samples = 5
    enc2.num_samples = 5
    graphsage = SupervisedGraphSage(2, enc2, enc1, agg1)
    graphsage.load_state_dict(torch.load('./parameter_pubmed.pkl'))
    return graphsage


def softmax(X):
    exp_X = np.exp(X)
    # print(exp_X)
    sum_ = list(np.sum(exp_X, axis=1))
    result = np.zeros(X.shape)
    for i in range(X.shape[0]):
        result[i] = exp_X[i] / sum_[i]
    return result


def data_labels_distances_mapping_text(index, feat_data, x, classifier_fn, num_samples,
                                       adj_lists):  # 记录训练样本中的非0元素索引，扰动数据，得到扰动数据的预测结果
    # distance_fn = lambda x : sklearn.metrics.pairwise.cosine_distances(x[0],x)[0] * 100
    # print("X:",x,x.shape)
    feat_data_1 = copy.deepcopy(feat_data)
    x = x.reshape(1, -1)
    features = x.nonzero()[1]  # 非零特征的索引
    # print("features:",features)
    # print("len of features:", len(features))
    vals = np.array(x[x.nonzero()])[0]  # 非零特征的特征值
    # print("vals:",vals,vals.shape)
    # print("sp.sparse.find(x):",sp.sparse.find(x))
    # print("x:",x,x.shape)
    doc_size = len(sp.sparse.find(x)[2])
    # print("doc_size:",doc_size)
    sample = np.random.randint(1, doc_size, num_samples - 1)  # sample:[1,2,4,3。。。。]是一个随机数的数组，长度为len(14999)
    # print("len:",len(sample))
    data = np.zeros((num_samples, len(features)))
    inverse_data = np.zeros((num_samples, len(features)))
    data[0] = np.ones(doc_size)
    inverse_data[0] = vals
    features_range = range(len(features))  # features_range是一个0~len（features）的数组
    # 以下代码开始对一个sample进行15000的扰动，得到训练数据
    for i, s in enumerate(sample, start=1):
        active = np.random.choice(features_range, s, replace=False)  # 从feature_range中随机选择s个
        # print("active:",active)
        data[i, active] = 1
        for j in active:
            inverse_data[i, j] = 1
    # print("yesss",(data==inverse_data).all())
    # print(inverse_data.shape[0], x.shape[1])
    sparse_inverse = sp.sparse.lil_matrix((inverse_data.shape[0], x.shape[1]))
    sparse_inverse[:, features] = inverse_data
    sparse_inverse = sp.sparse.csr_matrix(sparse_inverse)
    sparse_inverse = sparse_inverse.toarray()
    mapping = features
    # labels = classifier_fn(sparse_inverse)
    # classifier = define_graphsage_model(feat_data)
    # print("labels:",labels.shape)
    # distances = distance_fn(sparse_inverse)
    # print("distances",distances)
    labels = np.zeros((data.shape[0], 2))
    graphsage = classifier_fn
    for i in range(data.shape[0]):
        # print(i)
        feat_data_1[index] = sparse_inverse[i]
        graphsage.enc1.features.weight = nn.Parameter(torch.FloatTensor(feat_data_1))
        label = softmax(graphsage.forward(np.array([index])).data.numpy())
        labels[i] = label
    return data, labels, mapping


# This is LIME
class GeneralizedLocalExplainer:
    def __init__(self,
                 fea_data,
                 kernel_fn,
                 data_labels_distances_mapping_fn,
                 num_samples=5000,
                 lasso=True,
                 mean=None,
                 return_mean=False,
                 return_mapped=False,
                 lambda_=None,
                 verbose=True,
                 positive=False):
        # Transform_classifier, transform_explainer,
        # transform_explainer_to_classifier all take raw data in, whatever that is.
        # perturb(x, num_samples) returns data (perturbed data in f'(x) form),
        # inverse_data (perturbed data in x form) and mapping, where mapping is such
        # that mapping[i] = j, where j is an index for x form.
        # distance_fn takes raw data in. what we're calling raw data is just x
        self.fea_data = fea_data
        self.lambda_ = lambda_
        self.kernel_fn = kernel_fn
        self.data_labels_distances_mapping_fn = data_labels_distances_mapping_fn
        self.num_samples = num_samples
        self.lasso = lasso
        self.mean = mean
        self.return_mapped = return_mapped
        self.return_mean = return_mean
        self.verbose = verbose
        self.positive = positive

    def softmax(self, X):
        exp_X = np.exp(X)
        # print(exp_X)
        sum_ = list(np.sum(exp_X, axis=1))
        result = np.zeros(X.shape)
        for i in range(X.shape[0]):
            result[i] = exp_X[i] / sum_[i]
        return result

    def data_map_sample_neighbors(self, i, adj_list, feat_data, classfier):
        neighbors = list(adj_list[i])
        neighbors1 = []
        for j in neighbors:
            neighbors1.extend(list(adj_list[j]))
        neighbors1.extend(neighbors)
        neighbors2 = []
        for j in neighbors1:
            neighbors2.extend(list(adj_list[j]))
        neighbors2.extend(neighbors1)
        neighbors = list(set(neighbors2))
        data = feat_data[neighbors]
        labels = self.softmax(classfier.forward(neighbors).data.numpy())
        # labels = classfier.forward(neighbors).data.numpy()
        # print("!!!!!!!:",data)
        return data, labels

    def reset(self):
        pass

    def data_labels_distances_mapping(self, raw_data, classifier_fn):
        data, labels, distances, mapping = self.data_labels_distances_mapping_fn(raw_data, classifier_fn,
                                                                                 self.num_samples)
        return data, labels, distances, mapping

    def generate_lars_path(self, weighted_data, weighted_labels):
        X = weighted_data
        alphas, active, coefs = linear_model.lars_path(X, weighted_labels, method='lasso', verbose=False,
                                                       positive=self.positive)
        return alphas, coefs

    def explain_instance_with_data(self, data, labels, label, num_features, method):
        # weights = self.kernel_fn(distances)
        # print()
        print("data.shape",data.shape)
        # if data.shape[1] >=10:
        #     return []
        # print("data:::",data[34].sum())
        if self.mean is None:
            mean = np.mean(labels[:, label])
            # print(mean)
        else:
            mean = self.mean
        Y = labels[:, label]
        # print("Y:",Y)

        # data = np.random.randn(data.shape[0],data.shape[1])
        # print("Y shape:",Y.shape)
        # ss = StandardScaler()
        # data = ss.fit_transform(data)
        # print("data:", data)
        # data = data
        # print("data:",data)
        if method == "hsic_lasso":
            # print("hsic_lasso")
            hsic_lasso = HSICLasso()
            # print(data)
            hsic_lasso.input(data, Y)
            if data.shape[1] > 10:
                #     print("num_fea",data.shape[1])
                # print("numfea:", num_features)
                print("start hsic", num_features)
                hsic_lasso.regression(num_features)
                print("finish hsic")
                used_features = hsic_lasso.get_index()
                print("len of noisy features:", len([x for x in used_features if x > 1432]))
                coef = hsic_lasso.get_index_score()
                print("coef:", coef)
                print("used_features:", used_features)
                debiased_model = linear_model.Ridge(alpha=0, fit_intercept=False)
                # debiased_model.fit(data[:, used_features], Y-mean)
                # print(data[:,used_features])
                debiased_model.fit(data[:, used_features], Y)
                print("debiased_model_alpha:", debiased_model.coef_)
            else:
                print("data feature is not enough！！！！！！！！！！！！！！")
                used_features = [i for i in range(data.shape[1])]
                debiased_model = linear_model.Ridge(alpha=0, fit_intercept=False)
                # debiased_model.fit(data[:, used_features], Y-mean)
                debiased_model.fit(data[:, used_features], Y - mean)
                '''
            if self.return_mean:
                return sorted(zip(used_features,
                                  debiased_model.coef_),
                              key=lambda x: np.abs(x[1]), reverse=True), mean
            else:
    
                return sorted(zip(used_features,
                                  debiased_model.coef_),
                          key=lambda x: np.abs(x[1]), reverse=True)
                          
        '''

        if method == "K-Lasso":
            print("K_lasso")
            # weighted_data = data * weights[:, np.newaxis]
            if self.mean is None:
                mean = np.mean(labels[:, label])
                # print(mean)
            else:
                mean = self.mean
            # print("labels:",labels[:,label])
            shifted_labels = labels[:, label]  ## why does it mean the label values
            # print("shifted_labels",shifted_labels.shape)
            if self.verbose:
                print('mean', mean)
            weighted_labels = shifted_labels
            # print("weighted_labels:", weighted_labels)
            used_features = range(data.shape[1])
            nonzero = used_features
            # print("nonzero:",nonzero)
            alpha = 1
            # print("lambda",self.lambda_)
            # print("lasso",self.lasso)
            if self.lambda_:
                print("lambda")
                classif = linear_model.Lasso(alpha=self.lambda_, fit_intercept=False, positive=self.positive)
                classif.fit(data, weighted_labels)
                used_features = classif.coef_.nonzero()[0]
                if used_features.shape[0] == 0:
                    if self.return_mean:
                        return [], mean
                    else:
                        return []
            elif self.lasso:
                alphas, coefs = self.generate_lars_path(data, weighted_labels)
                for i in range(len(coefs.T) - 1, 0, -1):
                    nonzero = coefs.T[i].nonzero()[0]
                    print("nonzero:",coefs.T[i].shape)
                    if len(nonzero) <= num_features:
                        chosen_coefs = coefs.T[i]
                        alpha = alphas[i]
                        break
                used_features = nonzero
                print("real nonzero:", nonzero)
            debiased_model = linear_model.Ridge(alpha=0, fit_intercept=False)
            debiased_model.fit(data[:, used_features], weighted_labels)
            # print("debiased_model_alpha:", debiased_model.coef_)
            if self.verbose:
                print('Prediction_local', debiased_model.predict(data[0, used_features].reshape(1, -1)) + mean,
                      'Right:', labels[0, label])
        if self.return_mean:
            return sorted(zip(used_features,
                              debiased_model.coef_),
                          key=lambda x: np.abs(x[1]), reverse=True), mean
        else:
            # print("used_features:",used_features)
            # print("coefs:",debiased_model.coef_)
            return sorted(zip(used_features,
                              debiased_model.coef_),
                          key=lambda x: np.abs(x[1]), reverse=True)

    def explain_instance(self, index,
                         raw_data,
                         label,
                         classifier_fn,
                         num_features, adj_lists, method, dataset=None):
        # print("explain_instance))")
        # print(num_features)
        # print("return_mapped", self.return_mapped)
        # print("return_mean", self.return_mean)
        if not hasattr(classifier_fn, '__call__'):
            classifier_fn = classifier_fn.predict_proba
        data, labels, mapping = self.data_labels_distances_mapping_fn(index, self.fea_data, raw_data, classifier_fn,
                                                                      self.num_samples, adj_lists)
        # print("raw_data",raw_data.toarray())
        # print("data:", data, list(data[0]).count(0))
        # print("labels:", labels, labels.shape)
        # print("data",data.shape)
        # print("distance",distances,len(distances))
        if self.return_mapped:
            if self.return_mean:
                exp, mean = self.explain_instance_with_data(data, labels, label, num_features, method=method)
            else:
                exp = self.explain_instance_with_data(data, labels, label, num_features, method=method)
            exp = [(mapping[x[0]], x[1]) for x in exp]
            # print("selected features of K-lasso:",[exp2[0] for exp2 in exp])
            if self.return_mean:
                return exp, mean
            else:
                # print("exp:", exp)
                return exp

        return self.explain_instance_with_data(data, labels, label, num_features), mapping

    def explain_instance_gra(self, i,
                             adj_lists,
                             raw_data,
                             feat_data,
                             label,
                             classifier_fn,
                             num_features, method):
        # print("explain_instance))")
        # print(num_features)
        # print("return_mapped", self.return_mapped)
        # print("return_mean", self.return_mean)
        # print("hassttr:",hasattr(classifier_fn, '__call__'))
        '''
        if not hasattr(classifier_fn, '__call__'):
          classifier_fn = classifier_fn.predict_proba
        data, labels, distances, mapping = self.data_labels_distances_mapping(raw_data, classifier_fn)
        # print("raw_data",raw_data.toarray())
        # print("data",data.shape)
        # print("distance",distances,len(distances))
        '''
        data, labels = self.data_map_sample_neighbors(i, adj_lists, feat_data, classifier_fn)
        return self.explain_instance_with_data(data, labels, label, num_features, method)
        '''
        if self.return_mapped:
          if self.return_mean:
            exp, mean = self.explain_instance_with_data(data, labels, distances, label, num_features)
          else:
            exp =   self.explain_instance_with_data(data, labels, distances, label, num_features)
          exp = [(mapping[x[0]], x[1]) for x in exp]
          if self.return_mean:
            return exp, mean
          else:
            # print("exp:", exp)
            return exp
        return self.explain_instance_with_data(data, labels, distances, label, num_features), mapping
        '''
