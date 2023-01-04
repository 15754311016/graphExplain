import sys
import copy
sys.path.append('..')
import time
import numpy as np
import scipy as sp
import sklearn
import xgboost
import xgboost.sklearn
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
from graphsage.model import load_pubmed,SupervisedGraphSage,load_cora
import explainers
from load_datasets import *
from sklearn.metrics import accuracy_score
from sklearn import ensemble
from  sklearn import model_selection
import pickle
import parzen_windows
import argparse
def softmax(X):
    exp_X = np.exp(X)
    # print(exp_X)
    sum_ = list(np.sum(exp_X,axis=1))
    result = np.zeros(X.shape)
    for i in range(X.shape[0]):
        result[i] = exp_X[i]/sum_[i]
    return result
def get_random_indices(labels, class_, probability):
  nonzero = (labels == class_).nonzero()[0]
  if nonzero.shape[0] == 0 or probability == 0:
    return []
  return np.random.choice(nonzero, int(probability * len(nonzero)) , replace=False)
def add_corrupt_feature(feature_name, clean_train, clean_test, dirty_train,
                        train_labels, test_labels, class_probs_dirty, class_probs_clean, fake_prefix='FAKE'):
    """clean_train, clean_test, dirty_train will be corrupted"""
    for class_ in set(train_labels):
      indices = get_random_indices(train_labels, class_, class_probs_clean[class_])
      for i in indices:
        clean_train[i] += ' %s%s%s' % (fake_prefix, feature_name, fake_prefix)
      indices = get_random_indices(train_labels, class_, class_probs_dirty[class_])
      for i in indices:
        dirty_train[i] += ' %s%s%s' % (fake_prefix, feature_name, fake_prefix)
      indices = get_random_indices(test_labels, class_, class_probs_clean[class_])
      for i in indices:
        clean_test[i] += ' %s%s%s' % (fake_prefix, feature_name, fake_prefix)
def corrupt_dataset(independent_features, train_data, train_labels, test_data, test_labels):
    # independent_features: list [([.3, .8],[.5,.5], 3), ([.1, .1],[0, 0], 1)
    # ...]. Each element in list is a tuple (l,l2, n) where l a list
    # representing the probability of seeing the feature in each class in the
    # dirty train data, l2 is a list representing the probability of seeing the
    # feature in each class the clean test data and n is the number of features
    # with this distribution to add.
    # returns (clean_train, dirty_train, clean_test)
    dirty_train = copy.deepcopy(train_data)
    clean_train = copy.deepcopy(train_data)
    clean_test = copy.deepcopy(test_data)
    idx = 0
    for probs, probs2, n in independent_features:
        for i in range(n):
            add_corrupt_feature('%d' % idx, clean_train, clean_test, dirty_train, train_labels, test_labels, probs, probs2)
            idx += 1
    return clean_train, dirty_train, clean_test


def train_graphsage(feat_data, labels, adj_lists,train):
  np.random.seed(1)
  random.seed(1)
  num_nodes = 2708
  # feat_data, labels, adj_lists = load_pubmed()
  features = nn.Embedding(2708, 1433+10)
  features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
  # features.cuda()

  agg1 = MeanAggregator(features, cuda=True)
  enc1 = Encoder(features, 1433+10, 128, adj_lists, agg1, gcn=True, cuda=False)
  agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
  enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                 base_model=enc1, gcn=True, cuda=False)
  enc1.num_samples = 5
  enc2.num_samples = 5

  graphsage = SupervisedGraphSage(2, enc2, enc1, agg1)
  #    graphsage.cuda()
  # rand_indices = np.random.permutation(num_nodes)
  # test = rand_indices[:1000]
  # val = rand_indices[1000:1500]
  # train = list(rand_indices[1500:])

  optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
  times = []
  for batch in range(100):
    batch_nodes = train[:256]
    random.shuffle(train)
    start_time = time.time()
    optimizer.zero_grad()
    loss = graphsage.loss(batch_nodes,
                          Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
    loss.backward()
    optimizer.step()
    end_time = time.time()
    times.append(end_time - start_time)
    # print(batch, loss.item())
  # print("acc of train:",accuracy_score(labels[train],graphsage.forward(train).data.numpy().argmax(axis=1)))
  return graphsage

def main():
  parser = argparse.ArgumentParser(description='Evaluate some explanations')
  parser.add_argument('--dataset', '-d', type=str, required=True,help='dataset name')
  parser.add_argument('--output_folder', '-o', type=str, required=True, help='output folder')
  parser.add_argument('--num_features', '-k', type=int, required=True, help='num features')
  parser.add_argument('--num_rounds', '-r', type=int, required=True, help='num rounds')
  parser.add_argument('--start_id',  '-i', type=int, default=0,required=False, help='output start id')
  args = parser.parse_args()
  dataset = args.dataset
  # train_data, train_labels, test_data, test_labels, class_names = LoadDataset(dataset)
  # fea_data, labels, adj_list = load_cora()
  # train_labels = labels[:19000]
  # test_labels = labels[19000:]
  rho = 25
  kernel = lambda d: np.sqrt(np.exp(-(d**2) / rho ** 2))
  # local = explainers.GeneralizedLocalExplainer(fea_data, kernel, explainers.data_labels_distances_mapping_text, num_samples=15000, return_mean=True, verbose=False, return_mapped=True)
  # Found through cross validation
  sigmas = {'multi_polarity_electronics': {'neighbors': 0.75, 'svm': 10.0, 'tree': 0.5,
  'logreg': 0.5, 'random_forest': 0.5, 'embforest': 0.75},
  'multi_polarity_kitchen': {'neighbors': 1.0, 'svm': 6.0, 'tree': 0.75,
  'logreg': 0.25, 'random_forest': 6.0, 'embforest': 1.0},
  'multi_polarity_dvd': {'neighbors': 0.5, 'svm': 0.75, 'tree': 8.0, 'logreg':
  0.75, 'random_forest': 0.5, 'embforest': 5.0}, 'multi_polarity_books':
  {'neighbors': 0.5, 'svm': 7.0, 'tree': 2.0, 'logreg': 1.0, 'random_forest':
  1.0, 'embforest': 3.0}}
  # parzen1 = parzen_windows.ParzenWindowClassifier()
  # parzen1.sigma = sigmas[dataset]['random_forest']
  # parzen2 = parzen_windows.ParzenWindowClassifier()
  # parzen2.sigma = sigmas[dataset]['random_forest']
  random = explainers.RandomExplainer()
  for Z in range(args.num_rounds):
    exps1 = {}
    exps2 = {}
    explainer_names = ['lime_gra', 'parzen', 'random', 'greedy', 'mutual','lime_linear']
    for expl in explainer_names:
      exps1[expl] = []
      exps2[expl] = []
    print('Round', Z)
    sys.stdout.flush()
    fake_features_z = [([.1, .2], [.1,.1], 10)]#, ([.2, .1], [.1,.1], 10)]
    # clean_train, dirty_train, clean_test = corrupt_dataset(fake_features_z, train_data, train_labels, test_data, test_labels)
    # vectorizer = CountVectorizer(lowercase=False, binary=True)
    # dirty_train_vectors = vectorizer.fit_transform(dirty_train)
    # clean_train_vectors = vectorizer.transform(clean_train)
    # test_vectors = vectorizer.transform(clean_test)
    # terms = np.array(list(vectorizer.vocabulary_.keys()))
    # indices = np.array(list(vectorizer.vocabulary_.values()))
    # inverse_vocabulary = terms[np.argsort(indices)]
    # tokenizer = vectorizer.build_tokenizer()
    # c1 = ensemble.RandomForestClassifier(n_estimators=30, max_depth=5)
    # c2 = ensemble.RandomForestClassifier(n_estimators=30, max_depth=5)
    fea_data, labels, adj_list = load_cora()
    print("fea_data shape:",fea_data.shape)
    num_nodes = 2708
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[800:1000]
    train = list(rand_indices[1000:])
    # dirty_train_vectors = fea_data[train]
    train_labels = labels[train]
    test_labels = labels[test]
    noisy_fea = np.zeros((fea_data.shape[0],10))
    for i in range(noisy_fea.shape[0]):
      random_choice = np.random.choice(10,5)
      noisy_fea[i,random_choice] = 1
      # max_ = np.max(fea_data[i])
      # min_ = np.min(fea_data[i])
      # print("min,max:",min_,max_)
      # noisy_fea[i] = np.random.uniform(min_,max_,10)
    fea_data = np.column_stack((fea_data, noisy_fea))
    local = explainers.GeneralizedLocalExplainer(fea_data, kernel, explainers.data_labels_distances_mapping_text,
                                                 num_samples=50, return_mean=True, verbose=False, return_mapped=True)
    fea_data_greedy = copy.deepcopy(fea_data)
    # print("train_labels",train_labels)
    # print(train_labels==labels[train])
    # untrustworthy = [i for i, x in enumerate(inverse_vocabulary) if x.startswith('FAKE')]
    # for train_idx1,test_idx1 in model_selection.ShuffleSplit(n_splits=1, test_size=0.2).split(dirty_train_vectors):
    #     train_idx, test_idx = train_idx1,test_idx1
    untrustworthy = [i+1433 for i in range(10)]
    train_acc1 = train_acc2 = test_acc1 = test_acc2 = 0
    print('Trying to find trees:')
    sys.stdout.flush()
    iteration = 0
    found_tree = True
    #while np.abs(train_acc1 - train_acc2) < 0.05 or np.abs(test_acc1 - test_acc2) < 0.05 or (test_acc1<0.8 and test_acc2<0.8):
    while test_acc1<0.8:
      iteration += 1
      # c1.fit(dirty_train_vectors[train_idx], train_labels[train_idx])
      # c2.fit(dirty_train_vectors[train_idx], train_labels[train_idx])
      graphsage1 = train_graphsage(fea_data,labels,adj_list,train)
      # graphsage2 = train_graphsage(fea_data,labels,adj_list,train)
      # print("grasage1.predic:",graphsage1.forward(train).data.numpy().argmax(axis=1))
      # print("train_labels:",train_labels)
      train_acc1 = accuracy_score(labels[train],graphsage1.forward(train).data.numpy().argmax(axis=1))
      # train_acc2 = accuracy_score(labels[train],graphsage2.forward(train).data.numpy().argmax(axis=1))
      # train_acc1 = accuracy_score(train_labels[test_idx], c1.predict(dirty_train_vectors[test_idx]))
      # train_acc2 = accuracy_score(train_labels[test_idx], c2.predict(dirty_train_vectors[test_idx]))
      test_acc1 = accuracy_score(test_labels, graphsage1.forward(test).data.numpy().argmax(axis=1))
      # test_acc2 = accuracy_score(test_labels, graphsage2.forward(test).data.numpy().argmax(axis=1))
      print('Train acc1:', train_acc1, 'Train acc2:', train_acc2)
      print('Test acc1:', test_acc1, 'Test acc2:', test_acc2)
      if iteration == 3000:
        found_tree = False
        break
    if not found_tree:
      print('skipping iteration', Z)
      continue
    print('done')
    print('Train acc1:', train_acc1, 'Train acc2:', train_acc2)
    print('Test acc1:', test_acc1, 'Test acc2:', test_acc2)
    sys.stdout.flush()
    # predictions = c1.predict(dirty_train_vectors)
    predictions = graphsage1.forward(test).data.numpy().argmax(axis=1)
    # predictions2 = c2.predict(dirty_train_vectors)
    # predictions2 = graphsage2.forward(test).data.numpy().argmax(axis=1)
    predict_probas = softmax(graphsage1.forward(test).data.numpy())[:,1]
    predict_probas_greedy = softmax(graphsage1.forward(test).data.numpy())
    # predict_probas2 = softmax(graphsage2.forward(test).data.numpy())[:,1]
    # predict_probas2_greedy = softmax(graphsage2.forward(test).data.numpy())
    # cv_preds1 = model_selection.cross_val_predict(c1, dirty_train_vectors[train_idx], train_labels[train_idx], cv=5)
    # cv_preds2 = model_selection.cross_val_predict(c2, dirty_train_vectors[train_idx], train_labels[train_idx], cv=5)
    # parzen1.fit(dirty_train_vectors[train_idx], cv_preds1)
    # parzen2.fit(dirty_train_vectors[train_idx], cv_preds2)
    pp = []
    pp2 = []
    true_labels = []
    iteration = 0
    for ii,i in enumerate(test):
      print("~~~~",ii,i)
      print()
      if iteration % 50 == 0:
        print(iteration)
        sys.stdout.flush()
      iteration += 1
      pp.append(predict_probas[ii])
      # pp2.append(predict_probas2[ii])
      true_labels.append(labels[i])

      exp = explainers.explain_greedy_martens(i, fea_data_greedy, fea_data_greedy[i], predictions[ii], graphsage1,
                                              args.num_features)
      exps1['greedy'].append(exp)
      print("greedy exp:", exp)

      exp, mean = local.explain_instance(i, fea_data[i],1,graphsage1, args.num_features,adj_list,method ='K-Lasso')
      print("K-lasso!!!")
      print("exp:",exp)
      exp1, mean1 = local.explain_instance_gra(i, adj_list, None, fea_data, 1, graphsage1, args.num_features, method='hsic_lasso')
      exps1['lime_linear'].append((exp, mean))
      exps1['lime_gra'].append((exp1, mean1))

      # exp = parzen1.explain_instance(dirty_train_vectors[i], 1, c1.predict_proba, args.num_features, None)
      # mean = parzen1.predict_proba(dirty_train_vectors[i])[1]
      # exps1['parzen'].append((exp, mean))

      exp = random.explain_instance(fea_data[i], 1, None, args.num_features, None)
      exps1['random'].append(exp)



      '''
      # Classifier 2
      exp, mean = local.explain_instance(i, fea_data[i],1, graphsage2, args.num_features,adj_list,method = 'K-Lasso')
      exps2['lime_linear'].append((exp, mean))
      exp1, mean1 = local.explain_instance_gra(i, adj_list, None, fea_data, 1, graphsage2, args.num_features,
                                               method='hsic_lasso')
      exps2['lime_gra'].append((exp1, mean1))
      # exp = parzen2.explain_instance(dirty_train_vectors[i], 1, c2.predict_proba, args.num_features, None)
      # mean = parzen2.predict_proba(dirty_train_vectors[i])[1]
      # exps2['parzen'].append((exp, mean))

      exp = random.explain_instance(fea_data[i], 1, None, args.num_features, None)
      exps2['random'].append(exp)
      #
      exp = explainers.explain_greedy_martens(i, fea_data,fea_data[i], predictions[ii], graphsage2, args.num_features)
      exps2['greedy'].append(exp)
'''
    out = {'true_labels' : true_labels, 'untrustworthy' : untrustworthy, 'train_acc1' :  train_acc1, 'train_acc2' : train_acc2, 'test_acc1' : test_acc1, 'test_acc2' : test_acc2, 'exps1' : exps1, 'exps2': exps2, 'predict_probas1': pp, 'predict_probas2': pp2}
    pickle.dump(out, open(os.path.join(args.output_folder, 'comparing_%s_%s_%d.pickle' % (dataset, args.num_features, Z + args.start_id)), 'wb'))



if __name__ == "__main__":
    main()
