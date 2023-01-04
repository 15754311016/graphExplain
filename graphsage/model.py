import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score,accuracy_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
def softmax(X):
    exp_X = np.exp(X)
    # print(exp_X)
    sum_ = list(np.sum(exp_X,axis=1))
    result = np.zeros(X.shape)
    for i in range(X.shape[0]):
        result[i] = exp_X[i]/sum_[i]
    return result


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc, enc1, agg1):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.enc1 = enc1
        self.agg1 = agg1
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        # print("::::",nodes)
        embeds = self.enc(nodes)
        # print("embeds:",embeds.shape)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("./cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            # print(i,info)
            feat_data[i,:] = list(map(float, info[1:-1]))
            # print("count nonzero:",len(feat_data[i,:].nonzero()[0]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("./cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    # print(adj_lists)
    # print("labels:",list(list(labels.reshape(1,-1))[0]))
    # print("labels:",list(list(labels.reshape(1,-1))[0]).count(0))
    # print("labels:",list(list(labels.reshape(1,-1))[0]).count(1))
    # print("labels:",list(list(labels.reshape(1,-1))[0]).count(2))
    # print("labels:",list(list(labels.reshape(1,-1))[0]).count(3))
    # print("labels:",list(list(labels.reshape(1,-1))[0]).count(4))
    # print("labels:",list(list(labels.reshape(1,-1))[0]).count(5))
    # print("labels:",list(list(labels.reshape(1,-1))[0]).count(6))
    # print()
    # print('labels',labels,labels.shape)
    for i in range(labels.shape[0]):
        if labels[i,0] ==0:
            labels[i,0]=1
        else:
            labels[i,0] = 0
    return feat_data, labels, adj_lists

def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    print("adj_lists:",adj_lists)
    print(feat_data.shape)
    a = []
    for i in range(feat_data.shape[0]):
        a.append(list(feat_data[i]).count(1))
    print("a",a)
    features = nn.Embedding(2708, 1433)  #what does it mean
    # print("features:", feat_data[0])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # print("features.data:",np.array(nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False).data[0]))
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    # print("aaaaaa:",enc1([1,2,3]))
    print("start")
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    # print("enc1.embed_dim", enc1.embed_dim)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(2, enc2,enc1,agg1)
#    graphsage.cuda()
#     print("~~~")
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1000:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)  ##??
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        # print("batch_nodes:",batch_nodes)
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print("loss:",loss.data)
        print(batch, loss.item())
        if batch ==99:
            # torch.save(graphsage,'/graphsage.pkl')
            # torch.save(graphsage.state_dict(), './parameter.pkl')
            print("save successfully")
    val_output = graphsage.forward(val)
    result = softmax(val_output.data.numpy())
    # print("val_output:",val_output.data.numpy(),val_output.data.numpy().shape)
    print(result)
    # print("val_output:",val_output.data.numpy().argmax(axis=1))
    print("val_output:", result.argmax(axis=1))
    # print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Validation F1:", f1_score(labels[val], result.argmax(axis=1), average="micro"))
    print("acc of train:", accuracy_score(labels[train],graphsage.forward(train).data.numpy().argmax(axis=1)))
    print("Average batch time:", np.mean(times))

def test_corasave():
    feat_data, labels, adj_lists = load_cora()
    # print(adj_lists)
    features = nn.Embedding(2708, 1433)  # what does it mean
    # print("features:", feat_data[0])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # print("features.data:",np.array(nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False).data[0]))
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    # print("aaaaaa:",enc1([1,2,3]))
    print("start")
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    # print("enc1.embed_dim", enc1.embed_dim)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
    graphsage.load_state_dict(torch.load('\parameter.pkl'))
    result = graphsage.forward([i for i in range(1000)]).data.numpy()
    # print(result.data.numpy().argmax(axis=1))
    val = [i for i in range(1000)]
    print(labels[val].reshape(1,1000))
    print("Validation F1:", f1_score(labels[val], result.argmax(axis=1), average="micro"))

from sklearn import preprocessing
def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("./pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("./pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    for i in range(labels.shape[0]):
        if labels[i, 0] == 1:
            labels[i, 0] = 1
        else:
            labels[i, 0] = 0
    print(feat_data)
    feat_data = preprocessing.minmax_scale(feat_data)
    print("scale:",feat_data)
    return feat_data, labels, adj_lists

def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(2, enc2,enc1,agg1)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val)
    print("val_out:",val_output.data.numpy())
    print("val_out:",softmax(val_output.data.numpy()))
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))
    print("~~~~~~~~~~~~~~~~~~~")


if __name__ == "__main__":
    run_cora()
    # load_cora()
    # test_corasave()
    # run_pubmed()