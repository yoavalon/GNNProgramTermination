import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv import GATConv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 
import torch.functional as F
from torch.utils.tensorboard import SummaryWriter
from scikitplot.metrics import plot_roc
from scikitplot.metrics import plot_precision_recall
from datetime import datetime
from torchmetrics import Accuracy, Precision, Recall

from model import * 
from utils import *

fname = './models/model' #path for model

ct = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
writer = SummaryWriter('./tests/' + ct)

model = GAT()

#load model
#model.load_state_dict(torch.load(f'models/model'))
model = torch.load(f'models/model')

#configure for test
model.eval() 

batchsize = 100
g, l, f = getBatch(batchsize, False)

model.eval()

ep = 0
graphs = dgl.batch(g)    
logits, attention = model(graphs, graphs.srcdata['x'])

logits = torch.argmax(logits, dim = 1)

l = torch.squeeze(torch.Tensor(l).long())

accuracy = Accuracy()
acc = accuracy(logits, l)
writer.add_scalar('metrics/acc', acc, ep)

precision = Precision()
pre = precision(logits,l)
writer.add_scalar('metrics/precision', pre, ep)

recall = Recall()#.cuda()
rec = recall(logits, l)
writer.add_scalar('metrics/recall', rec, ep)

f1 = 2*(pre*rec)/(pre+rec)
writer.add_scalar('metrics/f1', f1, ep)


graphs = dgl.batch(g)    
logits, attention = model(graphs, graphs.srcdata['x'])  

fig = plt.figure()                
N = g[0].to_networkx()
num = N.number_of_edges()
nx.draw_networkx(N, pos=nx.kamada_kawai_layout(N), alpha = 0.8, font_size=8, node_color='#c3d7e0', edge_color=attention.detach().numpy()[:num], width=5.0, edge_cmap=plt.cm.coolwarm, arrows = False) #, labels = mapping, with_labels=True, arrows=True)
plt.title(str(l[0]))
writer.add_figure("graph", fig, ep, close=True)
plt.close()

fig = plt.figure(figsize=(12, 9), dpi=120) 
N = g[0].to_networkx()
num = N.number_of_edges()
nod = np.argmax(g[0].ndata['x'].detach().numpy(),1)
labs = np.array(labelDict)[nod]
labs = dict(enumerate(labs))
nx.draw_networkx(N, pos=nx.kamada_kawai_layout(N), alpha = 0.8, font_size=6, node_color='#c3d7e0', edge_color=attention.detach().numpy()[:num], width=5.0, edge_cmap=plt.cm.coolwarm, arrows = False, labels = labs, with_labels=True) 

writer.add_figure("graphDetail", fig, ep, close=True)
plt.close()

fig, ax = plt.subplots()
plot_roc(l, logits.detach().numpy(), ax=ax)
writer.add_figure("roc", fig, ep, close=True)
plt.close()
                
fig, ax = plt.subplots()
plot_precision_recall(l, logits.detach().numpy(), ax=ax)
writer.add_figure("precision-recall", fig, ep, close=True)
plt.close()

with open(f[0], 'r') as fi :
    program = fi.read()
        
    program = program.replace('\n', '  \n    ')
    writer.add_text('programs', '    ' + program, ep)