import networkx as nx
import ast
from astmonkey import visitors, transformers
import os
import numpy as np 
import torch
import dgl

#not necessary (label Dict (which is a list) can be loaded dynamicall (in time))
labelDict = ['Module()', 'Import()', "alias(name='time', asname=None)", 'Assign()', "Name(id='x', ctx=ast.Store())", 'Constant(value=1)', "Name(id='y', ctx=ast.Store())", 'Constant(value=8)', 'Constant(value=6)', 'While()', 'Compare()', "Name(id='y', ctx=ast.Load())", "Name(id='x', ctx=ast.Load())", 'If()', 'Constant(value=0)', 'Expr()', 'Call()', "Attribute(attr='sleep', ctx=ast.Load())", "Name(id='time', ctx=ast.Load())", 'Constant(value=2)', 'Constant(value=10)', 'AugAssign()', 'Add()', 'Constant(value=17)', 'AugAssign(op=ast.Sub())', 'AugAssign(op=ast.Add())', 'For()', "Name(id='i', ctx=ast.Store())", "Name(id='range', ctx=ast.Load())", 'Constant(value=7)', 'Constant(value=16)', 'Constant(value=12)', 'Constant(value=15)', 'Constant(value=19)', 'Constant(value=18)', 'Constant(value=4)', 'Constant(value=9)', 'Constant(value=5)', 'Constant(value=14)', 'Constant(value=13)', 'Constant(value=3)', 'Constant(value=11)']

def codeToDgl(filename) : 

    with open(filename, 'r') as f :
        program = f.read()

    node = ast.parse(program)
    node = transformers.ParentChildNodeTransformer().visit(node)  

    visitor = visitors.GraphNodeVisitor()
    visitor.visit(node)
    
    N = nx.nx_pydot.from_pydot(visitor.graph)

    mapping = {}
    features = []

    for i in N._node :         
        lab = N._node[i]['label'][4:]
        lab = lab.replace(', type_comment=None', '')
        lab = lab.replace(', annotation=None', '')
        lab = lab.replace(', kind=None', '')
        lab = lab.replace(', returns=None', '')
        lab = lab.replace('type_comment=None', '')

        if lab in labelDict : 
            num = labelDict.index(lab)
        else : 
            labelDict.append(lab)
            num = labelDict.index(lab)

        mapping[i] = num        

        #manually defined max of dictinoary !
        features.append(np.eye(80)[num])

    #added !!
    #remapping before conversion
    mappingRelabel = {}
    for count, i in enumerate(N._node) :
        mappingRelabel[i] = count               
    N = nx.relabel_nodes(N, mapping=mappingRelabel)


    g = dgl.from_networkx(N)

    return g, features #also return label here


path1 = './datasets/DS_A/train/0'
path2 = './datasets/DS_A/train/1'
path3 = './datasets/DS_A/test/0'
path4 = './datasets/DS_A/test/1'

files1 = os.listdir(path1)[:500]
files2 = os.listdir(path2)[:500]
files3 = os.listdir(path3)[:100]
files4 = os.listdir(path4)[:100]


def getBatch(n = 10, train = True) :    
    
    graphs = []
    labels = []
    filenames = []

    for i in range(n) :    
        if train :     
            if np.random.rand() <0.5 :
                file = os.path.join(path1, np.random.choice(files1))
                label = np.array([0])
            else : 
                file = os.path.join(path2, np.random.choice(files2))
                label = np.array([1])
        else : 
            if np.random.rand() <0.5 :
                file = os.path.join(path3, np.random.choice(files3))
                label = np.array([0])
            else : 
                file = os.path.join(path4, np.random.choice(files4))
                label = np.array([1])

        g, features = codeToDgl(file)
        filenames.append(file)

        features = np.array(features)
        x = torch.FloatTensor(features)
        g.ndata['x'] = x

        graphs.append(g)        
        labels.append(label)
        
    return graphs, labels, filenames
