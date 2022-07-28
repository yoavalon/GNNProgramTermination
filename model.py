import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv import GATConv

class GAT(torch.nn.Module):
    def __init__(self, ):
        super(GAT, self).__init__()
        self.layer1 = GATConv(80, 160, 1)       #feature space 80
        self.layer2 = GATConv(160, 240, 1)
        self.layer3 = GATConv(240, 180, 1, )
        self.layer4 = GATConv(180, 120, 1, )
        
        self.lin1 = nn.Linear(120,60)
        self.lin2 = nn.Linear(60,30)
        self.lin3 = nn.Linear(30,2)
        self.sm = nn.Softmax(dim =1)

    def forward(self, g, h):

        h = self.layer1(g, h)
        h = torch.relu(h)        
        h = self.layer2(g, h)    
        h = torch.relu(h)        
        h = self.layer3(g, h)    
        h = torch.relu(h)        
        h, att = self.layer4(g, h, get_attention=True)        
        h = h.squeeze() 
        
        with g.local_scope():
            g.ndata['h'] = h            
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                        
            res = torch.relu(self.lin1(hg))
            res = torch.relu(self.lin2(res))
            res = self.sm(self.lin3(res))

            return res, torch.squeeze(att)