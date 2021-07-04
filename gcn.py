import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import pickle
import gzip
import numpy as np
import time


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphEncoder(Module):
    def __init__(self, device, entity, emb_size, kg, embeddings=None, fix_emb=True, seq='rnn', gcn=True, hidden_size=100, layers=1, rnn_layer=1):
        super(GraphEncoder, self).__init__()
        self.embedding = nn.Embedding(entity, emb_size, padding_idx=entity-1)
        if embeddings is not None:
            print("pre-trained embeddings")
            self.embedding.from_pretrained(embeddings,freeze=fix_emb)
        self.layers = layers
        self.user_num = len(kg.G['user'])
        self.item_num = len(kg.G['item'])
        self.PADDING_ID = entity-1
        self.device = device
        self.seq = seq
        self.gcn = gcn

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        if self.seq == 'rnn':
            self.rnn = nn.GRU(hidden_size, hidden_size, rnn_layer, batch_first=True)
        elif self.seq == 'transformer':
            self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400), num_layers=rnn_layer)

        if self.gcn:
            indim, outdim = emb_size, hidden_size
            self.gnns = nn.ModuleList()
            for l in range(layers):
                self.gnns.append(GraphConvolution(indim, outdim))
                indim = outdim
        else:
            self.fc2 = nn.Linear(emb_size, hidden_size)

    def forward(self, b_state):
        """
        :param b_state [N]
        :return: [N x L x d]
        """
        batch_output = []
        for s in b_state:
            #neighbors, adj = self.get_state_graph(s)
            neighbors, adj = s['neighbors'].to(self.device), s['adj'].to(self.device)
            input_state = self.embedding(neighbors)
            if self.gcn:
                for gnn in self.gnns:
                    output_state = gnn(input_state, adj)
                    input_state = output_state
                batch_output.append(output_state)
            else:
                output_state = F.relu(self.fc2(input_state))
                batch_output.append(output_state)

        seq_embeddings = []
        for s, o in zip(b_state, batch_output):
            seq_embeddings.append(o[:len(s['cur_node']),:][None,:]) 
        if len(batch_output) > 1:
            seq_embeddings = self.padding_seq(seq_embeddings)
        seq_embeddings = torch.cat(seq_embeddings, dim=0)  # [N x L x d]

        if self.seq == 'rnn':
            _, h = self.rnn(seq_embeddings)
            seq_embeddings = h.permute(1,0,2) #[N*1*D]
        elif self.seq == 'transformer':
            seq_embeddings = torch.mean(self.transformer(seq_embeddings), dim=1, keepdim=True)
        elif self.seq == 'mean':
            seq_embeddings = torch.mean(seq_embeddings, dim=1, keepdim=True)
        
        seq_embeddings = F.relu(self.fc1(seq_embeddings))

        return seq_embeddings
    
    
    def padding_seq(self, seq):
        padding_size = max([len(x[0]) for x in seq])
        padded_seq = []
        for s in seq:
            cur_size = len(s[0])
            emb_size = len(s[0][0])
            new_s = torch.zeros((padding_size, emb_size)).to(self.device)
            new_s[:cur_size,:] = s[0]
            padded_seq.append(new_s[None,:])
        return padded_seq
