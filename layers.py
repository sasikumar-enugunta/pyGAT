import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import construct_node_embeddings

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)))
        nn.init.normal_(self.W.data, mean=0, std=0.1)
        self.a = nn.Parameter(torch.empty(size=(self.in_features + 5, 1)))
        nn.init.normal_(self.a.data, mean=0, std=0.1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, ids, lstm_out, edges_list, adj_mat, first):
        final_lstm_dict = {}
        if first:
            lstm_dict, lstm_count = construct_node_embeddings(ids, lstm_out)
            for key, value in lstm_dict.items():
                if key in lstm_count:
                    final_lstm_dict[key] = value / lstm_count[key]
        if final_lstm_dict:
            dict_values = np.array(list(final_lstm_dict.values()))
            hij = torch.tensor(dict_values, dtype=torch.float)       # node embeddings
            wh = torch.mm(hij, self.W)
            return self._node_edge_embeddings(wh, edges_list, adj_mat)
        else:
            wh = torch.mm(lstm_out, self.W)
            return self._node_edge_embeddings(wh, edges_list, adj_mat)

    def _construct_node_edge_embeddings(self, wh, rij):
        n = wh.size()[0]
        ti = wh.repeat_interleave(n, dim=0)
        tj = wh.repeat(n, 1)
        hij = torch.cat([ti, rij, tj], dim=1)
        node_edge_embeddings = self.leakyrelu(torch.matmul(hij, self.a).squeeze(1))
        return node_edge_embeddings.view(n, n)

    def _node_edge_embeddings(self, wh, edges_list, adj_mat):
        node_edge_embeddings = self._construct_node_edge_embeddings(wh, edges_list)
        zero_vec = -9e15 * torch.ones_like(node_edge_embeddings)
        a = np.ones((node_edge_embeddings.shape[0], node_edge_embeddings.shape[0]))
        np.fill_diagonal(a, 0)
        adj = torch.from_numpy(a)
        attention = torch.where(adj_mat > 0, node_edge_embeddings, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, wh)
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'