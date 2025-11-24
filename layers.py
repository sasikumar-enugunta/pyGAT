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
        # Better initialization: Xavier uniform for better gradient flow
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # Attention parameter: [ti, rij, tj] where ti and tj are out_features, rij is 5
        # So total dimension is 2 * out_features + 5
        self.a = nn.Parameter(torch.empty(size=(2 * self.out_features + 5, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

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
        expected_rij_size = n * n
        if rij.size()[0] != expected_rij_size:
            # Reshape or pad rij to match expected size
            if rij.size()[0] < expected_rij_size:
                # Pad with zeros
                padding_size = expected_rij_size - rij.size()[0]
                padding = torch.zeros(padding_size, rij.size()[1], dtype=rij.dtype, device=rij.device)
                rij = torch.cat([rij, padding], dim=0)
            else:
                # Truncate to expected size
                rij = rij[:expected_rij_size]
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
        attention = self.dropout_layer(attention)
        h_prime = torch.mm(attention, wh)
        
        # Add residual connection if dimensions match
        if h_prime.size() == wh.size():
            h_prime = h_prime + wh  # Residual connection
        
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'