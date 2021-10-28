from __future__ import division
from __future__ import print_function

import time
import argparse
import torch
import torch.optim as optim

from sklearn.model_selection import train_test_split
from model import GAT
from utils import load_data, get_combined_paddings, load_edge_embed_data, load_coordinates_file, compute_f1, \
                  get_closest_neighbors_adj, get_edges_list_with_ids, combine_all_features, construct_new_tags

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.33, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

parser.add_argument('--embedding_dim', type=int, default=500, help='Number of hidden units in LSTM layer.')
parser.add_argument('--hidden1', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--hidden3', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--hidden4', type=int, default=32, help='Number of hidden units.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

idx_features_labels, word_to_ix, tag_to_ix, w2v, embed_dim = load_data()
edge_embeddings = load_edge_embed_data()
coordinates_file = load_coordinates_file()

training_data, test_data = train_test_split(idx_features_labels, test_size=0.2, random_state=0)
print('No. Documents : ', len(idx_features_labels), '\nNo. Words : ', len(word_to_ix), '\nNo. Tags : ', len(tag_to_ix),
      '\nTrain Size : ', len(training_data), '\nTest Size : ', len(test_data), '\n==========================')

model = GAT(nfeat=len(word_to_ix),
            nhid1=args.hidden1,
            nhid2=args.hidden2,
            nhid3=args.hidden3,
            nhid4=args.hidden4,
            embed=args.embedding_dim,
            tag_to_ix=tag_to_ix,
            dropout=args.dropout,
            alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(epochs):
    for epoch in range(epochs):
        print('Epoch: {:04d}'.format(epoch+1))
        t = time.time()
        count = 1
        for ids, sentence, tags, bold, underline, color in training_data:
            model.zero_grad()
            sentence_padded, bold_padded, underline_padded, color_padded = get_combined_paddings(sentence, bold, underline, color, word_to_ix)
            new_tag_list = construct_new_tags(ids, tags)
            targets = torch.tensor([tag_to_ix[t] for t in new_tag_list], dtype=torch.long)
            final_features_list = combine_all_features(sentence_padded, bold_padded, underline_padded, color_padded)
            ids_int, edges_list = get_edges_list_with_ids(ids, edge_embeddings)
            filtered_coordinates = coordinates_file.loc[coordinates_file['id'].isin(ids_int)]
            adj_mat = get_closest_neighbors_adj(filtered_coordinates, ids_int[0])
            loss = model.neg_log_likelihood(w2v, ids_int, sentence_padded, targets, edges_list, final_features_list, adj_mat)
            loss.backward()
            optimizer.step()
            if count % 100 == 0:
                print("Iteration %d : loss %f " % (count, loss))
            count += 1
        print('time: {:.4f}s'.format(time.time() - t))


def compute_test():
    nb_classes = len(tag_to_ix) - 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes, dtype=torch.int32)
    y_true, y_pred_list = [], []
    with torch.no_grad():
        for ids, sentence, tags, bold, underline, color in test_data:
            sentence_padded, bold_padded, underline_padded, color_padded = get_combined_paddings(sentence, bold, underline, color, word_to_ix)
            new_tag_list = construct_new_tags(ids, tags)
            targets = torch.tensor([tag_to_ix[t] for t in new_tag_list], dtype=torch.long)
            ids_int, edges_list = get_edges_list_with_ids(ids, edge_embeddings)
            final_features_list = combine_all_features(sentence_padded, bold_padded, underline_padded, color_padded)
            filtered_coordinates = coordinates_file.loc[coordinates_file['id'].isin(ids_int)]
            adj_mat = get_closest_neighbors_adj(filtered_coordinates, ids_int[0])
            _, y_pred = model(w2v, ids_int, sentence_padded, edges_list, final_features_list, adj_mat)
            pred = torch.tensor(y_pred)
            y_true.append(targets)
            y_pred_list.append(pred)
            for t, p in zip(targets.view(-1), pred.view(-1)):
                confusion_matrix[t.int(), p.int()] += 1

    print(confusion_matrix)
    print(confusion_matrix.diag() / confusion_matrix.sum(1))
    precision, recall, f1_score = compute_f1(confusion_matrix)
    print('Precision : ', precision)
    print('Recall : ', recall)
    print('F1-Score : ', f1_score)


t_total = time.time()
train(10)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

compute_test()