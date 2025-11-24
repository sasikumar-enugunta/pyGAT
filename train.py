from __future__ import division
from __future__ import print_function

import time
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np

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
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.33, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping.')
parser.add_argument('--grad_clip', type=float, default=5.0, help='Gradient clipping threshold.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay factor.')
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints.')
parser.add_argument('--use_weighted_loss', action='store_true', default=False, help='Use class-weighted loss.')

parser.add_argument('--embedding_dim', type=int, default=500, help='Number of hidden units in LSTM layer.')
parser.add_argument('--hidden1', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--hidden3', type=int, default=128, help='Number of hidden units.')  # Increased for better capacity
parser.add_argument('--hidden4', type=int, default=64, help='Number of hidden units.')  # Increased for better capacity

args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')
print(f'Using device: {device}')

# Create checkpoint directory
os.makedirs(args.save_dir, exist_ok=True)

idx_features_labels, word_to_ix, tag_to_ix, w2v, embed_dim = load_data()
edge_embeddings = load_edge_embed_data()
coordinates_file = load_coordinates_file()

# Split into train, validation, and test sets
train_data, temp_data = train_test_split(idx_features_labels, test_size=0.3, random_state=args.seed)
val_data, test_data = train_test_split(temp_data, test_size=0.67, random_state=args.seed)  # 20% val, 10% test

print('No. Documents : ', len(idx_features_labels), 
      '\nNo. Words : ', len(word_to_ix), 
      '\nNo. Tags : ', len(tag_to_ix),
      '\nTrain Size : ', len(train_data),
      '\nVal Size : ', len(val_data),
      '\nTest Size : ', len(test_data), 
      '\n==========================')

# Calculate class weights for imbalanced classes
def calculate_class_weights(data, tag_to_ix):
    """Calculate class weights based on frequency"""
    class_counts = torch.zeros(len(tag_to_ix))
    total = 0
    for ids, sentence, tags, bold, underline, color in data:
        new_tag_list = construct_new_tags(ids, tags)
        for tag in new_tag_list:
            if tag in tag_to_ix:
                idx = tag_to_ix[tag]
                class_counts[idx] += 1
                total += 1
    
    # Calculate weights: inverse frequency with smoothing
    # Use balanced weights: n_samples / (n_classes * np.bincount(y))
    # But normalize to reasonable range
    max_count = class_counts.max().item()
    class_weights = max_count / (class_counts + 1e-6)  # Inverse frequency
    # Normalize to range [0.5, 2.0] to avoid extreme weights
    class_weights = 0.5 + 1.5 * (class_weights - class_weights.min()) / (class_weights.max() - class_weights.min() + 1e-6)
    
    print('Class distribution:', class_counts.tolist())
    print('Class weights:', class_weights.tolist())
    return class_weights

class_weights = calculate_class_weights(train_data, tag_to_ix).to(device)

model = GAT(nfeat=len(word_to_ix),
            nhid1=args.hidden1,
            nhid2=args.hidden2,
            nhid3=args.hidden3,
            nhid4=args.hidden4,
            embed=args.embedding_dim,
            tag_to_ix=tag_to_ix,
            dropout=args.dropout,
            alpha=args.alpha,
            class_weights=class_weights)

# Move model and data to device
model = model.to(device)
w2v = w2v.to(device)

# Initialize weights properly
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                torch.nn.init.zeros_(param.data)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, patience=args.patience//2)


def evaluate(data, name="Validation"):
    """Evaluate model on given dataset"""
    model.eval()
    nb_classes = len(tag_to_ix) - 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes, dtype=torch.int32)
    total_correct = 0
    total_tokens = 0
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for ids, sentence, tags, bold, underline, color in data:
            sentence_padded, bold_padded, underline_padded, color_padded = get_combined_paddings(sentence, bold, underline, color, word_to_ix)
            new_tag_list = construct_new_tags(ids, tags)
            targets = torch.tensor([tag_to_ix[t] for t in new_tag_list], dtype=torch.long).to(device)
            final_features_list = combine_all_features(sentence_padded, bold_padded, underline_padded, color_padded).to(device)
            ids_int, edges_list = get_edges_list_with_ids(ids, edge_embeddings)
            edges_list = edges_list.to(device)
            filtered_coordinates = coordinates_file.loc[coordinates_file['id'].isin(ids_int)]
            adj_mat = get_closest_neighbors_adj(filtered_coordinates, ids_int[0]).to(device)
            sentence_padded = sentence_padded.to(device)
            
            # Compute loss
            loss = model.neg_log_likelihood(w2v, ids_int, sentence_padded, targets, edges_list, final_features_list, adj_mat,
                                           use_weighted_loss=False)  # Don't use weighted loss for evaluation
            total_loss += loss.item()
            
            # Get predictions
            _, y_pred = model(w2v, ids_int, sentence_padded, edges_list, final_features_list, adj_mat)
            pred = torch.tensor(y_pred).to(device)
            
            # Update confusion matrix and accuracy
            for t, p in zip(targets.view(-1), pred.view(-1)):
                if t.int() < nb_classes and p.int() < nb_classes:
                    confusion_matrix[t.int(), p.int()] += 1
                    if t.int() == p.int():
                        total_correct += 1
                    total_tokens += 1
            count += 1
    
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    avg_loss = total_loss / count if count > 0 else 0.0
    precision, recall, f1_score = compute_f1(confusion_matrix)
    
    # Compute macro and micro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1_score)
    
    micro_precision = total_correct / total_tokens if total_tokens > 0 else 0.0
    micro_recall = total_correct / total_tokens if total_tokens > 0 else 0.0
    micro_f1 = micro_precision  # For classification, micro F1 = accuracy
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1_score,
        'confusion_matrix': confusion_matrix
    }

def train(epochs):
    best_val_f1 = -1
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        print('\nEpoch: {:04d}'.format(epoch+1))
        t = time.time()
        count = 1
        total_train_loss = 0
        
        for ids, sentence, tags, bold, underline, color in train_data:
            model.zero_grad()
            sentence_padded, bold_padded, underline_padded, color_padded = get_combined_paddings(sentence, bold, underline, color, word_to_ix)
            new_tag_list = construct_new_tags(ids, tags)
            targets = torch.tensor([tag_to_ix[t] for t in new_tag_list], dtype=torch.long).to(device)
            final_features_list = combine_all_features(sentence_padded, bold_padded, underline_padded, color_padded).to(device)
            ids_int, edges_list = get_edges_list_with_ids(ids, edge_embeddings)
            edges_list = edges_list.to(device)
            filtered_coordinates = coordinates_file.loc[coordinates_file['id'].isin(ids_int)]
            adj_mat = get_closest_neighbors_adj(filtered_coordinates, ids_int[0]).to(device)
            sentence_padded = sentence_padded.to(device)
            
            loss = model.neg_log_likelihood(w2v, ids_int, sentence_padded, targets, edges_list, final_features_list, adj_mat, 
                                           use_weighted_loss=args.use_weighted_loss)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            total_train_loss += loss.item()
            
            if count % 100 == 0:
                print("Iteration %d : loss %f " % (count, loss.item()))
            count += 1
        
        avg_train_loss = total_train_loss / len(train_data)
        print('Train Loss: {:.4f}, Time: {:.4f}s'.format(avg_train_loss, time.time() - t))
        
        # Validation
        val_metrics = evaluate(val_data, "Validation")
        print('Val Loss: {:.4f}, Val Accuracy: {:.4f}, Val Macro F1: {:.4f}, Val Micro F1: {:.4f}'.format(
            val_metrics['loss'], val_metrics['accuracy'], val_metrics['macro_f1'], val_metrics['micro_f1']))
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Early stopping and model checkpointing
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            print(f'New best model saved! Val F1: {best_val_f1:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nLoaded best model with Val F1: {best_val_f1:.4f}')
    
    return model


t_total = time.time()
trained_model = train(args.epochs)

print("\n" + "="*50)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print("="*50)

# Final evaluation on test set
print("\nEvaluating on Test Set...")
test_metrics = evaluate(test_data, "Test")

print("\n" + "="*50)
print("FINAL TEST RESULTS")
print("="*50)
print(f"Test Loss: {test_metrics['loss']:.4f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
print(f"Test Macro Precision: {test_metrics['macro_precision']:.4f}")
print(f"Test Macro Recall: {test_metrics['macro_recall']:.4f}")
print(f"Test Macro F1-Score: {test_metrics['macro_f1']:.4f}")
print(f"Test Micro Precision: {test_metrics['micro_precision']:.4f}")
print(f"Test Micro Recall: {test_metrics['micro_recall']:.4f}")
print(f"Test Micro F1-Score: {test_metrics['micro_f1']:.4f}")

print("\nPer-Class Metrics:")
print("Class\tPrecision\tRecall\t\tF1-Score")
for i, (p, r, f) in enumerate(zip(test_metrics['per_class_precision'], 
                                    test_metrics['per_class_recall'], 
                                    test_metrics['per_class_f1'])):
    print(f"{i}\t{p:.4f}\t\t{r:.4f}\t\t{f:.4f}")

print("\nConfusion Matrix:")
print(test_metrics['confusion_matrix'].numpy())
print("="*50)