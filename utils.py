import torch
import pandas as pd
import numpy as np
import io
import torch.nn as nn


def compare(y, y_pred):
    error_index = []
    if len(y) == len(y_pred):
        for i in range(0, len(y)):
            if y[i] != y_pred[i]:
                error_index.append(i)

    return error_index


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    ids = [to_ix[w] if w in to_ix else to_ix["<UNK>"] for w in seq]  # to_ix.has_key(x)
    return torch.tensor(ids, dtype=torch.long)


def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def prepare_pad(sentences, ids):
    max_seq_len = get_maximum_sequence_length(ids)
    main_list = []
    temp_list = []
    for i in range(len(ids) - 1):
        if ids[i] != ids[i + 1]:
            temp_list.append(sentences[i])
            if temp_list:
                main_list.append(temp_list)
                temp_list = []
        elif ids[i] == ids[i + 1]:
            temp_list.append(sentences[i])
            if i + 1 == len(ids) - 1 and temp_list:
                temp_list.append(sentences[i + 1])
                main_list.append(temp_list)
                temp_list = []
    features = pad_input(main_list, max_seq_len)
    return features


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def get_maximum_sequence_length(ids):
    count = 0
    for id_ in ids:
        curr_frequency = ids.count(id_)
        if curr_frequency > count:
            count = curr_frequency
    return count


def read_file(ner_file):
    data1 = []
    ids = []
    sentences = []
    labels = []
    bold = []
    underline = []
    color = []
    seq_id = []
    seq_data = []
    seq_label = []
    seq_bold = []
    seq_underline = []
    seq_color = []
    for i in ner_file.readlines():
        i = i.replace("\n", "")
        lst = i.split(",")
        if len(lst) == 6:
            if len(lst[1]) == 1:
                seq_id.append(lst[0])
                seq_data.append(lst[1])
                seq_label.append(lst[2])
                seq_bold.append(lst[3])
                seq_underline.append(lst[4])
                seq_color.append(lst[5])
            else:
                seq_data.append(lst[1])
                for k in range(len(lst[1].split(' '))):
                    seq_id.append(lst[0])
                    seq_label.append(lst[2])
                    seq_bold.append(lst[3])
                    seq_underline.append(lst[4])
                    seq_color.append(lst[5])
        else:
            idx = " ".join(seq_id)
            seq_id.clear()
            ids.append(idx)
            sent = " ".join(seq_data)
            seq_data.clear()
            sentences.append(sent)
            label = " ".join(seq_label)
            seq_label.clear()
            labels.append(label)
            bold_ = " ".join(seq_bold)
            seq_bold.clear()
            bold.append(bold_)
            underline_ = " ".join(seq_underline)
            seq_underline.clear()
            underline.append(underline_)
            color_ = " ".join(seq_color)
            seq_color.clear()
            color.append(color_)
    for i in range(len(sentences)):
        data1.append((ids[i].split(), sentences[i].split(), labels[i].split(), bold[i].split(), underline[i].split(),
                      color[i].split()))
    return data1


def get_edges_list_with_ids(ids, edge_embeddings):
    ids_int = [int(i) for i in ids]
    df1 = edge_embeddings.loc[edge_embeddings['src'].isin(ids_int)]
    new_df = df1[['src', 'hori_dist', 'vert_dist', 'ar_one', 'ar_two', 'ar_three', 'dest']]
    numpy_df = new_df.to_numpy()
    numpy_edges = numpy_df[:, 1:6]
    edges_list = torch.FloatTensor(numpy_edges)
    return ids_int, edges_list


def get_combined_paddings(sentence, bold, underline, color, word_to_ix):
    sentence_padded = prepare_sequence(sentence, word_to_ix)
    bold_padded = prepare_sequence(bold, word_to_ix)
    underline_padded = prepare_sequence(underline, word_to_ix)
    color_padded = prepare_sequence(color, word_to_ix)
    return sentence_padded, bold_padded, underline_padded, color_padded

def combine_all_features(sentence, bold, underline, color):
    features_list = []
    for a, b, c, d in zip(sentence.tolist(), bold.tolist(), underline.tolist(), color.tolist()):
        temp_list = [a, b, c, d]
        features_list.append(temp_list)
    return torch.tensor(features_list)


def construct_new_tags(ids, tags):
    new_tags = {}
    for i in range(len(ids)):
        if ids[i] not in new_tags:
            new_tags[ids[i]] = tags[i]
    return list(new_tags.values())


def compute_f1(c):
    # https://stackoverflow.com/questions/37615544/f1-score-per-class-for-multi-class-classification
    num_classes = np.shape(c)[0]
    f1_score = np.zeros(shape=(num_classes,), dtype='float32')
    precision = np.zeros(shape=(num_classes,), dtype='float32')
    recall = np.zeros(shape=(num_classes,), dtype='float32')
    np_arr = c.cpu().detach().numpy()
    epsilon = 1e-7
    for j in range(num_classes):
        tp = np.sum(np_arr[j, j])
        fp = np.sum(np_arr[j, np.concatenate((np.arange(0, j), np.arange(j + 1, num_classes)))])
        fn = np.sum(np_arr[np.concatenate((np.arange(0, j), np.arange(j + 1, num_classes))), j])
        precision[j] = tp / (tp + fp + epsilon)
        recall[j] = tp / (tp + fn + epsilon)
        f1_score[j] = 2 * (precision[j] * recall[j]) / (precision[j] + recall[j] + epsilon)
    return precision, recall, f1_score


def construct_node_embeddings(ids, lstm_out):
    lstm_dict = {}
    lstm_count = {}
    for i in range(lstm_out.shape[0]):
        if ids[i] not in lstm_dict:
            lstm_dict[ids[i]] = np.array(lstm_out[i].tolist())
            lstm_count[ids[i]] = 1
        else:
            lstm_dict[ids[i]] = np.add(lstm_dict[ids[i]], lstm_out[i].tolist())
            lstm_count[ids[i]] += 1
    return lstm_dict, lstm_count


def getVocab(idx_features_labels):
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tag_to_ix = {}
    word_to_ix = {"<UNK>": 0}  # Vocabulary to id=0
    for ids, sentence, tags, bold, underline, color in idx_features_labels:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for bold_ in bold:
            if bold_ not in word_to_ix:
                word_to_ix[bold_] = len(word_to_ix)
        for underline_ in underline:
            if underline_ not in word_to_ix:
                word_to_ix[underline_] = len(word_to_ix)
        for color_ in color:
            if color_ not in word_to_ix:
                word_to_ix[color_] = len(word_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    tag_to_ix[START_TAG] = 7
    tag_to_ix[STOP_TAG] = 8
    return word_to_ix, tag_to_ix


def init_embeddings(vocab_size, embed_dim, uniform=0.25):
    return np.random.uniform(-uniform, uniform, (vocab_size, embed_dim))


def read_word(f):
    s = bytearray()
    ch = f.read(1)
    while ch != b' ':
        s.extend(ch)
        ch = f.read(1)
    s = s.decode('utf-8')
    return s.strip(' \n')


def generate_w2v(vocab, path='./data/invoice/'):
    pretrained_embed_file = 'GoogleNews-vectors-negative300.bin'
    vocab_size = len(vocab)
    with io.open(path + pretrained_embed_file, "rb") as f:
        header = f.readline()
        file_vocab_size, embed_dim = map(int, header.split())
        weight = init_embeddings(vocab_size, embed_dim)
        if '[PAD]' in vocab:
            weight[vocab['[PAD]']] = 0.0
        width = 4 * embed_dim
        for i in range(file_vocab_size):
            word = read_word(f)
            raw = f.read(width)
            if word in vocab:
                vec = np.fromstring(raw, dtype=np.float32)
                weight[vocab[word]] = vec
    embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
    embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
    return embeddings, embed_dim


def load_data(path="./data/invoice/", dataset="invoice"):
    print('Loading {} dataset...'.format(dataset))
    ner_file = open("{}{}.final_feature_embeddings_updated_1".format(path, dataset), encoding="utf-8")
    idx_features_labels = read_file(ner_file)
    word_to_ix, tag_to_ix = getVocab(idx_features_labels)
    get_w2v, embed_dim = generate_w2v(word_to_ix)
    return idx_features_labels, word_to_ix, tag_to_ix, get_w2v, embed_dim


def load_edge_embed_data(path="./data/invoice/", dataset="invoice"):
    edge_file = open("{}{}.final_edge_embeddings_updated".format(path, dataset), encoding="utf-8")
    df = pd.read_csv(edge_file)
    return df


def load_coordinates_file(path="./data/invoice/", dataset="invoice"):
    coordinates_file = open("{}{}.final_coordinates_1".format(path, dataset), encoding="utf-8")
    df = pd.read_csv(coordinates_file)
    return df


def get_closest_neighbors_adj(df, start_index):
    # reference : https://github.com/dhavalpotdar/Graph-Convolution-on-Structured-Documents/blob/7400704346cb9698f6c6b1ad3307d2439d17fa33/grapher.py
    distances, nearest_dest_ids_vert = [], []
    x_src_coords_vert, y_src_coords_vert, x_dest_coords_vert, y_dest_coords_vert = [], [], [], []
    lengths, nearest_dest_ids_hori = [], []
    x_src_coords_hori, y_src_coords_hori, x_dest_coords_hori, y_dest_coords_hori = [], [], [], []
    for src_idx, src_row in df.iterrows():
        dest_attr_vert = []
        dest_attr_hori = []
        src_x_min = src_row['xmin1']
        src_x_max = src_row['xmax1']
        src_y_min = src_row['xmin2']
        src_y_max = src_row['ymax2']
        src_range_x = (src_x_min, src_x_max)
        src_center_y = (src_y_min + src_y_max) / 2
        src_range_y = (src_y_min, src_y_max)
        src_center_x = (src_x_min + src_x_max) / 2
        for dest_idx, dest_row in df.iterrows():
            dest_x_min = dest_row['xmin1']
            dest_x_max = dest_row['xmax1']
            dest_y_min = dest_row['xmin2']
            dest_y_max = dest_row['ymax2']
            is_beneath = False
            if not src_idx == dest_idx:
                dest_range_x = (dest_x_min, dest_x_max)
                dest_center_y = (dest_y_min + dest_y_max) / 2
                height = dest_center_y - src_center_y
                if dest_center_y > src_center_y:
                    if dest_range_x[0] <= src_range_x[0] and dest_range_x[1] >= src_range_x[1]:
                        x_common = (src_range_x[0] + src_range_x[1]) / 2
                        line_src = (x_common, src_center_y)
                        line_dest = (x_common, dest_center_y)
                        attributes = (dest_idx, line_src, line_dest, height)
                        dest_attr_vert.append(attributes)
                        is_beneath = True
                    elif dest_range_x[0] >= src_range_x[0] and dest_range_x[1] <= src_range_x[1]:
                        x_common = (dest_range_x[0] + dest_range_x[1]) / 2
                        line_src = (x_common, src_center_y)
                        line_dest = (x_common, dest_center_y)
                        attributes = (dest_idx, line_src, line_dest, height)
                        dest_attr_vert.append(attributes)
                        is_beneath = True
                    elif dest_range_x[0] <= src_range_x[0] and dest_range_x[1] >= src_range_x[0] and dest_range_x[1] < src_range_x[1]:
                        x_common = (src_range_x[0] + dest_range_x[1]) / 2
                        line_src = (x_common, src_center_y)
                        line_dest = (x_common, dest_center_y)
                        attributes = (dest_idx, line_src, line_dest, height)
                        dest_attr_vert.append(attributes)
                        is_beneath = True
                    elif dest_range_x[0] <= src_range_x[1] and dest_range_x[1] >= src_range_x[1] and dest_range_x[0] > src_range_x[0]:
                        x_common = (dest_range_x[0] + src_range_x[1]) / 2
                        line_src = (x_common, src_center_y)
                        line_dest = (x_common, dest_center_y)
                        attributes = (dest_idx, line_src, line_dest, height)
                        dest_attr_vert.append(attributes)
                        is_beneath = True
            if not is_beneath and not src_idx == dest_idx:
                dest_range_y = (dest_y_min, dest_y_max)
                dest_center_x = (dest_x_min + dest_x_max) / 2
                if dest_center_x > src_center_x:
                    length = dest_center_x - src_center_x
                else:
                    length = 0
                if dest_center_x > src_center_x:
                    if dest_range_y[0] >= src_range_y[0] and dest_range_y[1] <= src_range_y[1]:
                        y_common = (dest_range_y[0] + dest_range_y[1]) / 2
                        line_src = (src_center_x, y_common)
                        line_dest = (dest_center_x, y_common)
                        attributes = (dest_idx, line_src, line_dest, length)
                        dest_attr_hori.append(attributes)
                    if dest_range_y[0] <= src_range_y[0] and dest_range_y[1] <= src_range_y[1] and dest_range_y[1] > src_range_y[0]:
                        y_common = (src_range_y[0] + dest_range_y[1]) / 2
                        line_src = (src_center_x, y_common)
                        line_dest = (dest_center_x, y_common)
                        attributes = (dest_idx, line_src, line_dest, length)
                        dest_attr_hori.append(attributes)
                    if dest_range_y[0] >= src_range_y[0] and dest_range_y[1] >= src_range_y[1] and dest_range_y[0] < src_range_y[1]:
                        y_common = (dest_range_y[0] + src_range_y[1]) / 2
                        line_src = (src_center_x, y_common)
                        line_dest = (dest_center_x, y_common)
                        attributes = (dest_idx, line_src, line_dest, length)
                        dest_attr_hori.append(attributes)
                    if dest_range_y[0] <= src_range_y[0] and dest_range_y[1] >= src_range_y[1]:
                        y_common = (src_range_y[0] + src_range_y[1]) / 2
                        line_src = (src_center_x, y_common)
                        line_dest = (dest_center_x, y_common)
                        attributes = (dest_idx, line_src, line_dest, length)
                        dest_attr_hori.append(attributes)
        dest_attr_vert_sorted = sorted(dest_attr_vert, key=lambda x: x[3])
        dest_attr_hori_sorted = sorted(dest_attr_hori, key=lambda x: x[3])
        if len(dest_attr_vert_sorted) == 0:
            nearest_dest_ids_vert.append(-1)
            x_src_coords_vert.append(-1)
            y_src_coords_vert.append(-1)
            x_dest_coords_vert.append(-1)
            y_dest_coords_vert.append(-1)
            distances.append(0)
        else:
            nearest_dest_ids_vert.append(dest_attr_vert_sorted[0][0])
            x_src_coords_vert.append(dest_attr_vert_sorted[0][1][0])
            y_src_coords_vert.append(dest_attr_vert_sorted[0][1][1])
            x_dest_coords_vert.append(dest_attr_vert_sorted[0][2][0])
            y_dest_coords_vert.append(dest_attr_vert_sorted[0][2][1])
            distances.append(dest_attr_vert_sorted[0][3])
        if len(dest_attr_hori_sorted) == 0:
            nearest_dest_ids_hori.append(-1)
            x_src_coords_hori.append(-1)
            y_src_coords_hori.append(-1)
            x_dest_coords_hori.append(-1)
            y_dest_coords_hori.append(-1)
            lengths.append(0)
        else:
            try:
                nearest_dest_ids_hori.append(dest_attr_hori_sorted[0][0])
            except:
                nearest_dest_ids_hori.append(-1)
            try:
                x_src_coords_hori.append(dest_attr_hori_sorted[0][1][0])
            except:
                x_src_coords_hori.append(-1)
            try:
                y_src_coords_hori.append(dest_attr_hori_sorted[0][1][1])
            except:
                y_src_coords_hori.append(-1)
            try:
                x_dest_coords_hori.append(dest_attr_hori_sorted[0][2][0])
            except:
                x_dest_coords_hori.append(-1)
            try:
                y_dest_coords_hori.append(dest_attr_hori_sorted[0][2][1])
            except:
                y_dest_coords_hori.append(-1)
            try:
                lengths.append(dest_attr_hori_sorted[0][3])
            except:
                lengths.append(0)
    adj = np.zeros((df.shape[0], df.shape[0]), dtype=int)
    for i, (vert, hori) in enumerate(zip(nearest_dest_ids_vert, nearest_dest_ids_hori)):
        if vert != -1:
            adj[i, vert - start_index] = 1
        if hori != -1:
            adj[i, hori - start_index] = 1
    return torch.from_numpy(adj)

