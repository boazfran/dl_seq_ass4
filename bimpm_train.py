
import pandas as pd
import numpy as np
from torch.autograd import Variable
from torchtext.data.utils import get_tokenizer
from torch.utils.data import TensorDataset
from torch.nn.functional import cosine_similarity
from torch import LongTensor
import string
import csv
import torch
from torch import nn, optim
import random
from IPython.display import clear_output
import argparse
import time
from enum import Enum
import os
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))

label2index = pd.Series([0, 1, 2], index=['neutral', 'entailment', 'contradiction'])

def get_chars_vectors(sentences, char2index, is_train_set):

    # to be used only if is_train_set is False
    oov_index = len(char2index) + 1  # zero is the padding index

    sentences_chars = []
    for sentence in sentences:
        sentence_chars = []
        for word in sentence:
            word_chars = []
            for char in word:
                index = char2index.get(char)
                if char is not None:
                    word_chars.append(index)
                elif is_train_set:
                    char2index[char] = len(char2index) + 1  # zero is the padding index
                    word_chars.append(char2index[char])
                else:
                    word_chars.append(oov_index)

            sentence_chars.append(word_chars)

        sentences_chars.append(sentence_chars)

    return sentences_chars


def get_words_vectors(sentences, word2index, ukword2index, is_train_set):

    # to be used only if is_train_set is False
    oov_index = len(word2index) + len(ukword2index)

    sentences_words = []
    for i, sentence in enumerate(sentences):
        sentence_words = []
        for word in sentence:
            index = word2index.get(word)
            if index is not None:
                sentence_words.append(index)
                continue
            index = ukword2index.get(word)
            if index is not None:
                sentence_words.append(index)
                continue
            elif is_train_set: # training set
                ukword2index[word] = len(word2index) + len(ukword2index)
                sentence_words.append(ukword2index[word])
            else:
                sentence_words.append(oov_index)

        sentences_words.append(sentence_words)

    return sentences_words


def read_data(data, char2index, word2index, ukword2index, is_train_set):

    data = data[data.sentence1.notna()]
    data = data[data.sentence2.notna()]
    data = data[data.gold_label != '-']
    tokenizer = get_tokenizer("basic_english")
    data['sentence1_words'] = list(map(lambda x: tokenizer(x), data.sentence1))
    data['sentence2_words'] = list(map(lambda x: tokenizer(x), data.sentence2))
    sentence1_chars_indices = get_chars_vectors(data['sentence1_words'], char2index, is_train_set)
    sentence2_chars_indices = get_chars_vectors(data['sentence2_words'], char2index, is_train_set)
    sentence1_word_indices = get_words_vectors(data['sentence1_words'], word2index, ukword2index, is_train_set)
    sentence2_word_indices = get_words_vectors(data['sentence2_words'], word2index, ukword2index, is_train_set)
    labels = data.gold_label.apply(lambda x: label2index[x]).to_numpy()

    sentence1_chars_indices = np.array(sentence1_chars_indices, dtype=object)
    sentence2_chars_indices = np.array(sentence2_chars_indices, dtype=object)
    sentence1_word_indices = np.array(sentence1_word_indices, dtype=object)
    sentence2_word_indices = np.array(sentence2_word_indices, dtype=object)

    return sentence1_chars_indices, sentence2_chars_indices, sentence1_word_indices, sentence2_word_indices, labels


def create_batch(data, permute, start, end, word_pad_idx):
    sentence1_chars_indices = data[0][permute[start:end]]
    sentence2_chars_indices = data[1][permute[start:end]]
    sentence1_word_indices = data[2][permute[start:end]]
    sentence2_word_indices = data[3][permute[start:end]]
    labels = LongTensor(data[4][permute[start:end]])

    max_sentence1_word_len = max(list(map(lambda x: max(list(map(lambda y: len(y), x))), sentence1_chars_indices)))
    max_sentence1_len = max(list(map(lambda x: len(x), sentence1_chars_indices)))
    sentence1_char_input = Variable(torch.zeros((len(sentence1_chars_indices), 
                                                 max_sentence1_len,
                                                 max_sentence1_word_len))).long()
    for i, senetence_char in enumerate(sentence1_chars_indices):
        for j, word_chars in enumerate(senetence_char):
            sentence1_char_input[i, j, :len(word_chars)] = LongTensor(word_chars)

    max_sentence2_word_len = max(list(map(lambda x: max(list(map(lambda y: len(y), x))), sentence2_chars_indices)))
    max_sentence2_len = max(list(map(lambda x: len(x), sentence2_chars_indices)))
    sentence2_char_input = Variable(torch.zeros((len(sentence2_chars_indices), 
                                                 max_sentence2_len,
                                                 max_sentence2_word_len))).long()
    for i, senetence_char in enumerate(sentence2_chars_indices):
        for j, word_chars in enumerate(senetence_char):
            sentence2_char_input[i, j, :len(word_chars)] = LongTensor(word_chars)

    sentence1_word_input = Variable(torch.ones((len(sentence1_word_indices),
                                                 max_sentence1_len))).long() * word_pad_idx
    for i, senetence_words in enumerate(sentence1_word_indices):
        sentence1_word_input[i, :len(senetence_words)] = LongTensor(senetence_words)

    sentence2_word_input = Variable(torch.ones((len(sentence2_word_indices),
                                                 max_sentence2_len))).long() * word_pad_idx
    for i, senetence_words in enumerate(sentence2_word_indices):
        sentence2_word_input[i, :len(senetence_words)] = LongTensor(senetence_words)

    output_labels = LongTensor(labels)

    return TensorDataset(sentence1_char_input, sentence1_word_input, sentence2_char_input, sentence2_word_input, output_labels)


class Strategy(Enum):
    full = 1
    max_pool = 2
    attentive = 3
    max_attentive =4


class BiMPM_NN(nn.Module):
    def __init__(self, word2index, ukword2index, char2index, prespective_dim, dropout, strategies):

        def create_init_weight_matrix():
            tmp = torch.zeros((prespective_dim, 100))
            tmp = nn.init.xavier_uniform_(tmp)
            return nn.parameter.Parameter(tmp)

        super(BiMPM_NN, self).__init__()

        self.prespective_dim = prespective_dim
        self.strategies = strategies

        # drop out between the layers
        self.dropout = nn.Dropout(p=dropout)

        # embedding layer
        self.char_embedding_layer = nn.Embedding(len(char2index)+1, 20, padding_idx=0)
        self.char_lstm_layer = nn.LSTM(20, 50, bidirectional=False, batch_first=True, num_layers=1)
        self.oov_word_embedding_layer = nn.Embedding(len(ukword2index)+2, 300, padding_idx=len(ukword2index) + 1)
        self.word_embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(words.to_numpy()))

        # context layer
        self.lstm_context_layer = nn.LSTM(350, 100, bidirectional=True, batch_first=True, num_layers=1)
        
        # matching strategies weights - 2 (one for each direction - backward and forward) 
        # for each one of the 4 strategies
        if Strategy.full.value in strategies:
            self.full_matching_fw_layer = create_init_weight_matrix()
            self.full_matching_bw_layer = create_init_weight_matrix()
        if Strategy.max_pool.value in strategies:
            self.max_pool_matching_fw_layer = create_init_weight_matrix()
            self.max_pool_matching_bw_layer = create_init_weight_matrix()
        if Strategy.attentive.value in strategies:
            self.atten_matching_fw_layer = create_init_weight_matrix()
            self.atten_matching_bw_layer = create_init_weight_matrix()
        if Strategy.max_attentive.value in strategies:
            self.max_atten_matching_fw_layer = create_init_weight_matrix()
            self.max_atten_matching_bw_layer = create_init_weight_matrix()

        # aggregation layer
        self.lstm_aggregation_layer = nn.LSTM(prespective_dim*2*len(strategies),
                                              100, 
                                              bidirectional=True, 
                                              batch_first=True, 
                                              num_layers=1)

        # prediction layer
        self.prediction_layer = nn.Linear(400, 3)
        self.activation = nn.Tanh()

        # to be able to load the model and use this fields for reading test data
        self.word2index = word2index
        self.char2index = char2index
        self.ukword2index = ukword2index


    def init_hidden(self, hidden_dim, batch_size, lstm_num_layers, is_bidirectional):
        if is_bidirectional:
            return (Variable(torch.zeros(lstm_num_layers*2, batch_size,
                                  hidden_dim)).to(device),
                    Variable(torch.zeros(lstm_num_layers*2, batch_size,
                                  hidden_dim)).to(device))
        else:
            return (Variable(torch.zeros(lstm_num_layers, batch_size,
                                  hidden_dim)).to(device),
                    Variable(torch.zeros(lstm_num_layers, batch_size,
                                  hidden_dim)).to(device))
        
    def forward(self, sentence1_char_input, sentence1_word_input, sentence2_char_input, sentence2_word_input):


        def pairwise_cosine_similarity(h1, h2):
            '''
            h1(batch_size, h1_num_hidden_states, hidden_state_dim)
            h2(batch_size, h2_num_hidden_states, hidden_state_dim)
            '''
            # compare cosine similarity between each hidden state in h1 to each hidden state in h2
            # m2
            l = []
            for i in range(h1.shape[1]):
                l.extend([h1[:, i]] * h2.shape[1])
            m1 = torch.stack(l, dim=1)
            m1 = m1.view(h1.shape[0], h1.shape[1], h2.shape[1], h1.shape[2])
            m2 = h2.view(h1.shape[0], 1, m1.shape[2], m1.shape[3])

            m = cosine_similarity(m1, m2, dim=3)
            del m1, m2

            return m

        def full_matching(h1, h2, weights):
            '''
            h1(batch_size, h1_num_hidden_states, hidden_state_dim)
            h2(batch_size, hidden_state_dim)
            weights(perspecitve_dim, hidden_state_dim)
            '''
            # match all of h1 time steps against h2 - h2 which is passed to 
            # to this function is a single timestamp (the last one)
            m1 = torch.stack([h1] * self.prespective_dim, dim=2) * weights
            m2 = torch.stack([h2] * self.prespective_dim, dim=1) * weights
            m2 = m2.view(m2.shape[0], 1, m2.shape[1], m2.shape[2])

            m = cosine_similarity(m1, m2, dim=3)
            del m1, m2

            return m

        def max_pool_matching(h1, h2, weights):
            '''
            h1(batch_size, h1_num_hidden_states, hidden_state_dim)
            h2(batch_size, h2_num_hidden_states, hidden_state_dim)
            weights(perspecitve_dim, hidden_state_dim)
            '''
            # match each timestamp in h1 against each timestamp in h2. For this
            # we will create a matrix where each timestamp in h1 repeats itself
            # as the number of time steps in h2. We then can use this matrix wit
            # the cosine_simalarity function to calculate the pairwise distance
            # after computing the pairwise distance we take the maximum over 
            # each dimension where dimension i is the consince distnaces between
            # h1 i-th timestamp and all h2 time steps
            m1 = torch.stack([h1] * self.prespective_dim, dim=2)*weights
            m2 = torch.stack([h2] * self.prespective_dim, dim=2)*weights
            l = []
            for i in range(m1.shape[1]):
                l.extend([m1[:, i]] * m2.shape[1])
            m1 = torch.stack(l, dim=1)
            m1 = m1.view(m1.shape[0], h1.shape[1], h2.shape[1], m1.shape[2], m1.shape[3])
            m2 = m2.view(m1.shape[0], 1, m1.shape[2], m1.shape[3], m1.shape[4])

            m = cosine_similarity(m1, m2, dim=4)
            del m1, m2
            m = torch.max(m, dim=2)[0]
            
            return m

        def attentive_matching(h1, h2, pairwise, weights):
            '''
            h1(batch_size, h1_num_hidden_states, hidden_state_dim)
            h2(batch_size, h2_num_hidden_states, hidden_state_dim)
            pairwise(batch_size, h1_num_hidden_states, h2_num_hidden_states, cosine_distance_between_h1_h2_hidden_states)
            weights(perspecitve_dim, hidden_state_dim)
            '''
            # create a weighted average attention vector from the pairwise distances of the hidden states vectors
            numerator = torch.bmm(pairwise, h2)
            denominator = torch.sum(pairwise, axis=2)
            denominator = torch.stack([denominator]*numerator.shape[2], dim=2)
            atten = torch.div(numerator, denominator)
            m1 = torch.stack([h1] * self.prespective_dim, dim=2) * weights
            m2 = torch.stack([atten] * self.prespective_dim, dim=2) * weights

            m = cosine_similarity(m1, m2, dim=3)
            del m1, m2

            return m

        def max_attentive_matching(h1, h2 ,pairwise, weights):
            '''
            h1(batch_size, h1_num_hidden_states, hidden_state_dim)
            h2(batch_size, h2_num_hidden_states, hidden_state_dim)
            pairwise(batch_size, h1_num_hidden_states, h2_num_hidden_states, cosine_distance_between_h1_h2_hidden_states)
            weights(perspecitve_dim, hidden_state_dim)
            '''
            # create a weighted average attention vector from the max of the hidden states vectors
            m2 = torch.max(pairwise, dim=2)[0]
            m2 = m2.view(m2.shape[0], m2.shape[1], 1)

            m1 = torch.stack([h1] * self.prespective_dim, dim=2) * weights
            m2 = torch.stack([m2] * self.prespective_dim, dim=2) * weights

            m = cosine_similarity(m1, m2, dim=3)
            del m1, m2

            return m


        def sentence_char_embedding(sentence_char_input):
            # Extract 50dim word embedding from the chars using trainable char 
            # embedding and 1 layer unidirectional LSTM
            sentence_char_input = self.char_embedding_layer(sentence_char_input)
            sentence_char_input_shape = sentence_char_input.shape # remember original shape
            sentence_char_input = sentence_char_input.view(sentence_char_input.shape[0]*sentence_char_input.shape[1], 
                                                                sentence_char_input.shape[2], 
                                                                sentence_char_input.shape[3]) # line up all the words (flatten the sentences)
            _, sentence_char_input = self.char_lstm_layer(sentence_char_input, self.init_hidden(50, sentence_char_input.shape[0], 1, False))[-1]
            sentence_char_input = sentence_char_input.view(sentence_char_input_shape[0], 
                                                           sentence_char_input_shape[1], 
                                                           50) # return to sentences shape
            return sentence_char_input

        def sentence_word_embedding(sentence_word_input):
            # Extract 300dim word embedding using the external embedding and 
            # trainable oov word embedding
            oov = sentence_word_input >= len(self.word2index)
            if oov.sum() == 0:
                # easy, no oov words in this batch
                return self.word_embedding_layer(sentence_word_input)
            # words which has pretrained embedding
            sentence_word_input_copy = torch.clone(sentence_word_input)
            sentence_word_input[oov] = 0
            embedding = self.word_embedding_layer(sentence_word_input)
            # words which doesn't have a pretrained embedding
            sentence_word_input = sentence_word_input_copy
            del sentence_word_input_copy
            sentence_word_input -= len(self.word2index)
            sentence_word_input[~oov] = len(self.ukword2index) + 1 # padding
            embedding[oov] = self.oov_word_embedding_layer(sentence_word_input)[oov]
            return embedding

        ###########################              
        ##### Embedding layer #####
        ###########################

        sentence1_char_input = sentence_char_embedding(sentence1_char_input)
        sentence2_char_input = sentence_char_embedding(sentence2_char_input)

        sentence1_word_input = sentence_word_embedding(sentence1_word_input)
        sentence2_word_input = sentence_word_embedding(sentence2_word_input)

        sen1_word_and_char_embed = torch.cat((sentence1_word_input, sentence1_char_input), dim=2)
        del sentence1_word_input, sentence1_char_input
        sen2_word_and_char_embed = torch.cat((sentence2_word_input, sentence2_char_input), dim=2)
        del sentence2_word_input, sentence2_char_input

        # drop out
        sen1_word_and_char_embed = self.dropout(sen1_word_and_char_embed)
        sen2_word_and_char_embed = self.dropout(sen2_word_and_char_embed)

        ###########################
        ###### context layer ######
        ###########################
        sen1_output, _ = self.lstm_context_layer(sen1_word_and_char_embed, 
                                                 self.init_hidden(100, sen1_word_and_char_embed.shape[0], 1, True))
        sen2_output, _ = self.lstm_context_layer(sen2_word_and_char_embed,
                                                 self.init_hidden(100, sen2_word_and_char_embed.shape[0], 1, True))

        sen1_fw, sen1_bw = torch.split(sen1_output, 100, -1)
        sen2_fw, sen2_bw = torch.split(sen2_output, 100, -1)

        # drop out
        sen1_fw = self.dropout(sen1_fw)
        sen1_bw = self.dropout(sen1_bw)
        sen2_fw = self.dropout(sen2_fw)
        sen2_bw = self.dropout(sen2_bw)

        ##########################
        ##### matching layer #####
        ##########################
        # accumlate s1 matches and s2 matches from all strategies and directions 
        # in tuples to concatenate them later before inserting to the aggregation 
        # LSTM 
        s1_matches = ()
        s2_matches = ()

        ### Full Match ###
        if Strategy.full.value in self.strategies:
            # Forward
            # P -> Q
            s1_matches += (full_matching(sen1_fw, sen2_fw[:, -1, :], self.full_matching_fw_layer),)
            # Q -> P
            s2_matches += (full_matching(sen2_fw, sen1_fw[:, -1, :], self.full_matching_fw_layer),)

            # Backward
            # P -> Q
            s1_matches += (full_matching(sen1_bw, sen2_bw[:, 0, :], self.full_matching_bw_layer),)
            # Q -> P
            s2_matches += (full_matching(sen2_bw, sen1_bw[:, 0, :], self.full_matching_bw_layer),)

        ### Max Pool Match ###
        if Strategy.max_pool.value in self.strategies:
            # Forward
            # P -> Q
            s1_matches += (max_pool_matching(sen1_fw, sen2_fw, self.max_pool_matching_fw_layer),)
            # Q -> P
            s2_matches += (max_pool_matching(sen2_fw, sen1_fw, self.max_pool_matching_fw_layer),)

            # Backward
            # P -> Q
            s1_matches += (max_pool_matching(sen1_bw, sen2_bw, self.max_pool_matching_bw_layer),)
            # Q -> P
            s2_matches += (max_pool_matching(sen2_bw, sen1_bw, self.max_pool_matching_bw_layer),)

        ### Attentive Match ###
        if Strategy.attentive.value in self.strategies:
            # Forward
            fw_pairwise = pairwise_cosine_similarity(sen1_fw, sen2_fw)
            # P -> Q
            s1_matches += (attentive_matching(sen1_fw, sen2_fw, fw_pairwise, self.atten_matching_fw_layer),)
            # Q -> P
            s2_matches += (attentive_matching(sen2_fw, sen1_fw, fw_pairwise.transpose(2, 1), self.atten_matching_fw_layer),)

            # Backward
            bw_pairwise = pairwise_cosine_similarity(sen1_bw, sen2_bw)
            # P -> Q
            s1_matches += (attentive_matching(sen1_bw, sen2_bw, bw_pairwise, self.atten_matching_bw_layer),)
            # Q -> P
            s2_matches += (attentive_matching(sen2_bw, sen1_bw, bw_pairwise.transpose(2, 1), self.atten_matching_bw_layer),)

        ### Max Attentive Match ###
        if Strategy.max_attentive.value in self.strategies:
            # Forward
            # P -> Q
            s1_matches += (max_attentive_matching(sen1_fw, sen2_fw, fw_pairwise, self.max_atten_matching_fw_layer),)
            # Q -> P
            s2_matches += (max_attentive_matching(sen2_fw, sen1_fw, fw_pairwise.transpose(2, 1), self.max_atten_matching_fw_layer),)

            # Backward
            # P -> Q
            s1_matches += (max_attentive_matching(sen1_bw, sen2_bw, bw_pairwise, self.max_atten_matching_bw_layer),)
            # Q -> P
            s2_matches += (max_attentive_matching(sen2_bw, sen1_bw, bw_pairwise.transpose(2, 1), self.max_atten_matching_bw_layer),)

        s1_m = torch.cat(s1_matches, dim=2)
        del s1_matches
        s2_m = torch.cat(s2_matches, dim=2)
        del s2_matches

        # drop out
        s1_m = self.dropout(s1_m)
        s2_m = self.dropout(s2_m)

        #############################
        ##### aggregation layer #####
        #############################
        _, s1_m_hidden = self.lstm_aggregation_layer(s2_m, self.init_hidden(100, s1_m.shape[0], 1, True))
        s1_m_hidden = s1_m_hidden[0]

        _, s2_m_hidden = self.lstm_aggregation_layer(s1_m, self.init_hidden(100, s2_m.shape[0], 1, True))
        s2_m_hidden = s2_m_hidden[0]

        final = torch.cat((s1_m_hidden[0], s1_m_hidden[1], s2_m_hidden[0], s2_m_hidden[1]), dim=1)
        del s1_m, s2_m, s1_m_hidden, s2_m_hidden

        # drop out
        final = self.dropout(final)

        #########################
        ##### predict layer #####
        #########################
        output = self.activation(self.prediction_layer(final))
        return output


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


def train_model(model, train_data, dev_data, results, run_id, learning_rate, batch_size, n_epochs, model_file, results_file):
    permute = np.array(range(len(train_data[0])))
    model.apply(initialize_weights)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-08)
    criterion = nn.CrossEntropyLoss()

    word_pad_idx = len(model.word2index) + len(model.ukword2index) + 1

    for epoch in range(n_epochs):
        model.train()
        print(f'epoch number: {epoch+1}')
        result_id = len(results)
        results.loc[result_id, 'run_id'] = run_id
        results.loc[result_id, 'learning_rate'] = learning_rate
        results.loc[result_id, 'batch_size'] = batch_size
        results.loc[result_id, 'n_perspective'] = model.prespective_dim
        results.loc[result_id, 'epoch'] = epoch
        results.loc[result_id, 'dropout'] = dropout
        results.loc[result_id, 'strategies'] = format(model.strategies)

        start = time.time()
        running_loss = 0.0
        correct = 0
        random.shuffle(permute) # shuffle data every epoch
        for i in range(len(train_data[0])//batch_size):
            batch = create_batch(train_data, permute, i*batch_size, (i+1)*batch_size, word_pad_idx)
            chars_sen1 = batch[:][0].to(device)
            words_sen1 = batch[:][1].to(device)
            chars_sen2 = batch[:][2].to(device)
            words_sen2 = batch[:][3].to(device)
            batch_labels = batch[:][4].to(device)
            optimizer.zero_grad()
            output = model(chars_sen1,words_sen1, chars_sen2,  words_sen2)
            del chars_sen1, words_sen1, chars_sen2, words_sen2
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            outputs_max_inds = torch.argmax(output, axis=1)
            correct += torch.sum(outputs_max_inds == batch_labels)
            del batch_labels
            acc_batch = 100*correct/(i*batch_size)
            if (i+1)%1000==0:
                clear_output(wait=True)
            if (i+1)%100==0:
                print(f"Train batch accuracy: {acc_batch:.20f}%")
        accuracy = 100 * correct/(i*batch_size)
        avg_loss = running_loss/(i*batch_size)
        print(f"Train loss: {avg_loss:.3f}, Train accuracy: {accuracy:.3f}%")
        results.loc[result_id, 'train_acc'] = accuracy.cpu().float()
        results.loc[result_id, 'train_loss'] = avg_loss

        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            correct = 0
            dev_indices = np.array(range(len(dev_data[0])))
            for i in range(len(dev_data[0])//batch_size):
                batch = create_batch(dev_data, dev_indices, i*batch_size, (i+1)*batch_size, word_pad_idx)
                chars_sen1 = batch[:][0].to(device)
                words_sen1 = batch[:][1].to(device)
                chars_sen2 = batch[:][2].to(device)
                words_sen2 = batch[:][3].to(device)
                batch_labels = batch[:][4].to(device)
                output = model(chars_sen1,words_sen1, chars_sen2,  words_sen2)
                del chars_sen1, words_sen1, chars_sen2, words_sen2
                loss = criterion(output, batch_labels)
                running_loss += loss.item()
                outputs_max_inds = torch.argmax(output, axis=1)
                correct += torch.sum(outputs_max_inds == batch_labels)
                del batch_labels
            if len(dev_data[0]) % batch_size != 0:
                batch = create_batch(dev_data, dev_indices, i*batch_size, len(dev_data[0]), word_pad_idx)
                chars_sen1 = batch[:][0].to(device)
                words_sen1 = batch[:][1].to(device)
                chars_sen2 = batch[:][2].to(device)
                words_sen2 = batch[:][3].to(device)
                batch_labels = batch[:][4].to(device)
                output = model(chars_sen1,words_sen1, chars_sen2,  words_sen2)
                del chars_sen1, words_sen1, chars_sen2, words_sen2
                loss = criterion(output, batch_labels)
                running_loss += loss.item()
                outputs_max_inds = torch.argmax(output, axis=1)
                correct += torch.sum(outputs_max_inds == batch_labels)
                del batch_labels
            accuracy = 100 * correct/len(dev_data[0])
            avg_loss = running_loss/len(dev_data[0])
            print(f"Dev loss: {avg_loss:.3f}, Dev accuracy: {accuracy:.3f}%")
            results.loc[result_id, 'dev_acc'] = accuracy.cpu().float()
            results.loc[result_id, 'dev_loss'] = avg_loss
            results.loc[result_id, 'duration_sec'] = time.time() - start

        torch.save(model, model_file + '.model')
        results.to_csv(results_file, index_label='run_id')

    return model.cpu()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BiMPM pytorch implementation')
    parser.add_argument('--output_dir', default='./', help='output dir to save the model')
    parser.add_argument('--model', help='Continue training an existing model - provide path. Default is None')
    parser.add_argument('--word_embedding_file', default='glove.6B.300d.txt',
                        help='Path to the pretrained word embedding file. Default is "glove.6B.300d.txt".')
    parser.add_argument('--snli_folder', default='./',
                        help='Folder with the SNLI files. Default is "./')
    parser.add_argument('--results_file', default='results.csv',
                        help='Output file to save results. Default is "results.csv"')
    parser.add_argument('--n_epochs', default=3, type=int,
                        help='Number of train epochs. Default is 3.')
    parser.add_argument('--learning_rate', default=[0.001], type=float, nargs='+',
                        help='Space separated list of learning rates. Default is 0.001')
    parser.add_argument('--dropout', default=[0.1], type=float, nargs='+',
                        help='Space separated list of dropouts. Default is 0.1.')
    parser.add_argument('--batch_size', default=[64], type=int, nargs='+',
                        help='Space separated list of batch sizes. Default is 64.')
    parser.add_argument('--n_perspective', type=int, nargs='+',
                        help='Space separated list of perspective sizes. Default is 1.')
    parser.add_argument('--strategies',
                        type=lambda x: Strategy.__members__.get(x).value, nargs='+',
                        help='Space separated list of matching strategies - all strategies in the list will be '
                             '*applied together* (and not in different runs) in the model. Options are: '
                             'full/max_pool/attentive/max_attentive. Default is all strategies.')
    args = parser.parse_args()

    if args.model is not None:
        if args.n_perspective is not None:
            print('Cannot change perspective of an existing model')
            exit(1)
        elif args.strategies is not None:
            print('Cannot change strategies of an existing model')
            exit(1)
    else:
        if args.n_perspective is None:
            args.n_perspective = [1]
        if args.strategies is None:
            args.strategies = [Strategy.full.value, Strategy.max_pool.value, Strategy.attentive.value,
                               Strategy.max_attentive.value]

    results = pd.DataFrame(columns=['run_id', 'epoch', 'train_acc', 'train_loss',  'dev_acc', 'dev_loss', 'duration_sec',
                                    'learning_rate', 'batch_size', 'n_perspective', 'dropout', 'strategies'])

    # read pretrained word embedding
    glove_data_file = open(args.word_embedding_file, mode='r')
    words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    glove_data_file.close()
    word2index = pd.Series(range(0, len(words)), index=words.index).to_dict()
    ukword2index = dict()

    # create char mapping
    chars = list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation)
    char2index = pd.Series(range(1, len(chars) + 1), index=chars).to_dict()# start from 1 because zero is for padding

    train = pd.read_csv(os.path.join(args.snli_folder, 'snli_1.0_train.txt'), sep='\t')
    train_data = read_data(train, char2index, word2index, ukword2index, True)

    dev = pd.read_csv(os.path.join(args.snli_folder, 'snli_1.0_dev.txt'), sep='\t')
    dev_data = read_data(dev, char2index, word2index, ukword2index, False)

    ts_str = time.strftime("%Y%m%d_%H%M%S")

    results_file = os.path.join(os.path.dirname(args.results_file), ts_str + '_' + os.path.basename(args.results_file))

    run_id = 0
    for learning_rate in args.learning_rate:
        for batch_size in args.batch_size:
            for dropout in args.dropout:
                model_file = os.path.join(args.output_dir, ts_str + '_run_id' + str(run_id) + '.model')
                if args.model is not None:
                    model = torch.load(args.model)
                    model.dropout = nn.Dropout(p=dropout)
                    model = train_model(model, train_data, dev_data, results, run_id, learning_rate, batch_size,
                                        args.n_epochs, model_file, results_file)
                    torch.save(model, model_file + '.model')
                    run_id += 1
                else:
                    for n_perspective in args.n_perspective:
                        model = BiMPM_NN(word2index, ukword2index, char2index, n_perspective, dropout, args.strategies)
                        model = train_model(model, train_data, dev_data, results, run_id, learning_rate, batch_size,
                                            args.n_epochs, model_file, results_file)
                        torch.save(model, model_file + '.model')
                        run_id += 1

    results.to_csv(results_file, index_label='run_id')
    print('results are saved to ', args.results_file)
