import torch
import argparse
import pandas as pd
from bimpm_train import *
import numpy as np
from torch import nn


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='test BiMPM model')
    parser.add_argument('model', help='Path to the model file.')
    parser.add_argument('test_file', help='path to the test file.')
    args = parser.parse_args()

    model = torch.load(args.model)

    test = pd.read_csv(args.test_file, sep='\t')
    test_data = read_data(test, model.char2index, model.word2index, model.ukword2index, False)
    criterion = nn.CrossEntropyLoss()

    word_pad_idx = len(model.word2index) + len(model.ukword2index) + 1

    model.word2index = None
    model.char2index = None
    model.ukword2index = None

    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        correct = 0
        dev_indices = np.array(range(len(test_data[0])))
        for i in range(len(test_data[0])):
            batch = create_batch(test_data, dev_indices, i, i + 1, word_pad_idx)
            chars_sen1 = batch[:][0].to(device)
            words_sen1 = batch[:][1].to(device)
            chars_sen2 = batch[:][2].to(device)
            words_sen2 = batch[:][3].to(device)
            batch_labels = batch[:][4].to(device)
            output = model(chars_sen1, words_sen1, chars_sen2, words_sen2)
            del chars_sen1, words_sen1, chars_sen2, words_sen2
            loss = criterion(output, batch_labels)
            running_loss += loss.item()
            outputs_max_inds = torch.argmax(output, axis=1)
            correct += torch.sum(outputs_max_inds == batch_labels)
            del batch_labels

        accuracy = 100 * correct / len(test_data[0])
        avg_loss = running_loss / len(test_data[0])
        print(f"Test loss: {avg_loss:.3f}, Test accuracy: {accuracy:.3f}%")