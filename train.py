import logging
import os

import math
from scipy import stats
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch import nn, Tensor
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from argparse import ArgumentParser
import csv
import pandas as pd

torch.manual_seed(0)


def padding(text, pad, max_len=50):
    return text if len(text) >= max_len else (text + [pad] * (max_len - len(text)))


def encode_batch(text, berts, max_len=50):
    tokenizer = berts[0]
    t1 = []
    for line in text:
        t1.append(padding(tokenizer.encode(line, add_special_tokens=True, max_length=max_len, truncation=True),
                          tokenizer.pad_token_id, max_len))
    return t1


def data_iterator(train_x, train_y, batch_size=64):
    n_batches = math.ceil(len(train_x) / batch_size)
    for idx in range(n_batches):
        x = train_x[idx * batch_size:(idx + 1) * batch_size]
        y = train_y[idx * batch_size:(idx + 1) * batch_size]
        yield x, y


def get_metrics(model, test_x, test_y, args, tokenizer, test=False, save_path='test_prediction_final.txt'):
    cuda = args.cuda
    all_preds = []
    test_iterator = data_iterator(test_x, test_y, batch_size=64)
    all_y = []
    all_x = []
    model.eval()
    for x, y in test_iterator:
        ids = encode_batch(x, (tokenizer, model), max_len=args.max_len)
        with torch.no_grad():
            if cuda:
                input_ids = Tensor(ids).cuda().long()
                labels = torch.cuda.FloatTensor(y)
            else:
                input_ids = Tensor(ids).long()
                labels = torch.FloatTensor(y)
            outputs = model(input_ids, labels=labels)
            loss, y_pred = outputs[:2]

        predicted = y_pred.cpu().data
        all_preds.extend(predicted.numpy())
        all_y.extend(y)
        all_x.extend(x)

    all_res = np.array(all_preds).flatten()
    if test and save_path:
        with open(save_path, 'w') as w:
            for i in range(len(all_y)):
                if i < 2:
                    print(all_x[i], all_res[i], test_y[i])
                w.writelines(all_x[i] + '\t' + str(all_y[i]) + '\t' + str(all_res[i]) + '\n')

    return loss, stats.pearsonr(all_res, all_y)[0]


def run_epoch(model, train_data, val_data, tokenizer, args, optimizer):
    train_x, train_y = train_data[0], train_data[1]
    val_x, val_y = val_data[0], val_data[1]
    iterator = data_iterator(train_x, train_y, args.batch_size)
    train_losses = []
    val_accuracies = []
    losses = []

    for i, (x, y) in tqdm(enumerate(iterator), total=int(len(train_x) / args.batch_size)):
        # print('iteration', i)
        model.zero_grad()

        ids = encode_batch(x, (tokenizer, model), max_len=args.max_len)

        if args.cuda:
            input_ids = Tensor(ids).cuda().long()
            labels = torch.cuda.FloatTensor(y)
        else:
            input_ids = Tensor(ids).long()
            labels = torch.FloatTensor(y)

        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        loss.backward()
        losses.append(loss.data.cpu().numpy())
        optimizer.step()

        if (i + 1) % 1 == 0:
            avg_train_loss = np.mean(losses)
            train_losses.append(avg_train_loss)
            losses = []

            # Evalute Accuracy on validation set
            model.eval()
            all_preds = []
            val_iterator = data_iterator(val_x, val_y, args.batch_size)
            for x, y in val_iterator:
                ids = encode_batch(x, (tokenizer, model), max_len=args.max_len)

                with torch.no_grad():

                    if args.cuda:
                        input_ids = Tensor(ids).cuda().long()
                        labels = torch.cuda.FloatTensor(y)
                    else:
                        input_ids = Tensor(ids).long()
                        labels = torch.FloatTensor(y)
                    outputs = model(input_ids, labels=labels)
                    loss, y_pred = outputs[:2]

                predicted = y_pred.cpu().data

                all_preds.extend(predicted.numpy())

            all_res = np.array(all_preds).flatten()
            score = (np.square(val_y - all_res)).mean()
            val_accuracies.append(score)
            model.train()

    return train_losses, val_accuracies


def get_test_result(model, test_x, test_y, args, tokenizer, pure_predict=False):
    cuda = args.cuda
    all_raw = []
    all_preds = []
    all_y = []
    all_x = []
    test_iterator = data_iterator(test_x, test_y, batch_size=256)
    model.eval()
    i = 0
    for x, y in test_iterator:
        print(str(i * 256) + '/' + str(len(test_x)))
        i += 1
        ids = encode_batch(x, (tokenizer, model), max_len=args.max_len)

        with torch.no_grad():
            if cuda:
                input_ids = Tensor(ids).cuda().long()
            else:
                input_ids = Tensor(ids).long()
            outputs = model(input_ids)
            y_pred = outputs[0]

        predicted = y_pred.cpu().data
        all_preds.extend(predicted.numpy())
        all_y.extend(y)
        all_x.extend(x)

    all_res = np.array(all_preds).flatten()

    if not pure_predict:
        print('mse:', (np.square(all_y - all_res)).mean())
        print('pearson r:', stats.pearsonr(all_res, all_y)[0])

    return all_res, all_y


def arguments():
    parser = ArgumentParser()
    parser.set_defaults(show_path=False, show_similarity=False)

    parser.add_argument('--mode')
    parser.add_argument('--pre_trained_model_name_or_path')
    parser.add_argument('--train_path', default='train.txt')
    parser.add_argument('--val_path', default='val.txt')
    parser.add_argument('--test_path', default='test.txt')
    parser.add_argument('--log_saving_path', default='log.log')
    parser.add_argument('--predict_data_path')
    parser.add_argument('--model_saving_path', default=None)
    parser.add_argument('--test_saving_path', default=None)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)

    return parser.parse_args()


if __name__ == '__main__':

    args = arguments()


    def get_csv_data(path):
        print('open:', path)
        text = []
        bias_type = []
        y = []
        lines = open(path, 'r', newline='')
        lines_reader = csv.reader(lines)
        for line in lines_reader:
            t = line[0]
            text.append(t)
            if len(line) == 3:
                bt = line[1]
                l = line[2]
                bias_type.append(bt)
                y.append(float(l))
        return text, y


    def get_csv_predict_data(path):
        print('open:', path)
        sentence_list = []
        y_list = []
        lines = open(path, 'r', newline='')
        lines_reader = csv.reader(lines)
        next(lines_reader)
        for i, line in enumerate(lines_reader):
            sentence = line[0]
            sentence_list.append(sentence)
            y_list.append(0.0)
        return sentence_list, y_list


    tokenizer = AutoTokenizer.from_pretrained(args.pre_trained_model_name_or_path, num_labels=1,
                                              output_attentions=False, output_hidden_states=False)

    model = AutoModelForSequenceClassification.from_pretrained(args.pre_trained_model_name_or_path, num_labels=1,
                                                               output_attentions=False, output_hidden_states=False)
    if torch.cuda.is_available():
        args.cuda = True

    if args.cuda:
        model.cuda()
    test_result = []

    if args.mode == 'train':
        log_directory = 'logs'

        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        log_file_path = os.path.join(log_directory, f'{args.log_saving_path}')

        logging.basicConfig(filename=log_file_path, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        train_text, train_labels = get_csv_data(args.train_path)
        val_text, val_labels = get_csv_data(args.val_path)
        test_text, test_labels = get_csv_data(args.test_path)

        train_x = train_text
        train_y = np.array(train_labels)
        val_x = val_text
        val_y = np.array(val_labels)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

        train_data = [train_x, train_y]
        val_data = [val_x, val_y]

        test_x = test_text
        test_y = np.array(test_labels)
        best_val = 100.0
        best_test = 100.0
        best_r = 100

        for i in range(args.max_epochs):
            logging.info(f"Epoch: {i}")

            train_losses, val_accuracies = run_epoch(model, train_data, val_data, tokenizer, args, optimizer)
            test_acc, test_r = get_metrics(model, test_x, test_y, args, tokenizer, test=True,
                                           save_path=args.test_saving_path)

            logging.info(f"Average training loss: {np.mean(train_losses)}")
            logging.info(f"Average Val MSE: {np.mean(val_accuracies)}")

            if np.mean(val_accuracies) < best_val:
                best_val = np.mean(val_accuracies)
                best_test = test_acc
                best_r = test_r
                if i >= 1 and args.model_saving_path:
                    model.save_pretrained(f"{args.model_saving_path}/{args.pre_trained_model_name_or_path}")
                    tokenizer.save_pretrained(f"{args.model_saving_path}/{args.pre_trained_model_name_or_path}")

        logging.info(f"model saved at {args.model_saving_path}/{args.pre_trained_model_name_or_path}")
        logging.info(f"best_val_loss: {best_val}")
        logging.info(f"best_test_loss: {best_test}")
        logging.info(f"best_test_pearsonr: {best_r}")
    elif args.mode == 'predict':
        final_test_text, final_test_y = get_csv_predict_data(args.predict_data_path)
        test_result, test_score = get_test_result(model, final_test_text, final_test_y, args, tokenizer,
                                                  pure_predict=True)

        df = pd.read_csv(args.predict_data_path)
        df['score'] = test_result
        df.to_csv(args.test_saving_path, index=False)



