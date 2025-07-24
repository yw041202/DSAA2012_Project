import argparse
import copy
import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import get_cosine_schedule_with_warmup

from dataset import RedditDataset
from loss import loss_function
from model import RedditModel
from utils import gr_metrics, make_31, pad_collate_reddit

np.set_printoptions(precision=5)
os.environ['HF_HOME'] = 'D:\\transformers_cache'

def train_loop(model, expt_type, dataloader, optimizer, device, dataset_len, loss_type, scale=1):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for bi, inputs in enumerate(dataloader):
        optimizer.zero_grad()

        labels, tweet_features, lens = inputs

        labels = labels.to(device)
        tweet_features = tweet_features.to(device)

        output = model(tweet_features, lens, labels)

        _, preds = torch.max(output, 1)

        loss = loss_function(output, labels, loss_type, expt_type, scale)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / dataset_len

    return epoch_loss, epoch_acc


def eval_loop(model, expt_type, dataloader, device, dataset_len, loss_type, scale=1):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    fin_targets = []
    fin_outputs = []

    fin_conf = []

    for bi, inputs in enumerate(dataloader): # based on batch
        labels, tweet_features, lens = inputs

        labels = labels.to(device)
        tweet_features = tweet_features.to(device)

        with torch.no_grad():
            output = model(tweet_features, lens, labels)

        _, preds = torch.max(output, 1)

        loss = loss_function(output, labels, loss_type, expt_type, scale=scale)

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        fin_conf.append(output.cpu().detach().numpy())

        fin_targets.append(labels.cpu().detach().numpy())
        fin_outputs.append(preds.cpu().detach().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_corrects.double() / dataset_len

    return epoch_loss, epoch_accuracy, np.hstack(fin_outputs), np.hstack(fin_targets), fin_conf


def main(config):
    # 解析命令行传入的参数
    expt_type = config['expt_type']

    batch_size = config['batch_size']

    epochs = config['epochs']

    hidden_dim = config['hidden_dim']

    embedding_dim = config['embed_dim']

    num_layers = config['num_layers']

    dropout = config['dropout']

    dist_values = [15, 21, 42, 13, 9]

    learning_rate = config['learning_rate']

    scale = config['scale']

    loss_type = "OE"

    model_type = config['model_type']

    number_of_runs = config['num_runs']

    metrics_dict = {}

    data_dir = config['data_dir']

    # get dir path
    model_save_dir = os.path.join(os.path.dirname(__file__), 'model_save')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    for i in trange(number_of_runs):
        print(f'\nRun {i + 1} of {number_of_runs} times')
        
        data_name = os.path.join(data_dir, f'reddit_data_processed_without_frames.pkl')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open(data_name, 'rb') as f:
            df = pickle.load(f)

            if expt_type == 4: 
                df['label'] = df['label'].apply(make_31) 

        df_train, df_test, _, __ = train_test_split(
            df, df['label'].tolist(), test_size=0.2, stratify=df['label'].tolist())

        train_dataset = RedditDataset(
            df_train.label.values, df_train.enc.values)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=pad_collate_reddit, shuffle=True)
       

        test_dataset = RedditDataset(df_test.label.values, df_test.enc.values)
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, collate_fn=pad_collate_reddit)

        if config['model_type'] == 'lstm+att':
            model = RedditModel(expt_type, embedding_dim,
                                hidden_dim, num_layers, dropout)
        # elif config['model_type'] == 'lstm':
        #     model = RedditNoAttModel(
        #         expt_type, embedding_dim, hidden_dim, num_layers, dropout)
        # elif config['model_type'] == 'avg-pool':
        #     model = BertPoolRedditModel(expt_type, embedding_dim, dropout)

        model.to(device)

        optimizer = AdamW(model.parameters(),
                         lr=learning_rate)

        scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=epochs)

        early_stop_counter = 0
        early_stop_limit = config['early_stop']

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = np.inf

        for epoch in trange(epochs, leave=False):
            loss, accuracy = train_loop(model,
                                        expt_type,
                                        train_dataloader,
                                        optimizer,
                                        device,
                                        len(train_dataset),
                                        loss_type,
                                        scale)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            if scheduler is not None:
                scheduler.step() # 更新学习率

            if loss >= best_loss: # 当前损失没有改进
                early_stop_counter += 1
            else:
                best_model_wts = copy.deepcopy(model.state_dict()) # 保存当前最佳模型到参数中，后续可加载
                early_stop_counter = 0 # 重置计数器
                best_loss = loss

            if early_stop_counter == early_stop_limit:
                print(f"Early stopping at epoch {epoch + 1} with loss {best_loss:.4f}")
                break

        # tc = time.time()
        model.load_state_dict(best_model_wts)

        # save the best model weights
        model_save_path = os.path.join(model_save_dir, f'model_wf_run_{i+1}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model from run {i+1} saved to {model_save_path}")

        # evaluate the model
        _, _, y_pred, y_true, conf = eval_loop(model,
                                               expt_type,
                                               test_dataloader,
                                               device,
                                               len(test_dataset),
                                               loss_type,
                                               scale=1)

        m = gr_metrics(y_pred, y_true)
        # classwise_FScores = class_FScore(y_pred, y_true, expt_type)
        if 'Precision' in metrics_dict:
            metrics_dict['Precision'].append(m[0])
            metrics_dict['Recall'].append(m[1])
            metrics_dict['FScore'].append(m[2])
            metrics_dict['OE'].append(m[3])
        else:
            metrics_dict['Precision'] = [m[0]]
            metrics_dict['Recall'] = [m[1]]
            metrics_dict['FScore'] = [m[2]]
            metrics_dict['OE'] = [m[3]]

        cm = confusion_matrix(y_true, y_pred) 
        print(f"Confusion Matrix for run {i+1}:\n{cm}")
        cm_save_path = os.path.join(results_dir, f'confusion_matrix_wf_run_{i+1}.csv')
        pd.DataFrame(cm).to_csv(cm_save_path)
        print(f"Confusion matrix saved to {cm_save_path}")

    df = pd.DataFrame(metrics_dict) # record the metrics
    metrics_save_path = os.path.join(results_dir, f'metrics_wf_run_{i+1}.csv')
    df.to_csv(metrics_save_path, index = False)
    print(f"Metrics saved to {metrics_save_path}")

    return df['FScore'].median() # take the median of FScore


if __name__ == "__main__": # 脚本被直接运行时代码块被运行
    model_types = ('avg-pool', 'lstm+att', 'lstm')

    experiment_type = (4, 5)

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--expt-type", type=int, default=5, choices=experiment_type,
                        help="expt type")

    parser.add_argument("--batch-size", type=int, default=8,
                        help="batch size")

    parser.add_argument("--epochs", type=int, default=50,
                        help="number of epochs")

    parser.add_argument("--num-runs", type=int, default=50,
                        help="number of runs")

    parser.add_argument("--early-stop", type=int, default=10,
                        help="early stop limit")

    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="hidden dimensions")

    parser.add_argument("--embed-dim", type=int, default=768,
                        help="embedding dimensions")

    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of layers")

    parser.add_argument("--dropout", type=float, default=0.3,
                        help="dropout probablity")

    parser.add_argument("--learning-rate", type=float, default=0.002,
                        help="learning rate")

    parser.add_argument("--scale", type=float, default=1.8,
                        help="scale factor alpha")

    parser.add_argument("--data-dir", type=str, default="",
                        help="directory for data")

    parser.add_argument("--model-type", type=str, default="lstm+att",
                        choices=model_types, help="type of model")

    args = parser.parse_args()
    main(args.__dict__)
