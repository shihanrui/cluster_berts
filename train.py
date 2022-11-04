# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
# from sklearn.metrics import accuracy_score
import time
import copy
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# writer = SummaryWriter("runs/train")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 2
batch_size = 8
learning_rate = 1e-3

# dataset


class TextDataset(Dataset):

    def __init__(self, cluster_num):

        # Initialize, download data.
        df_train = pd.read_csv(
            'raw_data_clustered.csv')
        df_train = df_train[df_train[df_train.columns[2]] == cluster_num]
        self.n_samples = df_train.shape[0]

        self.content = df_train[df_train.columns[0]].tolist()
        self.label_orig = df_train[df_train.columns[1]].tolist()

        label_dict = np.load(
            '/home/lumenglin/shihanrui/reprocessed_data/label_dict_v2.npy', allow_pickle=True).tolist()
        label_dict_reverse = {}
        for key, val in label_dict.items():
            label_dict_reverse[val] = int(key)
        df_train["label"] = df_train[df_train.columns[1]].map(
            label_dict_reverse)
        self.label = df_train['label'].tolist()

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.content[index], self.label[index], self.label_orig[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


class Bert(nn.Module):
    def __init__(self) -> None:
        super(Bert, self).__init__()
        self.bert = AutoModel.from_pretrained(
            "hfl/chinese-macbert-base")  # bert预训练模型
        self.linear = nn.Linear(768, 198)

    def forward(self, input_ids_pt, attention_mask_pt):
        output_pt = self.bert(
            input_ids=input_ids_pt, attention_mask=attention_mask_pt)[1]
        output_pt = output_pt - torch.mean(output_pt, dim=0)
        output_pt = self.linear(output_pt)
        return output_pt


model = Bert().to(device)
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
tokenizer.model_max_length = 512
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        for i, (contents, labels, label_origs) in enumerate(tqdm(train_loader, desc=f"model{cluster_num} training")):

            input_pt = tokenizer(list(contents), return_tensors='pt',
                                 padding=True, truncation=True).to(device)
            labels = torch.tensor(list(labels)).to(device)

            # Forward pass
            # track history if only in train

            outputs = model(input_pt['input_ids'], input_pt['attention_mask'])
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)

            # if (i+1) % 5 == 0:
            #     print(
            #         f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}, loss = {loss.item(): .4f}')
        epoch_loss = running_loss/len(dataset)
        epoch_acc = running_corrects.double()/len(dataset)

        print(
            f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Finished Training')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


for cluster_num in range(3):

    print("#####################################")
    print(f"cluster {cluster_num} begin training...")

    dataset = TextDataset(cluster_num)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)

    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples/batch_size)

    # examples = iter(train_loader)
    # examples_contents, examples_labels, _ = examples.next()
    # print(examples_contents, examples_labels)
    # sys.exit()

    # writer.add_graph(model)
    # writer.close()
    # sys.exit()

    model = train_model(model, criterion, optimizer, num_epochs=20)

    FILE = f'./model/model{cluster_num}.pth'
    torch.save(model.state_dict(), FILE)
    print(f"model{cluster_num} saved.")

    # print(model.state_dict())
    # loaded_model = Bert()
    # # it takes the loaded dictionary, not the path file itself
    # loaded_model.load_state_dict(torch.load(FILE))
    # loaded_model.eval()

    # print(loaded_model.state_dict())
