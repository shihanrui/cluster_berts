# 单gpu

import os  # NOQA: E402
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'  # NOQA: E402
# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.nn import DataParallel
import numpy as np
import pandas as pd
import math
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
# from sklearn.metrics import accuracy_score
import time
import copy
import random


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:2"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


seed_torch()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device_ids = [2, 3, 4, 5]

# Hyper-parameters
num_epochs = 2
batch_size = 8
learning_rate = 1e-5

# dataset


class TextDataset(Dataset):

    def __init__(self, type):

        # Initialize, download data.
        if type == 'train':
            df = pd.read_csv(
                '/home/lumenglin/shihanrui/cluster/train_data_clustered.csv')
        elif type == 'test':
            df = pd.read_csv(
                "/home/lumenglin/shihanrui/cluster/test_data_clustered.csv")
        self.n_samples = df.shape[0]

        self.content = df[df.columns[0]].tolist()
        self.label_orig = df[df.columns[1]].tolist()

        label_dict = np.load(
            '/home/lumenglin/shihanrui/reprocessed_data/label_dict_v2.npy', allow_pickle=True).tolist()
        label_dict_reverse = {}
        for key, val in label_dict.items():
            label_dict_reverse[val] = int(key)
        df["label"] = df[df.columns[1]].map(
            label_dict_reverse)
        self.label = df['label'].tolist()

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
        output_pt = output_pt - torch.mean(output_pt, dim=0)  # 差分
        output_pt = self.linear(output_pt)
        return output_pt


class Macbert(nn.Module):
    def __init__(self) -> None:
        super(Macbert, self).__init__()
        self.bert = AutoModel.from_pretrained(
            "hfl/chinese-macbert-base")  # bert预训练模型
        self.linear = nn.Linear(768, 198)

    def forward(self, input_ids_pt, attention_mask_pt):
        output_pt = self.bert(
            input_ids=input_ids_pt, attention_mask=attention_mask_pt)[1]
        # output_pt = output_pt - torch.mean(output_pt, dim=0)  # 差分
        output_pt = self.linear(output_pt)
        return output_pt


model = Macbert().to(device)
# model = torch.nn.DataParallel(
#     model, device_ids=device_ids)
# model = model.cuda(device=device_ids[0])

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
tokenizer.model_max_length = 512
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = test_loader

            running_loss = 0.0
            running_corrects = 0

            for i, (contents, labels, label_origs) in enumerate(tqdm(dataloader, desc=f"model {phase}ing:")):

                input_pt = tokenizer(
                    list(contents), return_tensors='pt', padding=True)
                input_pt = input_pt.to(device)
                labels = torch.tensor(list(labels)).to(device)

                # Forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(
                        input_pt['input_ids'], input_pt['attention_mask'])
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)

            # if (i+1) % 5 == 0:
            #     print(
            #         f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}, loss = {loss.item(): .4f}')
            if phase == 'train':
                #     scheduler.step()

                epoch_loss = running_loss/len(train_dataset)
                epoch_acc = running_corrects.double()/len(train_dataset)
            else:
                epoch_loss = running_loss/len(test_dataset)
                epoch_acc = running_corrects.double()/len(test_dataset)

            print(
                f'{phase}: Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
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


train_dataset = TextDataset('train')
test_dataset = TextDataset('test')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=2)
print('data loaded.')

total_samples = len(train_dataset)
n_iterations = math.ceil(total_samples/batch_size)
print(f"total_samples: {total_samples}, n_iterations:{n_iterations}")
model = train_model(model, criterion, optimizer,
                    step_lr_scheduler, num_epochs=num_epochs)

FILE = f'./model/macbert_finetuned.pth'
torch.save(model.state_dict(), FILE)
print(f"macbert_finetuned saved.")

# print(model.state_dict())
# loaded_model = Bert()
# # it takes the loaded dictionary, not the path file itself
# loaded_model.load_state_dict(torch.load(FILE))
# loaded_model.eval()

# print(loaded_model.state_dict())