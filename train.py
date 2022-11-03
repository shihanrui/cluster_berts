import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
from transformers import AutoTokenizer, AutoModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 2
batch_size = 4
learning_rate = 0.001

# dataset


class TextDataset(Dataset):

    def __init__(self):

        # Initialize, download data.
        df_train = pd.read_csv(
            '/home/hanrui/reprocessed_data/train_data_v2.csv')
        self.n_samples = df_train.shape[0]

        self.content = df_train[df_train.columns[0]].tolist()
        self.label_orig = df_train[df_train.columns[1]].tolist()

        label_dict = np.load(
            '/home/hanrui/reprocessed_data/label_dict_v2.npy', allow_pickle=True).tolist()
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


dataset = TextDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2)

total_samples = len(dataset)
n_iterations = math.ceil(total_samples/batch_size)


class Bert(nn.Module):
    def __init__(self) -> None:
        super(Bert, self).__init__()
        self.embedding = AutoModel.from_pretrained(
            "hfl/chinese-macbert-base")  # bert预训练模型
        self.linear = nn.Linear(768, 198)

    def forward(self, input_ids_pt, attention_mask_pt):
        output_pt = self.embedding(
            input_ids=input_ids_pt, attention_mask=attention_mask_pt)[1]
        output_pt = output_pt - torch.mean(output_pt, dim=0)
        output_pt = self.linear(output_pt)
        return output_pt


model = Bert().to(device)
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
tokenizer.model_max_length = 512
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# n_total_steps = len(train_loader)

for epoch in range(num_epochs):

    for i, (contents, labels, label_origs) in enumerate(train_loader):

        input_pt = tokenizer(list(contents), return_tensors='pt',
                             padding=True, truncation=True).to(device)
