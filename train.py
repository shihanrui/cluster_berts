import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 2
batch_size = 4
learning_rate = 0.001

# dataset


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
    input_pt = tokenizer(["上呼吸道感染", "感染性发热"], return_tensors='pt',
                         padding=True, truncation=True).to(device)
    labels = torch.tensor([0, 1])
    labels = labels.to(device)
    # Forward pass
    outputs = model(input_pt['input_ids'], input_pt['attention_mask'])
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print('Finished Training')
