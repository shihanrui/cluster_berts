from config import *
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans, KMeans
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def encoding(model, tokenizer, sentences):
    model.eval()
    model.to(device)
    max_char_len = 512
    sents_inputs = tokenizer(sentences, return_tensors='pt',
                             max_length=max_char_len, padding="max_length", truncation=True)
    input_ids = sents_inputs['input_ids']
    dataloader = DataLoader(input_ids, batch_size=8, shuffle=False)
    sents_vec = []
    tqdm_batch_iterator = tqdm(dataloader, desc='sentence encoding ')
    for index, batch in enumerate(tqdm_batch_iterator):
        input_ids = batch
        input_ids = input_ids.to(device)
        sents_vec.append(model(input_ids)[
                         'pooler_output'].detach().cpu().numpy().tolist())
    torch.cuda.empty_cache()
    sents_vec = [np.array(xi) for x in sents_vec for xi in x]
    return sents_vec


if __name__ == '__main__':

    df_train = preprocess_data(train_path)
    df_test = preprocess_data(test_path)
    tokenizer = AutoTokenizer.from_pretrained(model)
    auto_model = AutoModel.from_pretrained(model)
    train_vec = encoding(auto_model, tokenizer,
                         df_train.sentence.to_list())
    test_vec = encoding(auto_model, tokenizer,
                         df_test.sentence.to_list())

    print('kmeans begins..')
    mb_kmeans = MiniBatchKMeans(n_clusters=3)
    train_pred = mb_kmeans.fit_predict(pd.DataFrame(train_vec))
    test_pred = mb_kmeans.fit_predict(pd.DataFrame(test_vec))
    print('kmeans finished.')

    # 保存聚类结果
    feat_names_Kmeans = "Kmeans_" + str(3)
    train_kmeans = pd.concat(
        [df_train, pd.Series(name=feat_names_Kmeans, data=train_pred)], axis=1)
    train_kmeans.to_csv("./train_clustered.csv", index=False)

    test_kmeans = pd.concat([df_test, pd.Series(name=feat_names_Kmeans, data=test_pred)], axis=1)
    test_kmeans.to_csv("./test_clustered.csv", index=False)   

