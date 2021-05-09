from transformers import DistilBertTokenizer, DistilBertModel
from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import DataLoader
from model import DistillBERTClassifier
from custom_dataset import GenderDataset
import torch
from training_dataloader import TrainDataLoader

dataset = TrainDataLoader().load_train_data()
#print (len(dataset['train']['text']))
#print (len(dataset['train']['labels']))

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

tokenized = []
for x in dataset['text']:
    tokenized.append(tokenizer.encode(x))

#Add padding to tokens so they are same size
max_len = 0
for i in tokenized:
    if len(i) > max_len:
        max_len = len(i)

padded_token = np.array([i + [0]*(max_len-len(i)) for i in tokenized])

#transform np array to torch tensor
tensor_padded = torch.tensor(padded_token)

#hyperparameters##
BATCHSIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 2
####################

#create custom dataloader
tokened_dict = {'tokens' : tensor_padded, 'labels' : dataset['label']}
tokened_ds = Dataset.from_dict(tokened_dict)
#print(tokened_ds)
train_ds = GenderDataset(tokened_ds)
train_params = {'batch_size': BATCHSIZE,
                'shuffle': True,
                'num_workers': 1
                }
train_dl = DataLoader(training_set, **train_params)


#send input tensor to DistillBert
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
clf = DistillBERTClassifier(model, 768, 6)

#for baseline experiment
loss_func = torch.nn.CrossEntropyLoss()

#for label smoothing
loss_func = ScoreLabelSmoothedCrossEntropyLoss(label_smooth_temp = 0.1, label_smooth_power = 1)

opt = torch.optim.Adam(clf.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    clf.train()
    for _,data in enumerate(train_dl, 0):
        labels = data['label'].to(device, dtype = torch.long)
        tokens = data['tokens'].to(device, dtype = torch.float)
        outputs = model(tokens).squeeze()
        loss = loss_func(outputs, tokens)
        loss.backward()
        opt.step()
        opt.zero_grad()
