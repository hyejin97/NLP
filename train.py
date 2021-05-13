from transformers import DistilBertTokenizer, DistilBertModel
from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import DataLoader
from model import DistillBERTClassifier
import torch
from training_dataloader import TrainDataLoader
import numpy as np

device = torch.device("cuda")

#hyperparameters##
BATCHSIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 2
NUM_CLASSES = 3
MIX_OUT = 0
####################

#dataset = load_dataset("md_gender_bias")['train']
dataset = TrainDataLoader().load_train_data()

'''
#DEPENDENCY - If dataaugmentation
#you need to install nlpaug first and download word2vec pretrained model to run this code
#1. pip install nlpaug
#2. wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
#3. gzip -d GoogleNews-vectors-negative300.bin.gz

augmented_text = []
augmented_label = []
for i in range(len(dataset['text'])):
    inserttxt = augmentData(dataset['text'][i], 'insert')
    augmented_text.append(inserttxt)
    augmented_label.append(dataset['labels'][i])

    substtxt = augmentData(dataset['text'][i], 'substitute') 
    augmented_text.append(substtxt)
    augmented_label.append(dataset['labels'][i])


dataset['text'] = dataset['text'] + augmented_text
dataset['labels'] = dataset['labels'] + augmented_label
'''

#Tokenize data
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

onehotvec = []
for y in dataset['label']:
    vec = [0]*NUM_CLASSES
    vec[y] = 1
    onehotvec.append(vec)

#transform np array to torch tensor
tensor_padded = torch.tensor(padded_token)
label_vec = torch.tensor(onehotvec)


#create custom dataloader
tokened_dict = {'tokens' : tensor_padded, 'labels' : label_vec}
#tokened_dict = {'tokens' : tensor_padded, 'labels' : dataset['label']}
train_ds = Dataset.from_dict(tokened_dict)
train_ds.set_format(type='torch', columns = ['tokens', 'labels'])

train_params = {'batch_size': BATCHSIZE,
                'shuffle': True,
                'num_workers': 1
                }
train_dl = DataLoader(train_ds, **train_params)


#send input tensor to DistillBert
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
clf = DistillBERTClassifier(model, 768, NUM_CLASSES)

#loss for baseline experiment
loss_func = torch.nn.CrossEntropyLoss()

'''
#DEPENDENCY - loss for label smoothing
#loss_func = ScoreLabelSmoothedCrossEntropyLoss(temp = 0.2, power = 1, num_classes = NUM_CLASSES)
'''

opt = torch.optim.Adam(clf.parameters(), lr=LEARNING_RATE)

clf = clf.to(device)
for epoch in range(NUM_EPOCHS):
    clf.train()
    for _,data in enumerate(train_dl, 0):
        labels = data['labels'].to(device)
        tokens = data['tokens'].to(device)
        outputs = clf(tokens)
        loss = loss_func(outputs, labels)
        loss.backward()
        opt.step()
        opt.zero_grad()
    print("epoch: {}. loss: {}.".format(epoch, loss.item()))
'''
    # calculate Accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            labels = data['label'].to(device, dtype = torch.long)
            tokens = data['tokens'].to(device, dtype = torch.float)
            outputs = clf(tokens)
            _, predicted = torch.max(outputs.data, 1)
            total+= labels.size(0)
                
            # for gpu, bring the predicted and labels back to cpu fro python operations to work
            correct+= (predicted == labels).sum()
        accuracy = 100 * correct/total
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

'''
