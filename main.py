import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = load_dataset("md_gender_bias")
#print (len(dataset['train']['text']))
#print (len(dataset['train']['labels']))

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

tokenized = []
for x in dataset['train']['text']:
    tokenized.append(tokenizer.encode(x))
#Add padding to tokens so they are same size
max_len = 0
for i in tokenized:
    if len(i) > max_len:
        max_len = len(i)

padded_token = np.array([i + [0]*max_len-len[i] for i in tokenized])

#transform np array to tf tensor
tensor_padded = tf.convert_to_tensor(padded_token)

#send input tensor to TFDistillBert
model_output = model(tensor_padded)

#get features from output
features = model_output[0][:,0,:].numpy()

#Get ground label
label = dataset['train']['labels']

#split train test
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

#Use Scikit logic regression to train
clf = LogisticRegression()
clf.fit(train_features, train_labels)

#test score
clf.score(test_features, test_labels)
