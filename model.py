import torch
import torch.nn.functional as F
    
class DistillBERTClassifier(torch.nn.Module):
    def __init__(self, distill_bert, hidden_size, num_labels):
        super().__init__()
        self.distill_bert = distill_bert
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_tensor):
        distill_bert_out = self.distill_bert(input_tensor)[0]
        features = distill_bert_out[:, 0, :]
        output_1 = self.drop(features)
        output_1 = self.dense(output_1)
        output_1 = F.gelu(output_1)
        output_2 = self.drop(output_1)
        output = self.out(output_2)
        return output
