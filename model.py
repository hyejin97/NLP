import torch
    
class DistillBERTClassifier(torch.nn.Module):
    def __init__(self):
        super(distill_bert, hidden_size, num_labels).__init__()
        self.distill_bert = distill_bert
        self.drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_tensor):
        distilbert_output = self.distill_bert(input_tensor)
        features = distilbert_output
        output_1 = self.drop(features)
        output = self.out(output_1)
        return output