import torch
import torch.nn as nn

from mixout import MixLinear

from transformers import DistilBertModel
class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.linear1 = nn.Linear(784, 300)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(300, 100)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, input):
        input = self.drop1(self.relu1(self.linear1(input)))
        input = self.drop2(self.relu2(self.linear2(input)))
        return self.linear3(input)


def main():
    # Prepare the model configuration from pretraining. In this example,
    # I just use all one parameters as the pretraiend model configuration.
    model_config = {
        'linear1.weight': torch.ones(300, 784), 'linear1.bias': torch.zeros(300),
        'linear2.weight': torch.ones(100, 300), 'linear2.bias': torch.zeros(100),
        'linear3.weight': torch.ones(10, 100), 'linear3.bias': torch.zeros(10)
    }
    # Set up a model for finetuning.
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')


    #smodel.load_state_dict(model_config)

    # From now on, we are going to replace dropout with mixout.
    # Since dropout drops all parameters outgoing from the dropped neuron,
    # mixout mixes the parameters of the nn.Linear right after the nn.Dropout.
    for name, module in model.named_modules():
        if name == 'transformer.layer.0':
            print (module)
            for name_c, module_c, in module.named_modules():
                print (module_c)
        '''
        if name in ['transformer.layer.0.attention.dropout',
                    'transformer.layer.1.attention.dropout',
                    'transformer.layer.2.attention.dropout',
                    'transformer.layer.3.attention.dropout',
                    'transformer.layer.4.attention.dropout',
                    'transformer.layer.5.attention.dropout',
                    'transformer.layer.0.ffn.dropout',
                    'transformer.layer.1.ffn.dropout',
                    'transformer.layer.2.ffn.dropout',
                    'transformer.layer.3.ffn.dropout',
                    'transformer.layer.4.ffn.dropout',
                    'transformer.layer.5.ffn.dropout'] and isinstance(module, nn.Dropout):
            setattr(model, name, nn.Dropout(0))
        if name in ['transformer.layer.0.attention.q_lin',
                    'transformer.layer.1.attention.q_lin',
                    'transformer.layer.2.attention.q_lin',
                    'transformer.layer.3.attention.q_lin',
                    'transformer.layer.4.attention.q_lin',
                    'transformer.layer.5.attention.q_lin',
                    'transformer.layer.0.ffn.lin1',
                    'transformer.layer.1.ffn.lin1'
                    'transformer.layer.2.ffn.lin1'
                    'transformer.layer.3.ffn.lin1'
                    'transformer.layer.4.ffn.lin1',
                    'transformer.layer.5.ffn.lin1'] and isinstance(module, nn.Linear):
            target_state_dict = module.state_dict()
            bias = True if module.bias is not None else False
            new_module = MixLinear(module.in_features, module.out_features,
                                   bias, target_state_dict['weight'], 0.9)
            new_module.load_state_dict(target_state_dict)
            setattr(model, name, new_module)
        '''



if __name__ == "__main__":
    main()