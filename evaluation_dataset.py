from datasets import load_dataset, concatenate_datasets
class EvalDataLoader:

    def load_eval_data(self):
        new_data = load_dataset('md_gender_bias', 'new_data', split="train")
        new_data = new_data.map(self.convertLabels)
        new_data = new_data.rename_column('labels', 'label')
        print(new_data)
        return new_data

    def convertLabels(self, data):
        data['labels'] = data['labels'][0]
        return data
