from datasets import load_dataset, concatenate_datasets
class EvalDataLoader:

    def load_eval_data():
        funpedia = load_dataset('md_gender_bias', 'new_data')
