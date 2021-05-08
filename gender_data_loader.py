from datasets import load_dataset, concatenate_datasets
class GenderedDataLoader:

dataset = load_dataset('md_gender_bias', 'gendered_words')

    def __getitem__(self, word):
        selected_word = dataset.filter(lambda example: example['word_masculine'] == (word) or example['word_feminine'] == (word))
        if len(selected_word) != 0:
            if(selected_word[0]['word_masculine'] == word):
                return selected_word[0]['word_feminine']
            else return selected_word[0]['word_masculine']
        else return word
