
from datasets import load_dataset, concatenate_datasets
class TrainDataLoader:
    def load_train_data(self):
        return concatenate_datasets([self.load_about_data("train"), self.load_as_data("train"), self.load_to_data("train")])

    def load_about_train_data(self):
        return self.load_about_data("train")

    def load_as_train_data(self):
        return self.load_as_data("train")

    def load_to_train_data(self):
        return self.load_to_data("train")

    def load_about_data(self, split):
        funpedia = load_dataset('md_gender_bias', 'funpedia', split=split)
        funpedia = funpedia.rename_column('gender', 'label')
        funpedia = funpedia.remove_columns("title")
        funpedia = funpedia.remove_columns("persona")
        funpedia = funpedia.filter(lambda row: row['label'] != 0)
        funpedia = funpedia.map(self.modifyAboutLables)
        # imageChat = load_dataset('md_gender_bias', 'image_chat', split=split)

        wizard = load_dataset('md_gender_bias', 'wizard', split=split)
        wizard = wizard.rename_column('gender', 'label')
        wizard = wizard.remove_columns("chosen_topic")
        wizard = wizard.filter(lambda row: row['label'] != 0)
        wizard = wizard.map(self.modifyAboutLables)
        print (funpedia.features.type)
        print (wizard.features.type)
        assert funpedia.features.type == wizard.features.type
        return concatenate_datasets([wizard, funpedia])

    def load_as_data(self, split):
        yelp = load_dataset('md_gender_bias', 'yelp_inferred', split=split)
        yelp = yelp.rename_column('binary_label', 'label')
        yelp = yelp.remove_columns("binary_score")
        yelp = yelp.filter(lambda row: row['label'] == 0)
        yelp = yelp.map(self.modifyAsLables)

        convai2 = load_dataset('md_gender_bias', 'convai2_inferred', split=split)
        convai2 = convai2.rename_column('binary_label', 'label')
        convai2 = convai2.remove_columns("binary_score")
        convai2 = convai2.remove_columns("ternary_score")
        convai2 = convai2.remove_columns("ternary_label")
        convai2 = convai2.filter(lambda row: row['label'] == 0)
        convai2 = convai2.map(self.modifyAsLables)

        assert convai2.features.type == yelp.features.type
        return concatenate_datasets([convai2, yelp])

    def load_to_data(self, split):
        light = load_dataset('md_gender_bias', 'light_inferred', split=split)
        light = light.rename_column('ternary_label', 'label')
        light = light.remove_columns("binary_score")
        light = light.remove_columns("ternary_score")
        light = light.remove_columns("binary_label")
        light = light.filter(lambda row: row['label'] != 2)

        openSub = load_dataset('md_gender_bias', 'opensubtitles_inferred', split=split)
        openSub = openSub.rename_column('ternary_label', 'label')
        openSub = openSub.remove_columns("binary_score")
        openSub = openSub.remove_columns("ternary_score")
        openSub = openSub.remove_columns("binary_label")
        openSub = openSub.filter(lambda row: row['label'] != 2)

        light = light.map(self.modifyToLables)
        openSub = openSub.map(self.modifyToLables)

        return concatenate_datasets([light, openSub])

    def modifyAboutLables(self, row):
        if row['label'] == 0:
            row['label'] = 6
        elif row['label'] == 1:
            row['label'] = 0
        else:
            row['label'] = 1
        return row
    def modifyAsLables(self, row):
        if row['label'] == 0:
            row['label'] = 4
        elif row['label'] == 1:
            row['label'] = 5
        return row
    def modifyToLables(self, row):
        if row['label'] == 0:
            row['label'] = 2
        elif row['label'] == 1:
            row['label'] = 3
        else:
            row['label'] = 7
        return row
