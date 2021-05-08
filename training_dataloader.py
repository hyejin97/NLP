from datasets import load_dataset, concatenate_datasets
class TrainDataLoader:
    def load_train_data(self):
        return concatenate_datasets(self.load_about_data("train"), self.load_as_data("train"), self.load_to_data("train"))

    def load_about_train_data(self):
        return self.load_about_data("train")

    def load_as_train_data(self):
        return self.load_as_data("train")

    def load_to_train_data(self):
        return self.load_to_data("train")

    def load_about_data(self, split):
        funpedia = load_dataset('md_gender_bias', 'funpedia', split=split)
        funpedia = funpedia.rename_column('gender', 'label')

        imageChat = load_dataset('md_gender_bias', 'image_chat', split=split)

        wizard = load_dataset('md_gender_bias', 'wizard', split=split)
        wizard = wizard.rename_column('gender', 'label')
        return concatenate_datasets(wizard, funpedia)

    def load_as_data(self, split):
        yelp = load_dataset('md_gender_bias', 'yelp_inferred', split=split)
        yelp = yelp.rename_column('ternary_label', 'label')

        convai2 = load_dataset('md_gender_bias', 'convai2_inferred', split=split)
        convai2 = convai2.rename_column('ternary_label', 'label')

        return concatenate_datasets(yelp, convai2)

    def load_to_data(self, split):
        light = load_dataset('md_gender_bias', 'light_inferred', split=split)
        light = light.rename_column('ternary_label', 'label')

        openSub = load_dataset('md_gender_bias', 'opensubtitles_inferred', split=split)
        openSub = openSub.rename_column('ternary_label', 'label')

        return concatenate_datasets(light, openSub)
