from datasets import load_dataset, concatenate_datasets
class TestDataLoader:

    def load_funpedia_test(split="test"):
        funpedia = load_dataset('md_gender_bias', 'funpedia', split=split)
        funpedia = funpedia.rename_column('gender', 'label')
        return funpedia

    def load_imageChat_test(split="test"):
        imageChat = load_dataset('md_gender_bias', 'image_chat', split=split)
        return imageChat

    def load_wizard_test(split="test"):
        wizard = load_dataset('md_gender_bias', 'wizard', split=split)
        wizard = wizard.rename_column('gender', 'label')
        return wizard

    def load_yelp_test(split="test"):
        yelp = load_dataset('md_gender_bias', 'yelp_inferred', split=split)
        yelp = yelp.rename_column('ternary_label', 'label')
        return yelp

    def load_convai2_test(split="test"):
        convai2 = load_dataset('md_gender_bias', 'convai2_inferred', split=split)
        convai2 = convai2.rename_column('ternary_label', 'label')
        return convai2

    def load_light_test(split="test"):
        light = load_dataset('md_gender_bias', 'light_inferred', split=split)
        light = light.rename_column('ternary_label', 'label')

        return light

    def load_opensub_test(split="test"):
        openSub = load_dataset('md_gender_bias', 'opensubtitles_inferred', split=split)
        openSub = openSub.rename_column('ternary_label', 'label')

        return openSub
