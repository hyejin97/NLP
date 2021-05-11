import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

def augmentData(text, mode):
    '''mode is either insert or substitute'''
    aug = naw.WordEmbsAug(
        model_type='word2vec', model_path='GoogleNews-vectors-negative300.bin',
        action=mode)
    augmented_text = aug.augment(text)
    return augmented_text
