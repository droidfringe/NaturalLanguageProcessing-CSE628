#!/bin/python
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pickle
import os

clusterInfo = []
dictionary = {}

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    global clusterInfo, dictionary
    clusterFIleName = 'clusterInfo.pkl'
    # with open('word2vec_nce.model', 'rb') as fid:
    #     dictionary, steps, embeddings = pickle.load(fid)
    with open('dict.pkl', 'rb') as fid:
        dictionary = pickle.load(fid)

    if(os.path.isfile(clusterFIleName)):
        with open(clusterFIleName,'rb') as fid:
            numClusters, clusterInfo = pickle.load(fid)
    else:
        clusterInfo = np.zeros((len(dictionary)+1,))
        # numClusters = 100
        # clusterInfo = ClusterEmbeddings(embeddings, numClusters)
        # with open(clusterFIleName, 'wb') as fid:
        #     pickle.dump([numClusters, clusterInfo], fid)

    print('Preprocessing complete')


def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    # if(len(word) > 2 and word[-2:] == 'ed'):
    #     ftrs.append("ENDS_WITH_ED")
    # if(len(word) > 3 and word[-3:] == 'ing'):
    #     ftrs.append("ENDS_WITH_ING")
    # if(len(word) > 2 and word[-2:] == 'er'):
    #     ftrs.append("ENDS_WITH_ER")
    # if(len(word) > 2 and word[-2:] == 'my'):
    #     ftrs.append("ENDS_WITH_MY")
    #
    # if (len(word) > 2 and word[-2:] == 'ly'):
    #     ftrs.append("ENDS_WITH_LY")


    # Start features
    if(word[0] >= 'A' and word[0] <= 'Z'):
        ftrs.append("STARTS_WITH_CAPS")
    if(word[0] >= '0' and word[0] <= '9'):
        ftrs.append("STARTS_WITH_NUMERIC")


    # X category specific features
    symbols = ['@', '#', '!', ':', '.', '?', ',', ';', '&', '_', "'"]
    for symbol in symbols:
        if(word[0] == symbol):
            ftrs.append("STARTS_WITH_SYMBOL_"+symbol)
        if(symbol in word):
            ftrs.append("CONTAINS_SYMBOL_" + symbol)

    if (len(word) > 3 and word[0:4].lower() == 'http'):
        ftrs.append("STARTS_WITH_URL")
    if (len(word) > 2 and word[0:3].lower() == 'www'):
        ftrs.append("STARTS_WITH_URL")

    # DO NOT INCLUDE IN FINAL
    if(word.lower() == 'her'):
        ftrs.append("POSSIBLE_PRONOUN")
    if(word.lower() == 'his'):
        ftrs.append("POSSIBLE_PRONOUN")
    if(word.lower() == 'she'):
        ftrs.append("POSSIBLE_PRONOUN")
    if(word.lower() == 'him'):
        ftrs.append("POSSIBLE_PRONOUN")
    if(word.lower() == 'he'):
        ftrs.append("POSSIBLE_PRONOUN")

    # DO NOT INCLUDE IN FINAL
    # if(len(word) > 3 and word[-3:].lower() == 'ate'):
    #     ftrs.append("ENDS_WITH_ATE")
    # if(len(word) > 2 and word[-2:].lower() == 'en'):
    #     ftrs.append("ENDS_WITH_EN")
    # if(len(word) > 3 and word[-3:].lower() == 'ish'):
    #     ftrs.append("ENDS_WITH_ISH")
    # if(len(word) > 2 and word[-2:].lower() == 'fy'):
    #     ftrs.append("ENDS_WITH_FY")
    # if(len(word) > 3 and word[-3:].lower() == 'ize'):
    #     ftrs.append("ENDS_WITH_IZE")
    # if(len(word) > 3 and word[-3:].lower() == 'ise'):
    #     ftrs.append("ENDS_WITH_ISE")


    # Suffix features
    ftrs.append("ENDS_WITH_" + word[-1].lower())
    if(len(word) >= 2):
        ftrs.append("ENDS_WITH_" + word[-2:].lower())
    if (len(word) >= 3):
        ftrs.append("ENDS_WITH_" + word[-3:].lower())

    # # Adjective specific suffixes
    if(len(word) > 4 and word[-4:].lower() == 'able'):
        ftrs.append("POSSIBLE_ADJ")
    if(len(word) > 3 and word[-3:].lower() == 'ant'):
        ftrs.append("POSSIBLE_ADJ")
    if(len(word) > 2 and word[-2:].lower() == 'al'):
        ftrs.append("POSSIBLE_ADJ")
    if(len(word) > 3 and word[-3:].lower() == 'ent'):
        ftrs.append("POSSIBLE_ADJ")
    if(len(word) > 3 and word[-3:].lower() == 'ful'):
        ftrs.append("POSSIBLE_ADJ")
    if(len(word) > 2 and word[-2:].lower() == 'ic'):
        ftrs.append("POSSIBLE_ADJ")
    if(len(word) > 4 and word[-4:].lower() == 'ible'):
        ftrs.append("POSSIBLE_ADJ")
    if (len(word) > 3 and word[-3:].lower() == 'ive'):
        ftrs.append("POSSIBLE_ADJ")
    if (len(word) > 1 and word[-1:].lower() == 'y'):
        ftrs.append("POSSIBLE_ADJ")
    if (len(word) > 4 and word[-4:].lower() == 'less'):
        ftrs.append("POSSIBLE_ADJ")
    if (len(word) > 3 and word[-3:].lower() == 'ous'):
        ftrs.append("POSSIBLE_ADJ")



    # Adverb specific suffixes
    if(len(word) > 4 and word[-4:].lower() == 'ward'):
        ftrs.append("POSSIBLE_ADV")
    if(len(word) > 5 and word[-5:].lower() == 'wards'):
        ftrs.append("POSSIBLE_ADV")
    if(len(word) > 4 and word[-4:].lower() == 'wise'):
        ftrs.append("POSSIBLE_ADV")


    # DO NOT INCLUDE IN FINAL
    # Verb specific suffixes
    # if(len(word) > 3 and word[-3:].lower() == 'ate'):
    #     ftrs.append("POSSIBLE_VERB")
    # if(len(word) > 2 and word[-2:].lower() == 'en'):
    #     ftrs.append("POSSIBLE_VERB")
    # if(len(word) > 3 and word[-3:].lower() == 'ify'):
    #     ftrs.append("POSSIBLE_VERB")
    # if (len(word) > 3 and word[-3:].lower() == 'ise'):
    #     ftrs.append("POSSIBLE_VERB")
    # if (len(word) > 3 and word[-3:].lower() == 'ize'):
    #     ftrs.append("POSSIBLE_VERB")


    # DO NOT INCLUDE IN FINAL
    # Noun specific suffixes
    # if(len(word) > 3 and word[-3:].lower() == 'age'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if(len(word) > 2 and word[-2:].lower() == 'al'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if(len(word) > 4 and word[-4:].lower() == 'ance'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if(len(word) > 4 and word[-4:].lower() == 'ence'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if (len(word) > 3 and word[-3:].lower() == 'dom'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if (len(word) > 2 and word[-2:].lower() == 'ee'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if (len(word) > 2 and word[-2:].lower() == 'er'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if (len(word) > 2 and word[-2:].lower() == 'or'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if(len(word) > 4 and word[-4:].lower() == 'hood'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if (len(word) > 3 and word[-3:].lower() == 'ism'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if (len(word) > 3 and word[-3:].lower() == 'ist'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if (len(word) > 3 and word[-3:].lower() == 'ity'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if (len(word) > 2 and word[-2:].lower() == 'ty'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if(len(word) > 4 and word[-4:].lower() == 'ment'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if(len(word) > 4 and word[-4:].lower() == 'ness'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if (len(word) > 2 and word[-2:].lower() == 'ry'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if(len(word) > 4 and word[-4:].lower() == 'sion'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if(len(word) > 4 and word[-4:].lower() == 'tion'):
    #     ftrs.append("POSSIBLE_NOUN")
    # if(len(word) > 4 and word[-4:].lower() == 'ship'):
    #     ftrs.append("POSSIBLE_NOUN")


    # # Clustering features
    # Comment for LR, use for CRF
    global clusterInfo, dictionary
    if word in dictionary:
        clusterNo = clusterInfo[dictionary[word]]
    else:
        clusterNo = clusterInfo[dictionary['UNK']]
    ftrs.append("CLUSTER_"+str(clusterNo))

    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs


def ClusterEmbeddings(embeddings, numCluters):
    # seed = 0 for reproducibility
    # kmeans = KMeans(n_clusters=numCluters, random_state=0).fit(embeddings)
    kmeans = MiniBatchKMeans(n_clusters=numCluters, random_state=0, n_init=20).fit(embeddings)
    clusters = kmeans.predict(embeddings)
    return clusters


if __name__ == "__main__":
    sents = [
    [ "I", "love", "food", "and", "playing", "with", "red", "ball" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
