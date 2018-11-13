import os
import pickle
import numpy as np
from scipy import spatial


model_path = './models/'
loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word]]

==========================================================================
"""

def convertToWrite(w1, w2):
    return '"' + w1 + ':' + w2 + '"'

def findAnswers(dictionary, embeddings, filename, outfile='result.txt'):
    # f = open(filename, encoding="utf8")
    f = open(filename, 'r')
    outfile += '_' + loss_model + '.txt'
    out = open(outfile,"w")
    for line in f:
        #print(line)
        parts = line.split('||')
        ques = list(map(lambda s: s.strip('\n"').split(':'), parts[0].split(',')))
        choices = list(map(lambda s: s.strip('\n"').split(':'), parts[1].split(',')))
        print(ques)
        print(choices)
        print('next')
        for w in ques:
            w[0] = w[0] if w[0] in dictionary else 'UNK'
            w[1] = w[1] if w[1] in dictionary else 'UNK'
        for w in choices:
            w[0] = w[0] if w[0] in dictionary else 'UNK'
            w[1] = w[1] if w[1] in dictionary else 'UNK'
        print(ques)
        print(choices)

        qvecs = [(embeddings[dictionary[word[1]]] - embeddings[dictionary[word[0]]]) for word in ques]
        cvecs = [(embeddings[dictionary[word[1]]] - embeddings[dictionary[word[0]]]) for word in choices]
        qvec = np.mean(qvecs,axis=0)
        print(qvec.shape)
        sims = [1 - spatial.distance.cosine(qvec, vec) for vec in cvecs]
        imax = np.argmax(sims)
        imin = np.argmin(sims)
        #out.write(parts[0])
        outstr = ''
        for w in choices:
            outstr += convertToWrite(w[0], w[1]) + ' '
        outstr +=  convertToWrite(choices[imin][0], choices[imin][1]) + ' ' + convertToWrite(choices[imax][0], choices[imax][1]) + '\n'
        print(outstr)
        out.write(outstr)
        print(sims)
        print(qvecs[0].shape)
    f.close()
    out.close()


#findAnswers(dictionary, embeddings, 'word_analogy_dev.txt')
findAnswers(dictionary, embeddings, 'word_analogy_test.txt', 'word_analogy_test_predictions')

