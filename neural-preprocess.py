# freakin' python2 support:
from __future__ import print_function
from __future__ import division
# from sklearn import datasets
# DEPRECIATED
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import pandas as pd
import pdb
# import tensorflow as tf
# import gensim
# import pickle
import cPickle as pickle

class preprocess:

    def __init__(self):
        self.RANDOM_SEED = 42
        # tf.set_random_seed(self.RANDOM_SEED)
        # pass

    def target_to_num(self, array, uniqueArray):
        """Take in an array of features and convert them to numeric representation"""
        numberArray = []
        for item in array:
            numberArray.append(uniqueArray.index(item))
        assert(len(numberArray) == len(array))
        return numberArray

    def load_data(self, wordvecFile, catvecFile, inFile):
        """Read in data from old maxent classifier model"""
        # load word vectors for features
        print("loading word2vec")
        # self.token_model = gensim.models.Word2Vec.load_word2vec_format(wordvecFile, binary=True)
        # not a true word2vec binary:
        self.token_model = pickle.load(open(wordvecFile, 'rb'))
        # self.cat_model = gensim.models.Word2Vec.load_word2vec_format(catvecFile, binary=True)
        print("loading feats file")
        # catTagged = open(inFile, 'r')
        target = []
        data = []
        set_of_feats = set()
        # maxent_lines = []
        # a 'dictionary' for classes and features
        # load data from file
        print("building feature vectors")
        # sanity check to make sure we're getting words for every one
        words = []
        # l = 0
        for line in open(inFile, 'r'):
            # l += 1
            # print(line)
            line = line.split()
            target.append(line.pop(0))
            feats = []
            # print('len(line:)', len(line))
            for feat in line:
                # print(feat, feat[0:2])
                if feat[-3:] == '1.0':
                    if feat[0:2] == "PN":
                        word = feat.replace('PN=', '').replace(':1.0', '')
                        # remove sense numbers
                        if word[-3:] in ['.01', '.02', '.03', '.04',
                                         '.05', '.06', '.07', '.08',
                                         '.09', '.10', '.11', '.12',
                                         '.13', '.14', '.15', '.16',
                                         '.17', '.18', '.19', '.20']:
                            word = word[0:-3]
                    # print("feat", feat)
                    feats.append(feat)
                    set_of_feats.add(feat)
            # append the word at the end, and pull if off
            # before converting to onehots
            # print("word", word, l)
            assert(word != '')
            feats.append(word)
            words.append(word)
            data.append(feats)
        assert(len(words) == len(data))
        print("len(set_of_feats)", len(set_of_feats))
        # convert the feats into onehots
        list_of_feats = list(set_of_feats)
        all_X = []
        i = 0
        total = len(data)
        words_in = []
        words_out = []
        total_words = []
        for datum in data:
            # TODO there is a gradual memory leak here. Why?
            # just some output so we know it's not frozen
            if i % 100 == 0:
                print("On", i, "of", total)
            i += 1
            # trying 1-hots first
            word = datum[-1]
            total_words.append(word)
            oov_shape = self.token_model['the'].shape
            if word in self.token_model:
                words_in.append(word)
                wordVec = self.token_model[word]
            else:
                # If we can't find a word, replace it with Gaussian noise
                words_out.append(word)
                wordVec = np.random.normal(0, 0.5, oov_shape)
            featVec = []
            # appending bias
            featVec.append(1)
            for feat in list_of_feats:
                if feat in datum:
                    featVec.append(1)
                else:
                    featVec.append(0)
            assert(len(featVec) == len(list_of_feats) + 1)
            # when we're ready, add the feature vectors back
            # featVec = featVec + wordVec.tolist()
            all_X.append(featVec)
        with open('oov.txt', 'w') as outs:
            for wo in words_out:
                outs.write(wo + '\n')
        print("oovs:", len(words_out) / len(total_words))
        print("total oovs:", len(words_out))
        # add maxent baseline stuff here
        # print("Creating maxent baseline")
        assert(len(target) == len(all_X))
        # begin creating neural data
        print("converting to 1-hot vectors")
        classes = sorted(set(target))
        print("len(classes", len(classes))
        # convert into vectors in numpy
        target_as_num = np.asarray(self.target_to_num(target, classes))
        print("len(target", len(target))
        print("shape of target_as_num", target_as_num.shape)
        # and into one hot outputs
        all_Y = pd.get_dummies(target_as_num).values

        # make sure all_X is a numpy array
        all_X = np.asarray(all_X)

        print("shape all_Y", all_Y.shape)
        print("len(data)", len(data))
        data = np.asarray(data)
        print("shape data", data.shape)
        # print("prepending columns of 1s for the bias")
        # pdb.set_trace()
        # Prepend the column of 1s for bias
        # N, M  = data.shape
        # all_X = np.ones((N, M + 1))
        # all_X[:, 1:] = data
        print("shape of all_X", all_X.shape)

        train_X, test_X, train_Y, test_Y = train_test_split(all_X, all_Y, test_size=0.333, random_state=self.RANDOM_SEED)
        # make dev set
        val_X, test_X = np.split(test_X, 2)
        val_Y, test_Y = np.split(test_Y, 2)
        return train_X, test_X, val_X, train_Y, test_Y, val_Y

    def conv_to_1_hot(self, data):
        # quickly convert data into one hots
        # possibly broken -- originally written by some dude on the internet
        # use the pd.get_dummies(OBJECT).values function above instead
        num_labels = len(np.unique(data))
        all_ones = np.eye(num_labels)[data]  # One liner trick!
        return all_ones

if __name__ == '__main__':
    p = preprocess()
    # word2vec file, cat2vec file, input file
    if len(sys.argv) < 4:
        sys.exit("""
Please run this program like so:
python2 neural-preprocessor.py vectors.bin cat_embedd.bin inputfile.txt
        """)
    train_X, test_X, val_X, train_Y, test_Y, val_Y = p.load_data(sys.argv[1], sys.argv[2], sys.argv[3])
    print("train_X.shape", train_X.shape)
    print("test_X.shape", test_X.shape)
    print("train_Y.shape", train_Y.shape)
    print("test_Y.shape", test_Y.shape)
    print("val_X.shape", val_X.shape)
    print("val_Y.shape", val_Y.shape)
    print("writing pickles")
    # TODO spiking memory error happens here. Why?!?!
    pickle.dump(train_X, open('train_X.pkl', 'wb'))
    pickle.dump(train_Y, open('train_Y.pkl', 'wb'))
    pickle.dump(test_X, open('test_X.pkl', 'wb'))
    pickle.dump(test_Y, open('test_Y.pkl', 'wb'))
    pickle.dump(val_X, open('val_X.pkl', 'wb'))
    pickle.dump(val_Y, open('val_Y.pkl', 'wb'))
