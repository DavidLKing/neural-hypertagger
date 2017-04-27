# Neural Hypertagging!

Currently, the we don't have category embeddings yet, but there is a place holder for it in the code. 

The code was tested using a subset of the ht.feats file. Running it on the full dataset causes memory errors since we're manually creating one-hot vectors. The initial tests, detailed [here](http://ling.osu.edu/~king/hypertagging.pdf), show that we are getting some payoff switching from a maxent model to a multilayer perceptron, but we can't know for sure until we can run it across all the data.

To run the program, first preprocess your data:
`python2 neural-preprocessor.py vectors.bin cat_embedd.bin inputfile.txt`

Currently, `cat_embedd.bin` isn't being used. It's just a place hold. Feel free to point that to any random file on your system. `final_vocab.p` are lemmatized word vectors trained with word2vec. We'll eventually want to train our own, but feel free to used these. You can download those [here] (http://ling.osu.edu/~king/hypertagger/final_vocab.p). These were originally trained by Evan Jaffe.

One you have preprocessed data, just run this to evaluate the system accuracy on dev:
`python2 1feed-forward-tensorflow.py data/train_X.pkl data/val_X.pkl data/train_Y.pkl data/val_Y.pkl 0.001 0.1`
and for test:
`python2 1feed-forward-tensorflow.py data/train_X.pkl data/test_X.pkl data/train_Y.pkl data/test_Y.pkl 0.001 0.1`

Note, you can use `feed-forward-tensorflow.py`, `1feed-forward-tensorflow.py`, and `2feed-forward-tensorflow.py` for running a multi-layer perceptron with 0, 1, and 2 hidden layers, respectively. Feel free to alter whatever, but I think the primary concern is getting the sparse one-hot encodings to work.

Thanks! And let me know if you have any questions!
