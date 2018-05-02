import nltk
from nltk import word_tokenize

raw = open('datasets/WW_Dataset_seq.txt').read()
print(type(raw))

tokens = word_tokenize(raw)
corpus = [w.lower() for w in tokens]

spl = int(95*len(corpus)/100)
train = corpus[:spl]
test = corpus[spl:]

# Remove rare words from the corpus
fdist = nltk.FreqDist(train)
vocabulary = set(map(lambda x: x[0], filter(lambda x: x[1] >= 5, fdist)))

train = map(lambda x: x if x in vocabulary else "*unknown*", train)
test = map(lambda x: x if x in vocabulary else "*unknown*", test)

print ("... train")
from nltk.model import NgramModel
from nltk.probability import LidstoneProbDist

estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2) 
lm = NgramModel(5, train, estimator=estimator)

print ("len(corpus) = %s, len(vocabulary) = %s, len(train) = %s, len(test) = %s" % ( len(corpus), len(vocabulary), len(train), len(test) ))
print ("perplexity(test) =", lm.perplexity(test))
