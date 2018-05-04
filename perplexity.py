import nltk
from nltk import word_tokenize
from nltk.tokenize import line_tokenize
from nltk.corpus import gutenberg
from nltk.model import build_vocabulary
from nltk.model import count_ngrams
from nltk.model import MLENgramModel
from nltk.model import LidstoneNgramModel
# load doc into memory

raw = open('datasets/WW_Dataset.txt', 'r').read()
print(raw[:75])

tokens = word_tokenize(raw)
print(len(tokens))
lines = line_tokenize(raw)
test_lines = lines[3:5]
test_words = [w for s in test_lines for w in s]

print(test_words[:5])
corpus = [w.lower() for w in tokens]
text = nltk.Text(tokens)
words = [w.lower() for w in tokens]
print(words[:10])
vocab = sorted(set(words))
print(len(vocab))
spl = int(95*len(corpus)/100)
train = text[:spl]
test = text[spl:]
vocab = build_vocabulary(2, words)
bigram_counts = count_ngrams(2, vocab, text)
bigram_model = LidstoneNgramModel(3,bigram_counts)
#ex_score = bigram_model.score("yawned", ["he"])

print(bigram_model.perplexity("stopped and took the penny up and when the cripple nearer drew quoth andrew under halfacrown what a man finds is all his own and so my friend goodday to show and proud still in that dear vagrants looked up by reason bound as if in habit sympathy their spirit spare more oft love for thee that say stand in all works or their congenial powers that fear as pleasures round the stationary blasts of their sorrow heart those higher years that did i meditate to me and evil sweet respect to paint that musings of vice as high and words of excellence had with beatitude that pure gains and sedate"))