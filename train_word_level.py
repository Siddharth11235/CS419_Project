from numpy import array
import numpy as np
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Embedding



# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load
in_filename = 'datasets/WW_Dataset_seq.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
original_sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1

aligned_sequneces = []
for sequence in original_sequences:
	aligned_sequence = np.zeros(9, dtype=np.int64)
	aligned_sequence[:len(sequence)] = np.array(sequence, dtype=np.int64)
	aligned_sequneces.append(aligned_sequence)

sequences = np.array(aligned_sequneces)

# separate into input and output
sequences = array(sequences)
y = sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
X = sequences[:,:-1]
seq_length = X.shape[1]
X_train = np.reshape(X, (X.shape[0], 1, X.shape[1]))


# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(GRU(256, return_sequences=True))
model.add(GRU(256))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=100)
# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))