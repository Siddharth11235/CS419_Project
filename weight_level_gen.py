from random import randint
from pickle import load
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
 
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	counter = 0
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		
		sizer = randint(9, 13)
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input

		in_text += ' ' + out_word
		result.append(out_word)
		if counter >= sizer:
			result.append('\n')
			counter = 0
		counter+=1

	return ' '.join(result)
 
tokenizer = load(open('tokenizer.pkl', 'rb'))

# load cleaned text sequences
in_filename = 'datasets/WW_Dataset_seq.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
original_sequences = tokenizer.texts_to_sequences(lines)
aligned_sequneces = []
for sequence in original_sequences:
	aligned_sequence = np.zeros(11, dtype=np.int64)
	aligned_sequence[:len(sequence)] = np.array(sequence, dtype=np.int64)
	aligned_sequneces.append(aligned_sequence)

sequences = np.array(aligned_sequneces)
seq_length = len(lines[0].split()) - 1
 
# load the model
model = load_model('Word_weights/lstm_model.h5')
 
# load the tokenizer
 
# select a seed text
seed_text = lines[randint(0,len(lines))]
print(seed_text)
 
# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 100)
print(generated)