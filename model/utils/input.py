# model input functions

from config import model_parameters as mp
from utils.processing import WordsToSPAVocab

# will return a context vector of tokens dependent on time
def context_in(t, training_set=[], testing_set=[]):
	# training time length
	training_time = len(training_set)*mp.tr_impression
	# testing time length
	# testing_time = (len(testing_set)-1)*mp.tr_impression

	if t <= training_time:
		impression = mp.tr_impression # how long the model sees each word for (impression time)
		pair = training_set[int(t // impression)]
		left = WordsToSPAVocab(pair[0])
		return " + ".join(left) # returns training context
	else:
		t_test = t - training_time
		impression = mp.tr_impression # how long the model sees each word for (impression time)
		pair = testing_set[int(t_test // impression)]
		# left = WordsToSPAVocab([unknown_token if x not in vocab else x for x in pair[0]]) # strict case
		left = WordsToSPAVocab(pair[0])
		return " + ".join(left) # returns training context

# will return the desired predicted token dependent on time
def find_target(t, training_set=[], testing_set=[]):
	# training time length
	training_time = len(training_set)*mp.tr_impression
	# testing time length
	# testing_time = (len(testing_set)-1)*mp.tr_impression

	if t <= training_time:
		impression = mp.tr_impression # how long the model sees each word for (impression time)
		pair = training_set[int(t // impression)]
		return WordsToSPAVocab(pair[1])[0] # should find targets in parallel to context
	else:
		t_test = t - training_time
		impression = mp.tr_impression # how long the model sees each word for (impression time)
		pair = testing_set[int(t_test // impression)]
		# should find targets in parallel to context (strict case)
		# return WordsToSPAVocab(unknown_token if x not in vocab else x for x in pair[1])[0] 
		return WordsToSPAVocab(pair[1])[0]

def is_recall(t, training_time):
	return t > training_time

def is_one(input):
	return input == 1