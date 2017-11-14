from politicsApp.models import Articles, Ngram
import os, re, string
import pickle
from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer
import pandas as pd, tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder



def run():
	rawText = ''
	sourceName = ''
	sourcePath = "/Users/Vik/Desktop/Project/NLP_Test/"
	for subdir, dirs, files in os.walk(sourcePath):
		for each in dirs:
			sourceName = each
			print("Source:",sourceName)
			sourcedir = os.path.join(sourcePath,sourceName)
#			print('sourcedir',sourcedir)
			for textFile in os.listdir(sourcedir):
				if textFile.endswith(".txt"):			
					textFile = textFile.strip()
					fileName = ''
					for each in textFile:
						if each.isalpha() == True or each==' ' or each=='.':
							fileName+=each

					print("Filename: ",fileName)
					f = open(os.path.join(sourcedir,textFile),"r",encoding='utf-8', errors='ignore')
					rawText = f.read()
					break
#	print('rawText: ', rawText)

	# Import remove_list, bigram from database that is saved during training
	

	#load pickle file back to memory
	path = "/Users/Vik/Desktop/soper/project/politicsNLP/Files"
	filename = 'remove_list' + '.pk'
	filePath = os.path.join(path, filename)
	with open(filePath, 'rb') as fi:
		remove_list = pickle.load(fi)
#	print('remove_list: ', remove_list)

	filename = 'bigram' + '.pk'
	filePath = os.path.join(path, filename)
	with open(filePath, 'rb') as fi:
		bigram = pickle.load(fi)
#	print('bigram.vocab: ',bigram.vocab)

	# Raw text to unidecoded text
	unidecode_text = unidecode(rawText)
#	print('unidecode_text: ', unidecode_text)
	
	# unidecoded to processed text
	dict_output = textClean(unidecode_text)
	processedText=dict_output.get('string')
#	print("processedText: ----- ", processedText)

	# processed to sampled text
	# Supply remove list from app memory or database
	dict_output	= sampleText(processedText,remove_list)
	sampledText=dict_output.get('string')
#	print("sampledText: ----- ", sampledText)

	# sampled to phrased text
	# supply bigram list from app memory or database
	dict_output = Phrase(sampledText,bigram)
	phrasedText=dict_output.get('string')
#	print("phrasedText: ----- ", phrasedText)
	wordCount=dict_output.get('count')
#	print("wordCount: ---- ", wordCount)

	# Get ngrams from app memory or database
	# Get size 1 ngram - use filter to get that
	list_articleNgram =[]
	ngrams = Ngram.objects.filter(NgramSize=1)
	for ngram in ngrams:
		ngramId = ngram.NgramId
		gram = ngram.Ngram
#		print("gram: ", gram)
		ngramSize = ngram.NgramSize

		my_regex = r"\b" + gram + r"\b"
#		print("my_regex: ",my_regex)
		matches = re.findall(my_regex,phrasedText)
#		print("match count: ", len(matches))

		dict_articleNgram = {}
		dict_articleNgram["NgramId"] = ngramId
		dict_articleNgram["Frequency"] = len(matches)
		dict_articleNgram["StdFrequency"] = len(matches)/wordCount

		list_articleNgram.append(dict_articleNgram)

#	print("list_articleNgram: ",list_articleNgram)
	print("len(list_articleNgram): ",len(list_articleNgram))

	ngramId_list = []
	for item in list_articleNgram:
		ngramId = item.get('NgramId')
		ngramId_list.append(ngramId)

	nlp_df = pd.DataFrame(data=0.0, index=np.arange(1), columns=ngramId_list)

	# Set Source Column to 1 if Fox and 0 if MSNBC
	# set_value(row,column,value)
#	print('sourceName', sourceName)
	if sourceName == 'Fox':
		nlp_df['Fox']=1
		nlp_df['MSNBC']=0
	else:
		nlp_df['Fox']=0
		nlp_df['MSNBC']=1

	print('nlp_df  ---------- \n',nlp_df)
	print('lennnnnnn',len(nlp_df))

	# Set StdFrequency Values for corresponding ngram Id of each article
	for articleNgram in list_articleNgram:
		ngramId = articleNgram.get('NgramId')
		stdFrequency = articleNgram.get('StdFrequency')
#		print('ngramId',ngramId)
#		print('stdFrequency',stdFrequency)
		nlp_df.set_value(0,ngramId,stdFrequency)
			
	print(nlp_df)
	print('Shape of nlp_df \n',nlp_df.shape)

	x_input = np.array(nlp_df.iloc[:,0:len(nlp_df.columns)-2])
	y_output = np.array(nlp_df.iloc[:,len(nlp_df.columns)-2:])
	x_M = x_input[0,:]
	x_M = np.matrix(x_M).T
	x_M = np.matrix(x_M).T
	y_M = y_output[0,:]
	y_M = np.matrix(y_M).T
	y_M = np.matrix(y_M).T

	# Replicate same Tensorflow architecture
	# Each layer hidden nodes
	nodes_1st=int(len(nlp_df.columns)-2)
	nodes_2nd=int(len(nlp_df.columns)/2)
#	nodes_3rd=int(len(nlp_df.columns)/1000)
	nodes_output=2

	# Neural Network Design
	# Input nodes - a node for each ngram, rows: None means any number, columns: number of ngrams
	x = tf.placeholder(tf.float32,shape=[None,nodes_1st])

	# Variable to hold the actual output
	y_ = tf.placeholder(tf.float32,shape=[None,2])

	# Drop out probability variable
	keep_prob = tf.placeholder(tf.float32)

	# Weights and bias Variables for 1st layer
	W1 = tf.Variable(tf.truncated_normal(shape=[nodes_1st,nodes_2nd],stddev=0.1))
	B1 = tf.Variable(tf.constant(0.1,shape=[nodes_2nd]))

	# Output of first layer
	y1 = tf.matmul(x,W1)+B1

	# Weights and bias Variables for 2st layer
	W2 = tf.Variable(tf.truncated_normal(shape=[nodes_2nd,nodes_output],stddev=0.1))
	B2 = tf.Variable(tf.constant(0.1,shape=[nodes_output]))

	# Dropout 
	dropout = tf.nn.dropout(y1, keep_prob)

	# Output Probability
	y_prob = tf.matmul(dropout,W2)+B2
	
	# Final output
	y = tf.nn.softmax(y_prob)

	# Get saved model's weights and evaluate
	# Create save variable to save and restore all the variables.
	saver = tf.train.Saver()

	with tf.Session() as sess:

		# Restore variables from disk
		saver.restore(sess, "/tmp/model.ckpt")
		print("Model restored.")

		# Get the values of weight variables
		W1_predict = W1.eval()
		B1_predict = B1.eval()
		W2_predict = W2.eval()
		B2_predict = B2.eval()
		print("W1 : ",W1_predict)
		print("B1 : %s" % B1_predict)
		print("W2 : %s" % W2_predict)
		print("B2 : %s" % B2_predict)
		y_prob.eval(feed_dict={x:x_M,y_:y_M,W1:W1_predict,W2:W2_predict,B1:B1_predict,
									B2:B2_predict,keep_prob:1.0})
		
		print('y_prob',y_prob.eval(feed_dict={x:x_M,y_:y_M,W1:W1_predict,W2:W2_predict,B1:B1_predict,
									B2:B2_predict,keep_prob:1.0}).shape)

		print('y',y.eval(feed_dict={x:x_M,y_:y_M,W1:W1_predict,W2:W2_predict,B1:B1_predict,
									B2:B2_predict,keep_prob:1.0}).shape)		





# Functions
def sampleText(processedText,remove_list):

	sampledText = processedText
	finalString = ''
	for each_ngram in remove_list:
#		print('each_ngram',each_ngram)
		if len(each_ngram) > 0:
			my_regex = r"\b" + each_ngram + r"\b"
#			print("my_regex: ",my_regex)
			sampledText = re.sub(my_regex,'',sampledText)
#			print('sampledText',sampledText)
			# Remove trailing and leading spaces in sentence
			finalString = ''.join([doc.strip()+'.' for doc in sampledText.split('.')])

			# Remove multiple periods together
			finalString = finalString.replace('..','.')

	# Count number of words
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(finalString)

	dict_output={}
	dict_output['string']=finalString
	dict_output['count']=len(tokens)

	return dict_output

def textClean(rawText):
	newString = re.sub('[%s]' % string.digits, '', rawText)
	newString = newString.replace('\n','')
	newString = newString.replace(',','.')
	newString = newString.replace('-',' ')
	newString = newString.replace(';','.')
	newString = newString.replace('[','')
	newString = newString.replace(']','')
	newString = newString.replace('(','')
	newString = newString.replace(')','')
	newString = newString.replace('?',' ')
	newString = newString.replace('!',' ')
	newString = newString.replace('%',' ')
	newString = newString.replace('@','')
	newString = newString.replace(':',' ')
	newString = newString.replace('/',' ')
	newString = newString.replace('$',' ')
	newString = newString.replace('+',' ')
	newString = newString.replace('*',' ')
	newString = newString.replace('&',' ')
	newString = newString.replace('#',' ')

	# Convert single quote word to double quote
	newString = re.sub(r'\'\w+\'',toDoubleQuote,newString)
	
	# Double and single quotes
	newString = newString.replace('"','.')

	# Remove apostrophe and characters after that
#	newString = newString.replace("'",'')
	newString = re.sub(r'(?:\'\w+)','',newString)

	# Convert without period acronyms by with period acronyms
	newString = re.sub(r'(?<![A-Z])[A-Z]{2,7}(?![A-Z])',toPeriodsInbetween,newString)

	# Converting all character to lower
	newString = newString.lower()
#	print(" String: With Abbrevation Period \n",newString)
	print('\n')
	
	# Convert 2 to 10 letter acronym with period inbetween to upper and remove period
	newString = re.sub(r'(?:([a-z]\.)+([a-z]\.)+)',toUpperRemovePeriod,newString)

#	print("newString: Without period \n",newString)
	print('\n')

	# Remove trailing and leading spaces in sentence
	finalString = ''.join([doc.strip()+'.' for doc in newString.split('.')])
#	print("fString: Removing trailing and leading spaces and to lower \n",finalString)
	print('\n')

	# Remove multiple periods together
	finalString = finalString.replace('..','.')
#	print('processedText -------',finalString)

	dict_output={}
	dict_output['string']=finalString

	return dict_output

def addVocab(sampledText,bigram):
	# Using Genism
	# Make sentence stream
	document = sampledText.split('.')
	sentence_stream = [doc.split(' ') for doc in document]
#	print('sentence_stream -------',sentence_stream)

	# Add vocab to bigram # Training
	bigram.add_vocab(sentence_stream)


def Phrase(sampledText,bigram):
	sentence_list = sampledText.split('.')
	finalString = ''
	for each_sentence in sentence_list:
		input_string_in_list = each_sentence.split(' ')
#		print('input_string_in_list',input_string_in_list)
		# Send new string to the trained detector
		out = ' '.join(each for each in bigram[input_string_in_list])+'.'
#		print('out',out)
		finalString+=out
#	print('finalString',finalString)

	# Count number of words
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(finalString)

	dict_output={}
	dict_output['string']=finalString
	dict_output['count']=len(tokens)

	return dict_output

# Create ngrams of one words only - coz we are removing only one words from the text
def word_grams(words, min=1, max=7):
	s = []
	for n in range(min, max):
		for ngram in ngrams(words, n):
			p =' '.join(str(i) for i in ngram)
#			print('\n p ',p)
			s.append(p)
	return s

def toUpperRemovePeriod(match):
	groups = match.group()
#	print('groups---',groups)
	acronym = groups.replace('.','')
	acronym = acronym.upper()
#	print('acronym---',acronym)
	return acronym

def	toPeriodsInbetween(match):
	match = match.group()
	withPeriods = '.'.join(ch for ch in match)
	withPeriods+='.'
#	print('withPeriods',withPeriods)
	return withPeriods

def toLower(match):
	match = match.group()
	lowecase = match.lower()
	return lowecase	

def toDoubleQuote(match):
	match = match.group()
	double = match.replace("'",'"')
	return double	
