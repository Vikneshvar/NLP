import nltk
#nltk.download()
from nltk.util import ngrams
import string
from politicsApp.models import Articles, Ngram, ArticleNgram, Interaction
import os, re, string
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from django.db import connection
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from nltk.stem.porter import PorterStemmer
from unidecode import unidecode
import tensorflow as tf
from memory_profiler import profile
import pickle
import scipy.sparse as sp


def run():
	rawText = ''
	sourceName = ''
	sourcePath = "/Users/vik_work/Desktop/Workspace/NLP_Political/Data/NLP_Test"
	for subdir, dirs, files in os.walk(sourcePath):
		for categoryName in dirs:
			sourcedir = os.path.join(sourcePath,categoryName)
#			print('sourcedir',sourcedir)
			for textFile in os.listdir(sourcedir):
				if textFile.endswith(".txt"):			
					textFile = textFile.strip()
					fileName = ''
					for each in textFile:
						if each.isalpha() == True or each==' ' or each=='.':
							fileName+=each
					sourceName = categoryName
					print("Source:",sourceName)
					print("Filename: ",fileName)
					f = open(os.path.join(sourcedir,textFile),"r",encoding='utf-8', errors='ignore')
					rawText = f.read()
					break

	#load pickle file back to memory
	path = "/Users/vik_work/Desktop/Workspace/NLP_Political/politicsNLP/polictsproject/Files"
	filename = 'tfidf_idf' + '.pk'
	filePath = os.path.join(path, filename)
	with open(filePath, 'rb') as fi:
		tfidf_idf = pickle.load(fi)
	print('tfidf_idf: ', tfidf_idf)

	filename = 'tfidf_vocabulary' + '.pk'
	filePath = os.path.join(path, filename)
	with open(filePath, 'rb') as fi:
		tfidf_vocabulary = pickle.load(fi)
	print('tfidf_vocabulary: ', tfidf_vocabulary)

	# subclass TfidfVectorizer
	class MyVectorizer(TfidfVectorizer):
		# plug our pre-computed IDFs
		TfidfVectorizer.idf_ = tfidf_idf

	# instantiate vectorizer
	vectorizer = MyVectorizer(lowercase = True,
							  min_df = 1,
							  norm = 'l2',
							  strip_accents='ascii',
							  analyzer='word',
							  ngram_range=(1,7),
							  smooth_idf = True)

	# plug _tfidf._idf_diag
	vectorizer._tfidf._idf_diag = sp.spdiags(tfidf_idf,
											 diags = 0,
											 m = len(tfidf_idf),
											 n = len(tfidf_idf))

	vectorizer.vocabulary_ = tfidf_vocabulary

	print('vectorizer',vectorizer)

	# Raw text to unidecoded text
	unidecode_text = unidecode(rawText)
#	print('unidecode_text: ', unidecode_text)
	dict_output = textClean(unidecode_text)
	processedText=dict_output.get('string')
	print("processedText: ----- ", processedText)
	text = processedText.replace('.',' ')

	smatrix = vectorizer.transform([text])
	print('smatrix shape',smatrix.shape)
	print('smatrix',smatrix)
	print('smatrix data',smatrix.data)	
	print('smatrix column_index',smatrix.indices)	
	print('smatrix row_pointers',smatrix.indptr)
	
	feature_names = vectorizer.get_feature_names()
#	print('feature_names',feature_names)

	dmatrix = smatrix.todense()
	print('dmatrix',dmatrix)
	print('dmatrix.shape',dmatrix.shape)	
	
	# Set Source Column to 1 if Fox and 0 if MSNBC
	# set_value(row,column,value)
	if sourceName == 'Fox':
		print('Fox Document')
		y_M = [[1,0]]
	else:
		print('MSNBC Document')
		y_M = [[0,1]]

	# Each layer hidden nodes
	nodes_1st=dmatrix.size
	nodes_2nd=500
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

	# Final output
	y = tf.nn.softmax(tf.matmul(dropout,W2)+B2)

	# Cross entropy function 
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
	
	# Using Grdient Descent
	train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

	# comparison of y and y_
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	
	# Accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	# Get saved model's weights and evaluate
	# Create save variable to save and restore all the variables.
	saver = tf.train.Saver()

	# Use session to initialize all variables
	with tf.Session() as sess:
		
		# Restore variables from disk
		saver.restore(sess, "/tmp/model.ckpt")
		print("Model restored.")

		# Get the values of weight variables
		W1_predict = W1.eval()
		B1_predict = B1.eval()
		W2_predict = W2.eval()
		B2_predict = B2.eval()
		
		y_predicted = y.eval(feed_dict={x:dmatrix,W1:W1_predict,W2:W2_predict,B1:B1_predict,
									B2:B2_predict,keep_prob:1.0})

		print('y_predicted',y_predicted)

		print('correct_prediction',correct_prediction.eval(feed_dict={y:y_predicted,y_:y_M}))


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

	finalString = finalString.replace("'","")
#	print('processedText -------',finalString)
	
	# Count number of words
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(finalString)

	dict_output={}
	dict_output['string']=finalString
	dict_output['count']=len(tokens)

	return dict_output

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
