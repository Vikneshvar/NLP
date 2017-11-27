import nltk
#nltk.download()
from nltk.util import ngrams
import string
from politicsApp.models import Articles, Ngram, ArticleNgram, Interaction
import os, re, string
import numpy as np
import pandas as pd
from django.db import connection
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from nltk.stem.porter import PorterStemmer
from unidecode import unidecode
import tensorflow as tf
from memory_profiler import profile
import pickle
import shutil

#@profile
def run():
	articles = Articles.objects.filter(Type='Training')

	def stem_tokens(tokens, stemmer):
		stemmed = []
		for item in tokens:
			stemmed.append(stemmer.stem(item))
		return stemmed

	def tokenize(text):
		tokens = nltk.word_tokenize(text)
		stems = stem_tokens(tokens, stemmer)
		return tokens		

	token_dict = {}
	stemmer = PorterStemmer()
	
	for article in articles:
		print('Source', article.Source)
		text = article.ProcessedText
#		text = article.PhrasedText_2
#		text = article.PhrasedText_2
		text = text.replace('.',' ')
#		print('processedText -----', processedText)
		fileName = article.FileName
		token_dict[fileName] = text

#	print('token_dict',token_dict)
	tfidf = TfidfVectorizer(tokenizer=tokenize, strip_accents='ascii',analyzer='word',ngram_range=(1,7))
	smatrix = tfidf.fit_transform(token_dict.values())
	# print('tfidf',tfidf)
	# print('tfidf.vocabulary_',tfidf.vocabulary_)
	# print('tfidf.idf_',tfidf.idf_)
	
	# Create a pickle file
	path = "/Users/vik_work/Desktop/Workspace/NLP_Political/politicsNLP/polictsproject/Files"
	filename = 'tfidf_vocabulary' + '.pk'
	filePath = os.path.join(path, filename)

	with open(filePath,'wb') as fi:
		pickle.dump(tfidf.vocabulary_,fi)

	filename = 'tfidf_idf' + '.pk'
	filePath = os.path.join(path, filename)

	with open(filePath,'wb') as fi:
		pickle.dump(tfidf.idf_,fi)

	# print('smatrix shape',smatrix.shape)
	# print('smatrix',smatrix)
	# print('smatrix data',smatrix.data)	
	# print('smatrix column_index',smatrix.indices)	
	# print('smatrix row_pointers',smatrix.indptr)

	feature_names = tfidf.get_feature_names()
#	print('feature_names',feature_names)
	dmatrix = smatrix.todense()
	print('dmatrix',dmatrix)
	print('dmatrix.shape',dmatrix.shape)	

	nlp_df = pd.DataFrame(data=dmatrix, index=np.arange(1,201), columns=feature_names)
	nlp_df['Source'] = 0
	# 1 = Fox, 0 = MSNBC. First 100 Fox, next 100 MSNBC	
	nlp_df.iloc[0:100,-1]=1
	print(nlp_df)

	# Get column as array for encoding process
	float_encoded = np.array(nlp_df['Source'])
	onehot_encoder = OneHotEncoder(sparse=False)
	float_encoded = float_encoded.reshape(len(float_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(float_encoded)
	
	# Delete Source column as it is going to be encoded
	del nlp_df['Source']

	# Create 2 new encoded columns
	nlp_df['Fox']=onehot_encoded[:,[1]]
	nlp_df['MSNBC']=onehot_encoded[:,[0]]
	print('nlp_df  ---------- \n',nlp_df.iloc[98:102,:])
#	print('lennnnnnn',len(nlp_df))


	# Split data into train and test - 70:30
	msk = np.random.rand(len(nlp_df)) < 0.7
	train_df = nlp_df[msk]
#	print(train_df)
	test_df = nlp_df[~msk]
#	print(test_df)


	# Each layer hidden nodes
	nodes_1st=int(len(nlp_df.columns)-2)
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

	# Create save variable to save and restore all the variables.
	saver = tf.train.Saver()

	size = 0
	for variable in tf.all_variables():
		size += int(np.prod(variable.get_shape()))

	print(size)

	# Use session to initialize all variables
	with tf.Session() as sess:
		print('Hi 4444444')
		sess.run(tf.global_variables_initializer())	
		print('Hi 3333333')
		
		# Run the algorithm - On each iteration, batch of 25 articles goes in network 
		# Feed forward and back	propagaion happens 	
		for p in range(50):
#			print(' Iteration ------------------------- ',p)
			# Using training data
			x_input,y_output = batch(train_df,1)
#			print('x_input',x_input.shape)
			train_accuracy=accuracy.eval(feed_dict={x:x_input,y_:y_output,keep_prob:1.0})
			print('Step %d Training accuracy %g' %(p,train_accuracy))
			# Backpropagation
			train_step.run(feed_dict={x:x_input,y_:y_output,keep_prob:0.5})

		# Using test data
		x_input_test,y_output_test = batch(test_df,0)		
		test_accuracy = accuracy.eval(feed_dict={x:x_input,y_:y_output,keep_prob:1.0})
		print('Test Accuracy ', test_accuracy)

		# Save the variables to the disk
		save_path = saver.save(sess,"/tmp/model.ckpt")
#		print('Model saved in file: %s' % save_path)

def batch(df, trainFlag):
	if trainFlag == 1:
#		print('Training -------------- 1')
		new_batch = df.sample(n=25,replace=False)
		x_input = np.array(new_batch.iloc[:,0:len(new_batch.columns)-2])
		y_output = np.array(new_batch.iloc[:,len(new_batch.columns)-2:])
	else:
#		print('Test -------------- 0')			
		x_input = np.array(df.iloc[:,0:len(df.columns)-2])
		y_output = np.array(df.iloc[:,len(df.columns)-2:])
	return x_input,y_output

	