import tensorflow as tf
import sys,os
from sqlalchemy import create_engine
from django.db import connection
import pandas as pd
import numpy as np

def run():

	try:
		engine = create_engine("mysql+mysqldb://root:vik123@localhost:3306/nlp2")
		connection = engine.connect()
		sql_query="""select * from nlp2.politicsApp_nndata_latest"""
		nlp_df_t = pd.read_sql_query(con=engine,sql=sql_query)
		connection.close()
		engine.dispose()
	except:
		e = sys.exc_info()[0]
		print('Exception occured', e)
	finally:
		print('Successfully read the table')

#	print('nlp_df_t \n',nlp_df_t)
#	print('Shape of nlp_df_t got from db',nlp_df_t.shape)

	# Transpose to original dataframe
	nlp_df=nlp_df_t.transpose()

#	print('nlp_df - original ****** \n',nlp_df)
	print('Shape of nlp_df_t after transpose',nlp_df.shape)


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

	# Final output
	y = tf.nn.softmax(tf.matmul(dropout,W2)+B2)

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
		y.eval(feed_dict={x:x_M,y_:y_M,W1:W1_predict,W2:W2_predict,B1:B1_predict,
									B2:B2_predict,keep_prob:1.0})
		
		print('y',y.eval(feed_dict={x:x_M,y_:y_M,W1:W1_predict,W2:W2_predict,B1:B1_predict,
									B2:B2_predict,keep_prob:1.0}).shape)		


""" # Check weights of the ngram words

#	w = tf.argmax(weight,1)
#	print('-----Weight',weight)
#	print('-----Weight Shape',weight.shape)

#		
		ngrams = Ngram.objects.filter(NgramSize=1)
		ngram_list = []
		for ngram in ngrams:
			ngram_ = ngram.Ngram
			ngram_list.append(ngram_)

#		ngram_list.reshape()

		nlp_df = pd.DataFrame(data=w, index=ngram_list, columns=['Weights'])

		print('Max weight word',nlp_df['Weights'].max())
		print('Min weight word',nlp_df['Weights'].min())

		print('Above average weight word',nlp_df['Weights']>nlp_df['Weights'].mean().value)

			
			print('x shape',(x.eval(feed_dict={x:x_input})).shape)
			print('y_',y_.eval(feed_dict={y_:y_output}))
			print('W1',W1.eval())
			print('B1',B1.eval())
			print('y1',y1.eval(feed_dict={x:x_input}))
			print('W2',W2.eval())
			print('B2',B2.eval())
			print('correct_prediction',correct_prediction.eval(feed_dict={x:x_input,y_:y_output,keep_prob:1.0}))

"""	

	