from politicsApp.models import Articles, Ngram, ArticleNgram, Interaction
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from django.db import connection
from django.conf import settings
import MySQLdb
from sqlalchemy.dialects import mysql 
from sqlalchemy.types import VARCHAR
from sklearn.preprocessing import OneHotEncoder

def run():

#	for i in range(5,7):
#	print('Ngram size',i)
	interactions = Interaction.objects.all()
#		interactions = Interaction.objects.filter(NgramSize=i)
	articles = Articles.objects.filter(Type='Training')
	ngrams = Ngram.objects.all()
#		ngrams = Ngram.objects.filter(NgramSize=i)

	print('len(ngrams)',len(ngrams))
	print('len(interactions)',len(interactions))

	ngram_list = []
	ngramId_list = []
	for ngram in ngrams:
		ngram_ = ngram.Ngram
		ngramId = ngram.NgramId

		ngramId_list.append(ngramId)
#			ngram_list.append(ngram_)

#		ngram_list.append('Source')
	ngramId_list.append('Source')
	print('len(articles) ',len(articles))
	nlp_df = pd.DataFrame(data=0.0, index=np.arange(1,len(articles)+1), columns=ngramId_list)

	
	# Set Source Column to 1 if Fox and 0 if MSNBC
	# set_value(row,column,value)
	for article in articles:
		articleId = article.ArticleId
		source = article.Source
#		print('source', source)
		if source == 'Fox':
			source = 0
		else:
			source = 1
#		print('articleId',articleId)
#		print('source',source)
		nlp_df.set_value(articleId,'Source',source)

	# Get column as array for encoding process
	float_encoded = np.array(nlp_df['Source'])
	print('float_encoded', float_encoded)
	print('len float_encoded', len(float_encoded))
	onehot_encoder = OneHotEncoder(sparse=False)
	float_encoded = float_encoded.reshape(len(float_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(float_encoded)
	print(onehot_encoded)
	# Delete Source column as it is going to be encoded
	del nlp_df['Source']
	# Create 2 new encoded columns
	nlp_df['Fox']=onehot_encoded[:,[0]]
	nlp_df['MSNBC']=onehot_encoded[:,[1]]
	print('nlp_df  ---------- \n',nlp_df)
	print('lennnnnnn',len(nlp_df))

	# --------- Try multiprocessing for this
	# Set StdFrequency Values for corresponding ngram Id of each article
	for interaction in interactions:
		ngramId = interaction.NgramId_id
		articleId = interaction.ArticleId_id
		stdFrequency = interaction.StdFrequency
#		print('articleId',articleId)
#		print('ngramId',ngramId)
#		print('stdFrequency',stdFrequency)
		nlp_df.set_value(articleId,ngramId,stdFrequency)
			
	print(nlp_df)
	print('Shape of nlp_df before transpose \n',nlp_df.shape)

	# Saving dataframe in database doesnt work because
	# max column a MySQL table can have is 4096
	# So, do transpose and save in db
	nlp_df_t=nlp_df.transpose()
	print('nlp_df_t ***************',nlp_df_t)
	print('Shape of nlp_df_t after transpose',nlp_df_t.shape)


	# Database engine and save data frame to Mysql db
	try:
		engine = create_engine("mysql+mysqldb://root:vik123@localhost:3306/nlp2")
		connection = engine.connect()
#			table_name='politicsApp_nndata_ngram_size_'+str(i)
		table_name='politicsApp_nndata_latest'
		print('table_name',table_name)
		nlp_df_t.to_sql(con=engine, name=table_name,if_exists='replace',index=False)
		connection.close()
		engine.dispose()
	except MySQLdb.IntegrityError as e:
		print('Exception occured', e)
	finally:
		print('Successfully table created')




