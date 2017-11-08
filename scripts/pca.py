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
from sklearn.decomposition import PCA
from scipy.signal import correlate2d

def run():

	try:
		engine = create_engine("mysql+mysqldb://root:vik123@localhost:3306/nlp2")
		connection = engine.connect()
#		sql_query="""select * from nlp2.politicsApp_nndata_ngram_size_1"""
		sql_query="""select * from nlp2.politicsApp_nndata"""
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
	nlp_Df=nlp_df_t.transpose()
	nlp_Df=nlp_Df.iloc[:,0:len(nlp_Df.columns)-2]

	print('nlp_Df ****** \n',nlp_Df)
	print('Shape of nlp_df_t after transpose',nlp_Df.shape)
	ngram_list = []

#	for i in range(1,2):
#	print('Ngram size',i)
	articles = Articles.objects.all()
#		ngrams = Ngram.objects.filter(NgramSize=i)
	ngrams = Ngram.objects.all()
#       print('len(ngrams)',len(ngrams))
#       print('len(interactions)',len(interactions))
	for ngram in ngrams:
		ngram_ = ngram.Ngram
		ngramId = ngram.NgramId

#           ngramId_list.append(ngramId)
		ngram_list.append(ngram_)

	nlp_df = pd.DataFrame(data=nlp_Df.values, index=np.arange(1,len(articles)+1), columns=ngram_list)



	print(nlp_df)
	print('Shape of nlp_df \n',nlp_df.shape)

	X = nlp_df.iloc[:,0:]
	print('X',X)
		
	# Already in standardized form 
	X_std = X 
#	print('X_std',X_std)
	
	# PCA using sci-kit learn
	pca = PCA(n_components = 200)
	pca.fit(X_std)
	print('Explained variance',pca.explained_variance_)
	print('Explained variance ratio',pca.explained_variance_ratio_)
	print('Cumulative Explained variance ratio',pca.explained_variance_ratio_.cumsum())


	X_reduced = pca.fit_transform(X_std)
	print('X_reduced',X_reduced)
	print('X_reduced.shape',X_reduced.shape)
	
	components = pca.components_
	print('pca.components_ \n',components)
	print('pca.components_.shape',components.shape)
	
	nlp_df_pca_reduced = pd.DataFrame(data=X_reduced, index=['Article '+str(i) for i in range(200)], columns=['PC '+str(i) for i in range(200)])
	print('nlp_df_pca_reduced \n',nlp_df_pca_reduced)

	# Dot product of input variables and components
	# Failed - Process Killed
#	Y = X_std.T.dot(pca.components_)
#	print(Y)


	
	# PCA using elaborated eigen value, eigen vecotr method
	# Correlation of variables and components 
#	cor_mat1 = np.corrcoef(X_std.T)
#	eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
#	print('Eigenvectors \n%s' %eig_vecs)
#	print('\nEigenvalues \n%s' %eig_vals)


	# Database engine and save data frame to Mysql db
	try:
		engine = create_engine("mysql+mysqldb://root:vik123@localhost:3306/nlp2")
		connection = engine.connect()
		table_name='politicsApp_nndata_pca_reduced'
		print('table_name',table_name)
		nlp_df_pca_reduced.to_sql(con=engine, name=table_name,if_exists='replace',index=False)
		connection.close()
		engine.dispose()
	except MySQLdb.IntegrityError as e:
		print('Exception occured', e)
	finally:
		print('Successfully table created')



