from politicsApp.models import Ngram, Articles, ArticleNgram
import MySQLdb,re,sys
from django.db import connection
from multiprocessing import Process
from math import log

def run():

	articles = Articles.objects.all()
	ngrams = Ngram.objects.all()
	total_docs = len(articles)
	list_articleNgram = []

	for ngram in ngrams:
		gram = ngram.Ngram
		ngramId = ngram.NgramId
		ngramIdf = ngram.IDF
#		print("gram: ", gram)
		print("Ngram ID: ", ngramId)

		# Variable to update if the ngram is present in this document	
		for article in articles:
#			phrasedText = article.PhrasedText_2
			processedText = article.ProcessedText
			ngramSize = ngram.NgramSize
			articleId = article.ArticleId
#			print("Article ID: ", articleId)

			my_regex = r"\b" + gram + r"\b"
#			print("my_regex: ",my_regex)
			matches = re.findall(my_regex,processedText)
#			print("match count: ", len(matches))
				
			dict_articleNgram = {}
			dict_articleNgram["NgramId"] = ngramId
			dict_articleNgram["ArticleId"] = articleId
			dict_articleNgram["NgramSize"] = ngramSize
			dict_articleNgram["TF"] = len(matches)
			dict_articleNgram["TFIDF"] = len(matches)*ngramIdf

			list_articleNgram.append(dict_articleNgram)

#	print("list_articleNgram: ",list_articleNgram)
	print("len(list_articleNgram): ",len(list_articleNgram))

	try:
		cur = connection.cursor()
		stmt= """INSERT INTO nlp2.politicsApp_articlengram (NgramId_id, ArticleId_id,NgramSize_id,TF,TFIDF) 
					values (%(NgramId)s,%(ArticleId)s,%(NgramSize)s,%(TF)s,%(TFIDF)s)"""
		cur.executemany(stmt, list_articleNgram)
		connection.commit()
		print("affected rows {}".format(cur.rowcount))
	except MySQLdb.IntegrityError:
		print("failed to insert values")
	finally:
		cur.close()



	







