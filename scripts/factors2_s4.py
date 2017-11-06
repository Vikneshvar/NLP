from politicsApp.models import Ngram, Articles, ArticleNgram
import MySQLdb,re
from django.db import connection
from multiprocessing import Process

def run():

	articles = Articles.objects.all()
	ngrams = Ngram.objects.all()[0:10000]
	list_articleNgram = []
	for ngram in ngrams:
		gram = ngram.Ngram
		ngramId = ngram.NgramId
		print("gram: ", gram)
		
		for article in articles:

			processedText = article.ProcessedText
			articleId = article.ArticleId
			print("Ngram ID: ", ngramId)
			print("Article ID: ", articleId)

			my_regex = r"\b" + gram + r"\b"
			print("my_regex: ",my_regex)
			matches = re.findall(my_regex,processedText)
			print("match count: ", len(matches))

			dict_articleNgram = {}
			dict_articleNgram["NgramId"] = ngramId
			dict_articleNgram["ArticleId"] = articleId
			dict_articleNgram["Frequency"] = len(matches)
			dict_articleNgram["StdFrequency"] = 0

			list_articleNgram.append(dict_articleNgram)

	print("list_articleNgram: ",list_articleNgram)

	try:
	#	db = MySQLdb.connect(host="localhost", user="root", db="nlp")  
	#	cur = db.cursor()
		cur = connection.cursor()
		stmt= """INSERT INTO nlp2.politicsApp_articlengram (NgramId_id, ArticleId_id,Frequency,StdFrequency) 
					values (%(NgramId)s,%(ArticleId)s,%(Frequency)s,%(StdFrequency)s)"""
		cur.executemany(stmt, list_articleNgram)
		connection.commit()
		print("affected rows {}".format(cur.rowcount))
	#	db.commit()
	except MySQLdb.IntegrityError:
		print("failed to insert values")
	finally:
		cur.close()







