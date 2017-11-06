from django.db import connection
from politicsApp.models import Articles,ArticleNgram
import MySQLdb

def run():
	cur = connection.cursor()
	try:
		stmt = """INSERT INTO nlp2.politicsApp_interaction (ArticleNgramId,NgramId_id,Frequency,
				ArticleId_id,WordCount,StdFrequency,Source) select arng.ArticleNgramId,arng.NgramId_id,
				arng.Frequency,arng.ArticleId_id,article.WordCount,
				round((arng.Frequency/article.WordCount),2), article.Source 
				from nlp2.politicsApp_articlengram as arng, 
				nlp2.politicsApp_articles as article where article.ArticleId=arng.ArticleId_id 
				order by arng.ArticleNgramId asc"""
		cur.execute(stmt)
		connection.commit()
		print("affected rows {}".format(cur.rowcount))
	except MySQLdb.IntegrityError as e:
		print("failed to insert values",e)
	finally:
		cur.close()

