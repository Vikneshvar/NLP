import multiprocessing
from politicsApp.models import Ngram, Articles, ArticleNgram
import MySQLdb,re
from django.db import connection


def run():
	ngrams = Ngram.objects.all()
	articles = Articles.objects.filter(Type='Training')
	ngramIds = []
	ngramValues = []
	ngramSize = []
	articleIds = []
	phrasedText = []
	pT = []
	aI = []
	
	for ngram in ngrams:
		ngramIds.append(ngram.NgramId)
		ngramValues.append(ngram.Ngram)
		ngramSize.append(ngram.NgramSize)

	for article in articles:
		articleIds.append(article.ArticleId) 
		phrasedText.append(article.PhrasedText_2)   

	for i in range(len(ngramIds)):
		pT.append(phrasedText)

	for i in range(len(ngramIds)):
		aI.append(articleIds) 

#   print('ngramIds',ngramIds)
#   print('aI',aI)
	# Zip the parameters because pool.map() takes only one iterable
	params = zip(ngramIds, ngramValues, ngramSize, aI, pT)
	
	pool = multiprocessing.Pool()
	list_articleNgram = pool.map(runSimulation, params)
#	print('list_articleNgram',list_articleNgram)

	print('\n')

	cur = connection.cursor()
	for each in list_articleNgram:
		stmt= """INSERT INTO nlp2.politicsApp_articlengram (NgramId_id, ArticleId_id,NgramSize_id,Frequency,StdFrequency) 
					VALUES (%s,%s,%s,%s,%s)"""
		cur.executemany(stmt, each)
		connection.commit()
	print("affected rows {}".format(cur.rowcount))


def runSimulation(params):
	"""This is the main processing function. It will contain whatever
	code should be run on multiple processors."""
	ngramId, ngram,ngramSize, aI, pT = params
	processedData = []
	print('ngramId',ngramId)
	print('ngram',ngram)
	print('ngramSize',ngramSize)
#    print('aI',aI)

	list_articleNgram = []
	no_matches = []
	matches = 0
	for eachProcessedText in pT:
		my_regex = r"\b" + ngram + r"\b"
#       print("my_regex: ",my_regex)
		matches = re.findall(my_regex,eachProcessedText)
#       print("match count: ", len(matches))
		
		# Check for frequency before adding
		if len(matches)>0:
			t = (ngramId,pT.index(eachProcessedText)+1,ngramSize,len(matches),0)
			list_articleNgram.append(t)
			print('-------------Ngram matched Passed----------------')
		else:
			no_matches.append(len(matches))
			print('------------------------  Ngram match to article failed --------------------')			

	print('Number of non matches',len(no_matches))

	return list_articleNgram



"""
		dict_articleNgram = {}
		dict_articleNgram["NgramId"] = ngramId
		dict_articleNgram["ArticleId"] = pT.index(eachProcessedText)+1
		dict_articleNgram["Frequency"] = len(matches)
		dict_articleNgram["StdFrequency"] = 0
"""
