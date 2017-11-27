import nltk
#nltk.download()
from nltk.util import ngrams
import string,re
from politicsApp.models import Articles, Ngram, NgramDuplicates
from math import log
from django.db import connection


# Run twice - one for Ngram table and another for NgramDuplicates table
def run():
	articles = Articles.objects.filter(Type='Training')
	gramList = []
	ngramList = []
	total_docs = len(articles)
	list_Ngram = []
	
	for article in articles:
		processedText = article.ProcessedText
		phrasedText = article.PhrasedText_2
				
		processedSentenceList = processedText.split('.')
		phrasedSentenceList = phrasedText.split('.')
#		print('---------',len(processedSentenceList))
#		print('*********',len(phrasedSentenceList))

#		print('sentenceList:', sentenceList)
		for sentence in processedSentenceList:
			sentence_list = sentence.strip().split(' ')
			ng_list = word_grams(sentence_list)

			for item in ng_list:
				item = item.strip()
#				print(len(item))
				my_regex = r"(?! )[A-Za-z ]*(?<! )$"
				matches = re.match(my_regex,item)
				if len(item)>1 and matches:
					ngramList.append(item)
					print('item --'+item+'22')
				else:
					print('False')
					print('item --'+item+'$$')

#	print("\n *************** ngramList ", ngramList)
	print('len(ngramList))',len(ngramList))

#	ngramList_noDup = ngramList
	ngramList_noDup = list(set(ngramList))
	print('len(ngramList_noDup))',len(ngramList_noDup))

	size1=0
	size2=0
	size3=0
	size4=0
	size5=0
	size6=0
	for item in ngramList_noDup:
		if len(item.split(' ')) == 1:
			size1+=1
		if len(item.split(' ')) == 2:
			size2+=1			
		if len(item.split(' ')) == 3:
			size3+=1
		if len(item.split(' ')) == 4:
			size4+=1
		if len(item.split(' ')) == 5:
			size5+=1
		if len(item.split(' ')) == 6:
			size6+=1

	print('size1 {}, size2 {}, size3 {}, size4 {}, size5 {}, size6 {} '.format(size1,size2,size3,size4,size5,size6))

	print('len(ngramList_noDup after))',len(ngramList_noDup))
	
	ngramList_noDup.sort(key=len)

	for ngram in ngramList_noDup:

		df_count = 0
		for article in articles:
			processedText = article.ProcessedText
#			print("processedText", processedText)
			if bool(re.search(ngram, processedText)) == True:
				df_count+=1

#		print('df_count',df_count)
		idf=log(total_docs/df_count)		
		dict_Ngram = {}
		dict_Ngram["Ngram"] = ngram
		dict_Ngram["NgramSize"] = len(ngram.split(' '))
		dict_Ngram["IDF"] = idf

		list_Ngram.append(dict_Ngram)

	try:
		cur = connection.cursor()
		stmt= """INSERT INTO nlp2.politicsApp_ngram (Ngram,NgramSize,IDF) 
					values (%(Ngram)s,%(NgramSize)s,%(IDF)s)"""
		cur.executemany(stmt, list_Ngram)
		connection.commit()
		print("affected rows {}".format(cur.rowcount))
	except MySQLdb.IntegrityError:
		print("failed to insert values")
	finally:
		cur.close()


def word_grams(words, min=1, max=7):
	s = []
	for n in range(min, max):
		for ngram in ngrams(words, n):
			p =' '.join(str(i) for i in ngram)
#			print('\n p ',p)
			s.append(p)
	return s


