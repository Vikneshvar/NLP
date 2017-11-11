from politicsApp.models import Articles, Ngram, ArticleNgram, Interaction
import os, re, string
from django.db import connection
from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
from gensim.models import Phrases
from nltk.util import ngrams
from math import sqrt
import multiprocessing


# Clean the article
def run():
	articles = Articles.objects.all()
	ngramList =[]
	for article in articles:
		print('Raw String: ', article.RawText)
		unidecode_text = unidecode(article.RawText)
#		print('unidecode_text: ', unidecode_text)
		dict_output = textClean(unidecode_text,ngramList)
		processedText=dict_output.get('string')
		wordCount=dict_output.get('count')
		articleId = article.ArticleId
		Articles.objects.filter(ArticleId=articleId).update(ProcessedText=processedText)
	
	print('\n ngramList',ngramList)
	print('\n len(ngramList)',len(ngramList))
	print('\n ngramList after removing duplicates',len(list(set(ngramList))))
	# Without multiprocessing
	remove_list = getRemoveList(ngramList)

	# Remove duplicates from the remove list
	remove_list = list(set(remove_list))

	# With multiprocessing
#	pool = multiprocessing.Pool()
#	remove_list_list = pool.map(getRemoveList,ngramList[0:100])

#	remove_list = []
#	for each_remove_list_list in remove_list_list:
#		remove_list.append(each_remove_list_list)

	print('remove_list',remove_list)
	print('Number of ngrams to be removed',len(remove_list))
	print('\n remove_list after removing duplicates ',len(list(set(remove_list))))

	for article in articles[0:1]:
		processedText = article.ProcessedText		
		sampledText	= sampleText(processedText,remove_list)
		print('sampledText',sampledText)
		articleId = article.ArticleId
		Articles.objects.filter(ArticleId=articleId).update(SampledText=sampledText)

def getRemoveList(ngramList):
	count_overall = len(ngramList)
	remove_list =[]
	sample = 0.0005 
	
	for ngram in ngramList:
		print('Index',ngramList.index(ngram))
#		print('ngram',ngram)
		count_ngram = ngramList.count(ngram)
		zwi = count_ngram/count_overall
		probability = (sqrt(zwi/sample)+1)*(sample/zwi)
		print('probability',probability)
		if probability < 1:
			remove_list.append(ngram)

	return remove_list


def sampleText(processedText,remove_list):

	sampledText = processedText
	finalString = ''
	for each_ngram in remove_list:
		print('each_ngram',each_ngram)
		if len(each_ngram) > 0:
			my_regex = r"\b" + each_ngram + r"\b"
			print("my_regex: ",my_regex)
			sampledText = re.sub(my_regex,'',sampledText)
			print('sampledText',sampledText)
			# Remove trailing and leading spaces in sentence
			finalString = ''.join([doc.strip()+'.' for doc in sampledText.split('.')])

			# Remove multiple periods together
			finalString = finalString.replace('..','.')

	return finalString

# Create ngrams of one words only - coz we are removing only one words from the text
def word_grams(words, min=1, max=2):
	s = []
	for n in range(min, max):
		for ngram in ngrams(words, n):
			p =' '.join(str(i) for i in ngram)
#			print('\n p ',p)
			s.append(p)
	return s


def textClean(rawText,ngramList):
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

	# Double and single quotes
	newString = newString.replace('"','.')
	newString = newString.replace("'",'')

	# Convert without period acronyms by withperiod acronyms
	newString = re.sub(r'(?<![A-Z])[A-Z]{2,7}(?![A-Z])',toPeriodsInbetween,newString)

	# Converting all character to lower
	newString = newString.lower()
	print(" String: With Abbrevation Period \n",newString)
	print('\n')
	
	# Convert 2 to 10 letter acronym with period inbetween to upper and remove period
	newString = re.sub(r'[a-z]\.[a-z]\.{1,10}',toUpperRemovePeriod,newString)

	# 2 to 7 letter abbrevation with no period inbetweem
#	newString = re.sub(r'(?<![A-Z])[A-Z]{2,7}(?![A-Z])',toUpperRemovePeriod,newString)

	print("newString: Without period \n",newString)
	print('\n')
#	newString = re.sub(r'(?<![A-Z])[A-Z]{1}[a-z]{1,7}(?![A-Z])',toLower,newString)

	# Remove trailing and leading spaces in sentence
	finalString = ''.join([doc.strip()+'.' for doc in newString.split('.')])
	print("fString: Removing trailing and leading spaces and to lower \n",finalString)
#	print('\n')

	# Remove multiple periods together
	finalString = finalString.replace('..','.')
#	print('finalString -------',finalString)


	# Using Genism
	# Make sentence stream
#	document = finalString.split('.')
#	sentence_stream = [doc.split(' ') for doc in document]
#	print('sentence_stream -------',sentence_stream)
	

	# Count number of words
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(finalString)


				
	processedSentenceList = finalString.split('.')
	print('---------',len(processedSentenceList))

#		print('sentenceList:', sentenceList)
	for sentence in processedSentenceList:
		ng_list = []
		sentence_list = sentence.strip().split(' ')
		ng_list = word_grams(sentence_list)

		for item in ng_list:
			item = item.strip()
#				print(len(item))
#				print('item --'+item+'22')
			if len(item)!=0:
				ngramList.append(item)
			else:
				print('False')

	dict_output={}
	dict_output['string']=finalString
	dict_output['count']=len(tokens)

	return dict_output


def toUpperRemovePeriod(match):
	match = match.group()
	match = match.replace('.','')
	acronym = match.upper()
	return acronym

def	toPeriodsInbetween(match):
	match = match.group()
	withPeriods = '.'.join(ch for ch in match)
	withPeriods+='.'
	print(withPeriods)
	return withPeriods

def toLower(match):
	match = match.group()
	lowecase = match.lower()
	return lowecase	


