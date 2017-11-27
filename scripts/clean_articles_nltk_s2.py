from politicsApp.models import Articles, Ngram, ArticleNgram, Interaction
import os, re, string
from django.db import connection
from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
from gensim.models import Phrases


# Clean the article
def run():
	articles = Articles.objects.all()
	sentence_stream = [u'the', u'mayor']
	bigram = Phrases(sentence_stream, min_count=3, threshold=4)
	print('\n bigram',bigram)
	
	for article in articles:
		print('Raw String: ', article.RawText)
		unidecode_text = unidecode(article.RawText)
#		print('unidecode_text: ', unidecode_text)
		processedText = textClean(unidecode_text,bigram)
#		processedText=dict_output.get('string')
#		wordCount=dict_output.get('count')
		articleId = article.ArticleId
		Articles.objects.filter(ArticleId=articleId).update(ProcessedText=processedText)
		print('\n bigram',bigram)

	for article in articles:
		processedText = article.ProcessedText
#		print('processedText: ', processedText)
		dict_output = Phrase(processedText,bigram)
		phrasedText=dict_output.get('string')
		wordCount=dict_output.get('count')
#		print('phrasedText: ', phrasedText)
		print('wordCount: ', wordCount)
		articleId = article.ArticleId
		Articles.objects.filter(ArticleId=articleId).update(PhrasedText_2=phrasedText,WordCount=wordCount)


def textClean(rawText,bigram):
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

#	newString = re.sub(r'([a-z])(\.)([a-z])(\.)',toUpperRemovePeriod,newString)
#	newString = re.sub(r'([a-z])(\.)([a-z])(\.)([a-z])(\.)',toUpperRemovePeriod,newString)
#	newString = re.sub(r'([a-z])(\.)([a-z])(\.)([a-z])(\.)([A-Z])(\.)',toUpperRemovePeriod,newString)
#	newString = re.sub(r'([A-Z])(\.)([A-Z])(\.)([A-Z])(\.)([A-Z])(\.)([A-Z])(\.)',toUpperRemovePeriod,newString)
#	newString = re.sub(r'([A-Z])(\.)([A-Z])(\.)([A-Z])(\.)([A-Z])(\.)([A-Z])(\.)([A-Z])(\.)',toUpperRemovePeriod,newString)

	# 2 to 7 letter abbrevation with no period inbetweem
#	newString = re.sub(r'(?<![A-Z])[A-Z]{2,7}(?![A-Z])',toUpperRemovePeriod,newString)

	print("newString: Without period \n",newString)
	print('\n')
#	newString = re.sub(r'(?<![A-Z])[A-Z]{1}[a-z]{1,7}(?![A-Z])',toLower,newString)

	# Remove trailing and leading spaces in sentence
	finalString = ''.join([doc.strip()+'.' for doc in newString.split('.')])
#	print("fString: Removing trailing and leading spaces and to lower \n",finalString)
#	print('\n')

	# Remove multiple periods together
	finalString = finalString.replace('..','.')


	# Using Genism
	# Make sentence stream
	document = finalString.split('.')
	sentence_stream = [doc.split(' ') for doc in document]
#	print('sentence_stream -------',sentence_stream)
	
	# Add vocab to bigram # Training
	bigram.add_vocab(sentence_stream)

	# Count number of words
#	tokenizer = RegexpTokenizer(r'\w+')
#	tokens = tokenizer.tokenize(finalString)

#	dict_output={}
#	dict_output['string']=finalString
#	dict_output['count']=len(tokens)

	return finalString


def Phrase(processedText,bigram):
	sentence_list = processedText.split('.')
	outputString = ''
	for each_sentence in sentence_list:
		input_string_in_list = each_sentence.split(' ')
#		print('input_string_in_list',input_string_in_list)
		# Send new string to the trained detector
		out = ' '.join(each for each in bigram[input_string_in_list])+'.'
#		print('out',out)
		outputString+=out
#	print('outputString',outputString)

	# Count number of words
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(outputString)

	dict_output={}
	dict_output['string']=outputString
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


