from politicsApp.models import Articles, Ngram, ArticleNgram, Interaction, RemoveNgram
import os, re, string
from django.db import connection
from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
from gensim.models import Phrases
from nltk.util import ngrams
from math import sqrt
import pickle
import shutil

# Clean the article
def run():
	ngramList =[]
	articles = Articles.objects.filter(Type='Training')
	sentence_stream = []

	# Create a file in the directory to save bigrams and ngram removelist
	path = "/Users/Vik/Desktop/soper/project/politicsNLP/Files"
	if not os.path.exists(path):
		os.makedirs(path)

	# Bigram
	# For (min=2, thres=3) -> ngram =170352, ngram_after_dup =128380 - crct combo
	# For (min=2, thres=3) -> ngram =170352, ngram_after_dup =128380
	bigram = Phrases(sentence_stream, min_count=2, threshold=3)
	print('\n bigram ***',bigram.vocab)
	
	for article in articles:
#		print('Raw String: ', article.RawText)
		unidecode_text = unidecode(article.RawText)
#		print('unidecode_text: ', unidecode_text)
		dict_output = textClean(unidecode_text,ngramList)
		processedText=dict_output.get('string')
#		print('processedText -----', processedText)
		wordCount=dict_output.get('count')
		articleId = article.ArticleId
		Articles.objects.filter(ArticleId=articleId).update(ProcessedText=processedText)
		

#	print('\n ngramList',ngramList)
	print('\n len(ngramList)',len(ngramList))
	print('\n ngramList after removing duplicates',len(list(set(ngramList))))
	# Without multiprocessing
	remove_list = getRemoveList(ngramList)

	# Remove duplicates from the remove list
	remove_list = list(set(remove_list))
	
	# Create a pickle file to save ngram remove list
	filename = 'remove_list' + '.pk'
	filePath = os.path.join(path, filename)

	with open(filePath,'wb') as fi:
		pickle.dump(remove_list,fi)

	print('remove_list',remove_list)
	print('Number of ngrams to be removed',len(remove_list))
	print('\n remove_list after removing duplicates ',len(list(set(remove_list))))

	# Loop for getting sampled text
	for article in articles:
		processedText = article.ProcessedText		
		dict_output	= sampleText(processedText,remove_list)
		sampledText=dict_output.get('string')
		print('sampled text-----', sampledText)
		wordCount=dict_output.get('count')
		articleId = article.ArticleId
		Articles.objects.filter(ArticleId=articleId).update(SampledText=sampledText)

	# Phrased Text
	# Loop to add vocabolary to bigram
	for article in articles:
		sampledText = article.SampledText
		addVocab(sampledText,bigram)

	# Create a pickle file
	filename = 'bigram' + '.pk'
	filePath = os.path.join(path, filename)

	with open(filePath,'wb') as fi:
		pickle.dump(bigram,fi)

	# Loop for getting phrased text
	for article in articles:
		sampledText = article.SampledText
#		print('sampledText: ', sampledText)
		dict_output = Phrase(sampledText,bigram)
		phrasedText=dict_output.get('string')
		wordCount=dict_output.get('count')
		print('phrasedText: ', phrasedText)
#		print('wordCount: ', wordCount)
		articleId = article.ArticleId
		Articles.objects.filter(ArticleId=articleId).update(PhrasedText_2=phrasedText,WordCount=wordCount)


#	vocab = bigram.vocab
#	print('\n bigram ***',bigram.vocab)
#	print('\n vocab ***',vocab)
#	print(vocab['mark','zuckerberg'])


def getRemoveList(ngramList):
	count_overall = len(ngramList)
	remove_list =[]
	# Sample
	# Smaller values of ‘sample’ mean words are less likely to be kept.
	sample = 0.001 
	# sample = 0.001 - ngrams removed - 39 - best chosen
	# sample = 0.01 - ngram removed - 2
	# sample = 0.0001 - ngrams removed - 
	
	for ngram in ngramList:
#		print('Index',ngramList.index(ngram))
#		print('ngram',ngram)
		count_ngram = ngramList.count(ngram)
		zwi = count_ngram/count_overall
		probability = (sqrt(zwi/sample)+1)*(sample/zwi)
#		print('probability',probability)
		if probability < 1:
			remove_list.append(ngram)

	return remove_list

def sampleText(processedText,remove_list):

	sampledText = processedText
	finalString = ''
	for each_ngram in remove_list:
#		print('each_ngram',each_ngram)
		if len(each_ngram) > 0:
			my_regex = r"\b" + each_ngram + r"\b"
#			print("my_regex: ",my_regex)
			sampledText = re.sub(my_regex,'',sampledText)
#			print('sampledText',sampledText)
			# Remove trailing and leading spaces in sentence
			finalString = ''.join([doc.strip()+'.' for doc in sampledText.split('.')])

			# Remove multiple periods together
			finalString = finalString.replace('..','.')

	# Count number of words
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(finalString)

	dict_output={}
	dict_output['string']=finalString
	dict_output['count']=len(tokens)

	return dict_output

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

	# Convert single quote word to double quote
	newString = re.sub(r'\'\w+\'',toDoubleQuote,newString)
	
	# Double and single quotes
	newString = newString.replace('"','.')

	# Remove apostrophe and characters after that
#	newString = newString.replace("'",'')
	newString = re.sub(r'(?:\'\w+)','',newString)

	# Convert without period acronyms by with period acronyms
	newString = re.sub(r'(?<![A-Z])[A-Z]{2,7}(?![A-Z])',toPeriodsInbetween,newString)

	# Converting all character to lower
	newString = newString.lower()
#	print(" String: With Abbrevation Period \n",newString)
	print('\n')
	
	# Convert 2 to 10 letter acronym with period inbetween to upper and remove period
	newString = re.sub(r'(?:([a-z]\.)+([a-z]\.)+)',toUpperRemovePeriod,newString)

#	print("newString: Without period \n",newString)
	print('\n')

	# Remove trailing and leading spaces in sentence
	finalString = ''.join([doc.strip()+'.' for doc in newString.split('.')])
#	print("fString: Removing trailing and leading spaces and to lower \n",finalString)
	print('\n')

	# Remove multiple periods together
	finalString = finalString.replace('..','.')
#	print('processedText -------',finalString)

				
	processedSentenceList = finalString.split('.')
#	print('---------',len(processedSentenceList))
#	print('sentenceList:', sentenceList)
	
	for sentence in processedSentenceList:
		ng_list = []
		sentence_list = sentence.strip().split(' ')
		ng_list = word_grams(sentence_list)

		for item in ng_list:
			item = item.strip()
#				print(len(item))
#				print('item --'+item+'22')
			if len(item)>1:
				ngramList.append(item)
#			else:
#				print('False')
#				print(item)

	# Count number of words
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(finalString)

	dict_output={}
	dict_output['string']=finalString
	dict_output['count']=len(tokens)

	return dict_output

def addVocab(sampledText,bigram):
	# Using Genism
	# Make sentence stream
	document = sampledText.split('.')
#	print('document', document)
	sentence_stream = [doc.split(' ') for doc in document]
#	print('sentence_stream -------',sentence_stream)

	# Add vocab to bigram # Training
	bigram.add_vocab(sentence_stream)


def Phrase(sampledText,bigram):
	sentence_list = sampledText.split('.')
	finalString = ''
	for each_sentence in sentence_list:
		input_string_in_list = each_sentence.split(' ')
#		print('input_string_in_list',input_string_in_list)
		# Send new string to the trained detector
#		print('bigram[input_string_in_list] ----- \n', bigram[input_string_in_list])
		out = ' '.join(each for each in bigram[input_string_in_list])+'.'
#		print('out',out)
		finalString+=out
#	print('finalString',finalString)

	# Count number of words
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(finalString)

	dict_output={}
	dict_output['string']=finalString
	dict_output['count']=len(tokens)

	return dict_output

# Create ngrams of one words only - coz we are removing only one words from the text
def word_grams(words, min=1, max=2):
	s = []
	for n in range(min, max):
		for ngram in ngrams(words, n):
			p =' '.join(str(i) for i in ngram)
#			print('\n p ',p)
			s.append(p)
	return s

def toUpperRemovePeriod(match):
	groups = match.group()
#	print('groups---',groups)
	acronym = groups.replace('.','')
	acronym = acronym.upper()
#	print('acronym---',acronym)
	return acronym

def	toPeriodsInbetween(match):
	match = match.group()
	withPeriods = '.'.join(ch for ch in match)
	withPeriods+='.'
#	print('withPeriods',withPeriods)
	return withPeriods

def toLower(match):
	match = match.group()
	lowecase = match.lower()
	return lowecase	

def toDoubleQuote(match):
	match = match.group()
	double = match.replace("'",'"')
	return double	


