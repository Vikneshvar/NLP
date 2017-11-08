from politicsApp.models import Articles, Ngram, ArticleNgram, Interaction
import os, re, string
from django.db import connection
from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter


# Clean the article
def run():
	articles = Articles.objects.all()
	for article in articles:
		print('Raw String: ', article.RawText)
		unidecode_text = unidecode(article.RawText)
		print('unidecode_text: ', unidecode_text)
		dict_output = textClean(unidecode_text)
		processedText=dict_output.get('string')
		wordCount=dict_output.get('count')
		articleId = article.ArticleId
#		Articles.objects.filter(ArticleId=articleId).update(ProcessedText=processedText,WordCount=wordCount)

def textClean(rawText):
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
	newString = newString.lower()

	# 2 to 5 letter abbrevation
	newString = re.sub(r'([a-z])(\.)([a-z])(\.)',r'\1\3',newString)
	newString = re.sub(r'([a-z])(\.)([a-z])(\.)([a-z])(\.)',r'\1\3\5',newString)
	newString = re.sub(r'([a-z])(\.)([a-z])(\.)([a-z])(\.)([a-z])(\.)',r'\1\3\5\7',newString)

	print(" \n newString:",newString)
	print("\n")

	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(newString)

	dict_output={}
	dict_output['string']=newString
	dict_output['count']=len(tokens)

	return dict_output
	


