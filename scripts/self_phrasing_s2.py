import nltk
#nltk.download()
from nltk.util import ngrams
import os, re, string
from politicsApp.models import Articles, Ngram, NgramDuplicates
from django.db.models import Q
from unidecode import unidecode


def run():
	ngrams = NgramDuplicates.objects.filter (NgramSize_D=1) | NgramDuplicates.objects.filter(NgramSize_D=2)

	one_word_list = []
	two_word_list = []
	for ngram_ in ngrams:
		ngram = ngram_.Ngram_D
		if len(ngram.split(' '))==2:
			two_word_list.append(ngram)
		else:
			one_word_list.append(ngram)

	print('two_word_list',two_word_list)
	print('one_word_list',one_word_list)

	score_less_list=[]
	phrase_list = []
	
	#Don't combine the words if either word 1 or word 2 occur fewer than 
    # min_count (default = 5) times in the training text.
	min_count = 3
	for each_two_word in two_word_list:
		print(two_word_list.index(each_two_word))
		print('each_two_word',each_two_word)
		two_word_count = two_word_list.count(each_two_word)
		print('two_word_count',two_word_count)
		word1 = each_two_word.split(' ')[0]
		word2 = each_two_word.split(' ')[1]
		print('word1',word1)
		print('word2',word2)
		one_word1_count = one_word_list.count(word1)
		one_word2_count = one_word_list.count(word2)
		print('one_word1_count',one_word1_count)
		print('one_word2_count',one_word2_count)
#		score_two_word = (abs(two_word_count-min_count)/(one_word1_count*one_word2_count))*len(one_word_list)
		score_two_word = two_word_count
		print('score_two_word',score_two_word)
		if score_two_word < 20:
			phrase = word1+'_'+word2
			phrase_list.append(phrase)
		else:
			score_less_list.append(each_two_word)

#		score_dict ={}
#		score_dict['two_word']=each_two_word
#		score_dict['score']=score_two_word
#		score_dict_list.append(score_dict)

	print('score_less_list',score_less_list)
	print('phrase_list',phrase_list)
	print('len(score_less_list)',len(score_less_list))
	print('len(phrase_list)',len(phrase_list))

	articles = Articles.objects.all()[0:1]

	for article in articles:
		processedText = article.ProcessedText
		print('processedText: ', processedText)

		for each_phrase in phrase_list:
			phrasedText = makePhrases(processedText,each_phrase)
			articleId = article.ArticleId
#			Articles.objects.filter(ArticleId=articleId).update(ProcessedText=phrasedText)

def makePhrases(processedText,phrase):

	regex = phrase.split('_')[0]+' '+phrase.split('_')[1]
	my_regex = r"\b" + regex + r"\b"
	print("my_regex: ",my_regex)
	phrasedText = re.sub(my_regex,phrase,processedText)
	print("phrasedText: \n", phrasedText)

	return phrasedText



