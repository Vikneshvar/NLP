from politicsApp.models import Ngram, Articles, ArticleNgram

def run():
	articles = Articles.objects.all()
	ngrams = Ngram.objects.all()[:1000]
	options = {
		   1 : one,
		   2 : two,
		   3 : three,
		   4 : four,
		   5 : five,
           6 : six,
	}

	for ngram in ngrams:
		gram = ngram.Ngram
		ngramId = ngram.NgramId
		print('ngramId---------------------',ngramId)
		
		words_in_gram = len(gram.split())
		print('words in gram-------',words_in_gram)

		for article in articles:
			articleId = article.ArticleId
			
			if words_in_gram != 0: 
				finalCount = options[words_in_gram](gram,article)
				print('Total matches-----',finalCount)
"""
				articleNgram = ArticleNgram()
				articleNgram.NgramId = 	Ngram.objects.get(NgramId=ngramId)
				articleNgram.ArticleId = Articles.objects.get(ArticleId=articleId)
				articleNgram.Frequency = finalCount
				articleNgram.save()
"""

def one(gram,article):
	processedText = article.ProcessedText
	articleId = article.ArticleId
	print('articleId------------------------------------------------', articleId)
	count = 0
	for word in processedText.replace('.',' ').split():
		print('gram-------------',gram)
		print('word-------------',word)
		if gram == word:
			count+=1
			print('matches-----',count)	
	return count

def two(gram,article):
	processedText = article.ProcessedText
	articleId = article.ArticleId
	print('articleId------------------------------------------------', articleId)
	wordList = processedText.replace('.',' ').split()
	i=0
	j=0
	count = 0
	while(j<len(wordList)-1):
		j = i+1
		combo = ''
		m=i
		while(m<=j):
			combo = combo + wordList[m]
			if m!=j:
				combo = combo + ' '
			m+=1
		print('gram-------------',gram)
		print('combo-------------',combo)
		if combo == gram:
			count+=1
			print('matches-----',count)	
		i+=1	
	return count

def three(gram,article):
	processedText = article.ProcessedText
	articleId = article.ArticleId
	print('articleId------------------------------------------------', articleId)
	wordList = processedText.replace('.',' ').split()
	i=0
	j=0
	count = 0
	while(j<len(wordList)-1):
		j = i+2
		combo = ''
		m=i
		while(m<=j):
			combo = combo + wordList[m]
			if m!=j:
				combo = combo + ' '
			m+=1
		print('gram-------------',gram)
		print('combo-------------',combo)
		if combo == gram:
			count+=1
			print('matches-----',count)	
		i+=1	
	return count

def four(gram,article):
	processedText = article.ProcessedText
	articleId = article.ArticleId
	print('articleId------------------------------------------------', articleId)
	wordList = processedText.replace('.',' ').split()
	i=0
	j=0
	count = 0
	while(j<len(wordList)-1):
		j = i+3
		combo = ''
		m=i
		while(m<=j):
			combo = combo + wordList[m]
			if m!=j:
				combo = combo + ' '
			m+=1
		print('gram-------------',gram)
		print('combo-------------',combo)
		if combo == gram:
			count+=1
			print('matches-----',count)	
		i+=1	
	return count

def five(gram,article):
	processedText = article.ProcessedText
	articleId = article.ArticleId
	print('articleId------------------------------------------------', articleId)
	wordList = processedText.replace('.',' ').split()
	i=0
	j=0
	count = 0
	while(j<len(wordList)-1):
		j = i+4
		combo = ''
		m=i
		while(m<=j):
			combo = combo + wordList[m]
			if m!=j:
				combo = combo + ' '
			m+=1
#		print('gram-------------',gram)
#		print('combo-------------',combo)
		if combo == gram:
			count+=1
			print('matches-----',count)	
		i+=1	
	return count

def six(gram,article):
	processedText = article.ProcessedText
	articleId = article.ArticleId
	print('articleId------------------------------------------------', articleId)
	wordList = processedText.replace('.',' ').split()
	i=0
	j=0
	count = 0
	while(j<len(wordList)-1):
		j = i+5
		combo = ''
		m=i
		while(m<=j):
			combo = combo + wordList[m]
			if m!=j:
				combo = combo + ' '
			m+=1
		print('gram-------------',gram)
		print('combo-------------',combo)
		if combo == gram:
			count+=1
			print('matches-----',count)	
		i+=1	
	return count	






