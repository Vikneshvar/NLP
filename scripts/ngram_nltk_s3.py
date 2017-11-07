import nltk
#nltk.download()
from nltk.util import ngrams
import string
from politicsApp.models import Articles, Ngram

def run():
	articles = Articles.objects.all()
	gramList = []
	ngramList = []
	
	for article in articles:
		processedText = article.ProcessedText
		sentenceList = processedText.split('.')

		print('sentenceList:', sentenceList)
		for sentence in sentenceList:
			sentence_list = sentence.split(' ')
			ng_list = word_grams(sentence_list)
			gramList.append(ng_list)

#	print("Len gramList ", len(gramList))
	print("\n -------gramList ", gramList)

	for list_ in gramList:
		for ngram in list_:
			if not ngram == '' or ngram == ' ' or ngram == '  ':
				ngramList.append((ngram.strip(' ')))

	print("\n *************** ngramList ", ngramList)

	print('len(ngramList))',len(ngramList))
	ngramList_noDup = list(set(ngramList))
	print('len(ngramList_noDup))',len(ngramList_noDup))
	
	ngramList_noDup.sort(key=len)
	ngramList_noDup_dict=[(each,len(each.split(' '))) for each in ngramList_noDup]		
	
	# Store ngram in database
	for item in ngramList_noDup_dict:
		print(item[0])
		print(item[1])		
		ngram = Ngram(Ngram=item[0],NgramSize=item[1])
		ngram.save()


def word_grams(words, min=1, max=7):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s


