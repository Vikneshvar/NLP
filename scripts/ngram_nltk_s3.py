import nltk
#nltk.download()
from nltk.util import ngrams
import string
from politicsApp.models import Articles, Ngram, NgramDuplicates

# Run twice - one for Ngram table and another for NgramDuplicates table
def run():
	articles = Articles.objects.filter(Type='Training')
	gramList = []
	ngramList = []
	
	for article in articles:
		processedText = article.ProcessedText
		phrasedText = article.PhrasedText_2
				
		processedSentenceList = processedText.split('.')
		phrasedSentenceList = phrasedText.split('.')
#		print('---------',len(processedSentenceList))
#		print('*********',len(phrasedSentenceList))

#		print('sentenceList:', sentenceList)
		for sentence in phrasedSentenceList:
			sentence_list = sentence.strip().split(' ')
			ng_list = word_grams(sentence_list)

			for item in ng_list:
				item = item.strip()
#				print(len(item))
#				print('item --'+item+'22')
				if len(item)>1:
					ngramList.append(item)
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

	ngramList_noDup_dict=[(each,len(each.split(' '))) for each in ngramList_noDup]		
	
	# Store ngram in database
	for item in ngramList_noDup_dict:
#		print(item[0])
#		print(item[1])		
#		ngram = NgramDuplicates(Ngram_D=item[0],NgramSize_D=item[1])
		ngram = Ngram(Ngram=item[0],NgramSize=item[1])
		ngram.save()


def word_grams(words, min=1, max=7):
	s = []
	for n in range(min, max):
		for ngram in ngrams(words, n):
			p =' '.join(str(i) for i in ngram)
#			print('\n p ',p)
			s.append(p)
	return s


