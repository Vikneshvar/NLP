from politicsApp.models import Articles
import os, re
from django.db import connection

# Clean the article
def run():
#	cursor = connection.cursor()
#	cursor.execute("select * from politicsApp_articles;")
#	row = cursor.dictfetchall()
#	articles = Articles.objects.raw("select * from politicsApp_articles;")
	articles = Articles.objects.all()
	for article in articles:
		print('Raw String: ', article.RawText)
		dict_output = textClean(article.RawText)
		processedText=dict_output.get('string')
		wordCount=dict_output.get('count')
		articleId = article.ArticleId
		Articles.objects.filter(ArticleId=articleId).update(ProcessedText=processedText,WordCount=wordCount)

		
def textClean(rawText):
	dict_output=regExp(rawText)
	return dict_output

def regExp(text):
	newString = ''
	for ch in text:
		if ch.isalpha() == True or ch == ',' or ch == '.' or ch == ' ' or ch == ';' or ch == '-' or ch == '"':
			newString+= ch
		else:
			pass


	newString = newString.replace(',','.')
	newString = newString.replace('-','')
	newString = newString.replace('"','')
	newString = newString.replace(';','.')

	print(" \n String:",newString)
	print("\n")

	# Replace multiple space by single space
	newString = re.sub(r'\s+',' ',newString)
	# Replace multiple periods by single period 
	newString = re.sub(r'\.+','.',newString)
	newString = re.sub(r'\.+\s+\.+','.',newString)


	print(" \n newString:",newString)
	print("\n")
	
	# abbrevation
	newString = re.sub(r'([A-Z]+\.)\s+([a-zA-Z]+)',r'\1\2',newString)

	# 2 to 5 letter abbrevation
	newString = re.sub(r'([A-Z])(\.)([A-Z])(\.)([a-zA-Z]+)',r'\1\3',newString)
	newString = re.sub(r'([A-Z])(\.)([A-Z])(\.)([A-Z])(\.)([a-zA-Z]+)',r'\1\3\5',newString)
	newString = re.sub(r'([A-Z])(\.)([A-Z])(\.)([A-Z])(\.)([a-zA-Z]+)',r'\1\3\5\7',newString)
	newString = re.sub(r'([A-Z])(\.)([A-Z])(\.)([A-Z])(\.)([A-Z])(\.)([a-zA-Z]+)',r'\1\3\5\7\9',newString)


	# Remove space before a new sentence starts
	# example: President Trump. says the United States
	# Run 2 times to remove properly, else some places are missed
	newString = re.sub(r'([a-z]+\.)\s+([a-zA-Z]+)',r'\1\2',newString)
	newString = re.sub(r'([a-z]+\.)\s+([a-zA-Z]+)',r'\1\2',newString)

	# Remove space at the end of sentence
	# example: big prizes.but the .year.old
	# Run 2 times to remove properly, else some places are missed
	newString = re.sub(r'([a-z]+)\s+(\.+[a-zA-Z]+)',r'\1\2',newString)
	newString = re.sub(r'([a-z]+)\s+(\.+[a-zA-Z]+)',r'\1\2',newString)

	# Remove space before and after a sentence
	# example: ahead of Game . I dont
	newString = re.sub(r'([a-z]+)\s+(\.)\s+([a-zA-Z]+)',r'\1\2\3',newString)

	print("newString2222: ",newString)
	print("\n")



	finalString = ''
	wordCount = 0
	for each in newString.split('.'):
#		print("each:",each)		
		if(len(each.split(' '))>1):
			for word in each.split(' '):
				if (len(word)>1 and len(word)<7) and (word.isupper() == True):
					finalString+=word
					wordCount+=1
					if each.split(' ').index(word) == len(each.split(' '))-1:
						finalString+=''
					else:
						finalString+=' '
				else:
					finalString+=word.lower()
					wordCount+=1
#					print("flag",flag)
					if each.split(' ').index(word) == len(each.split(' '))-1:
						finalString+=''
					else:
						finalString+=' '						
#					print("finalString:",finalString)

		else:
			if (len(each)>1 and len(each)<7) and (each.isupper() == True):
					finalString+=each
					wordCount+=1
			else:
				finalString+=each.lower()
				wordCount+=1
		finalString+='.'
	
	finalString = re.sub(r'\.+','.',finalString)


	# Remove space before a new sentence starts
	# example: President Trump. says the United States
	# Run 2 times to remove properly, else some places are missed
	finalString = re.sub(r'([a-z]+\.)\s+([a-zA-Z]+)',r'\1\2',finalString)
	finalString = re.sub(r'([a-z]+\.)\s+([a-zA-Z]+)',r'\1\2',finalString)

	# Remove space at the end of sentence
	# example: big prizes.but the .year.old
	# Run 2 times to remove properly, else some places are missed
	finalString = re.sub(r'([a-z]+)\s+(\.+[a-zA-Z]+)',r'\1\2',finalString)
	finalString = re.sub(r'([a-z]+)\s+(\.+[a-zA-Z]+)',r'\1\2',finalString)

	# Remove space before and after a sentence
	# example: ahead of Game . I dont
	finalString = re.sub(r'([a-z]+)\s+(\.)\s+([a-zA-Z]+)',r'\1\2\3',finalString)




	print("finalString:",finalString)
	print("\n")
	print('wordCount:',wordCount)
	dict_output={}
	dict_output['string']=finalString
	dict_output['count']=wordCount

	return dict_output

"""	
	nextString = ''
	for each in newString.split('.'):
		# need changes - not working for "RAriz. conceded" - result is RAriz.
		if (len(each)>1 and len(each)<7)


		if ((any(x.isupper() for x in each) and any(x =='.' for x in each)) or each.isupper() == True) and (len(each)>1 and len(each)<7):
			pass
		else:
			each = each.lower()
		nextString+=each
		nextString+=' '
"""

	# Need to check for continous space and periods
	

	


