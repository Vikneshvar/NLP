from politicsApp.models import Articles
import os

def run():
	
	articles = Articles.objects.filter(Type='Test')

	for article in articles:
		rawText = article.RawText
		processedText = article.ProcessedText
		print(rawText)