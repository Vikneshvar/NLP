from politicsApp.models import Articles
import os

def run():
	
	rootdir = "/Users/Vik/Desktop/Project/NLP/"
	for subdir, dirs, files in os.walk(rootdir):
		for title in dirs:
			sourceName = title
			print("Source: ",sourceName)
			sourcedir = os.path.join(rootdir,sourceName)
			file_count = 0
			for textFile in os.listdir(sourcedir):
				if textFile.endswith(".txt"):
					file_count+=1
					print("file_count: ",file_count)
					print("Filename: ",textFile)
					f = open(os.path.join(sourcedir,textFile),"r",encoding='utf-8', errors='ignore')
					rawText = f.read()
					article = Articles(Source=sourceName, RawText=rawText)
					article.save()

				

			
