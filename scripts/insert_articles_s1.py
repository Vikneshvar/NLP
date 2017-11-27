from politicsApp.models import Articles
import os

def run():
	print('Hi')
	rootdir = "/Users/vik_work/Desktop/Workspace/NLP_Political/Data/NLP"
	for subdir, dirs, files in os.walk(rootdir):
		for subdir in dirs:
			print("subdir: ",subdir)
			type_ = subdir
			path=os.path.join(rootdir,subdir)
			print('path',path)
			for subdir, dirs, files in os.walk(path):
				for each in dirs:  
					sourceName = each
					print("Source:",sourceName)
					sourcedir = os.path.join(path,sourceName)
					print('sourcedir',sourcedir)
					file_count = 0
					for textFile in os.listdir(sourcedir):
						if textFile.endswith(".txt"):
							file_count+=1
							print("file_count: ",file_count)
							
							textFile = textFile.strip()
							fileName = ''
							for each in textFile:
								if each.isalpha() == True or each==' ' or each=='.':
									fileName+=each

							print("Filename: ",fileName)
							f = open(os.path.join(sourcedir,textFile),"r",encoding='utf-8', errors='ignore')
							rawText = f.read()
							article = Articles(Source=sourceName, RawText=rawText, FileName=fileName, Type = type_)
							article.save()

				

			
