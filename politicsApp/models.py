from django.db import models

# Database models created 
# If Char field is used, then the total size of all fields in the table is 65535
# Text field have no such limitation

# Articles table to store all the articles 
class Articles(models.Model):
	ArticleId = models.AutoField(primary_key=True)
	Source = models.CharField(max_length=20)
	RawText = models.TextField(null=False, blank=False)
	ProcessedText = models.TextField(blank=True)
	PhrasedText_2 = models.TextField(blank=True)
	SampledText = models.TextField(blank=True)
	WordCount = models.IntegerField(default=0)
	FileName = models.TextField(blank=True)
	Type = models.TextField(blank=True)

# Ngram table
class Ngram(models.Model):
	NgramId = models.AutoField(primary_key = True)
	Ngram = models.CharField(max_length=100)
	NgramSize = models.IntegerField(default=0)
	IDF = models.FloatField(default=0)

# Ngram with duplicates included
class NgramDuplicates(models.Model):
	NgramId_D = models.AutoField(primary_key = True)
	Ngram_D = models.CharField(max_length=100)
	NgramSize_D = models.IntegerField(default=0)

# Article and Ngram Interaction table
class ArticleNgram(models.Model):
	ArticleNgramId = models.AutoField(primary_key = True)
	ArticleId = models.ForeignKey(Articles,related_name='AN_AI')
	NgramId = models.ForeignKey(Ngram, related_name='AN_NI')
	NgramSize = models.ForeignKey(Ngram, related_name='AN_NS')
	TF = models.FloatField(default=0)
	TFIDF = models.FloatField(default=0)

# Article and Ngram Interaction table with standard frequency updated 
class Interaction(models.Model):
	ArticleNgramId = models.IntegerField(primary_key = True)
	ArticleId_id = models.IntegerField(default = 0)
	NgramId_id = models.IntegerField(default = 0)
	NgramSize = models.IntegerField(default = 0)
	Frequency = models.IntegerField(default=0)
	WordCount = models.IntegerField(default=0)
	StdFrequency = models.FloatField(default=0)
	Source = models.CharField(max_length=20)

