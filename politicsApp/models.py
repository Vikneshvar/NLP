from django.db import models

# Create your models here.
# If Char field is used, then the total size of all fields in the table is 65535
# Text field have no such limitation 
class Articles(models.Model):
	ArticleId = models.AutoField(primary_key=True)
	Source = models.CharField(max_length=20)
	RawText = models.TextField(null=False, blank=False)
	ProcessedText = models.TextField(blank=True)
	WordCount = models.IntegerField(default=0)

class Ngram(models.Model):
	NgramId = models.AutoField(primary_key = True)
	Ngram = models.CharField(max_length=100)

class ArticleNgram(models.Model):
	ArticleNgramId = models.AutoField(primary_key = True)
	ArticleId = models.ForeignKey(Articles, on_delete=models.CASCADE)
	NgramId = models.ForeignKey(Ngram, on_delete=models.CASCADE)
	Frequency = models.IntegerField(default=0)
	StdFrequency = models.FloatField(default=0)

class Interaction(models.Model):
	ArticleNgramId = models.IntegerField(primary_key = True)
	ArticleId_id = models.IntegerField(default = 0)
	NgramId_id = models.IntegerField(default = 0)
	Frequency = models.IntegerField(default=0)
	WordCount = models.IntegerField(default=0)
	StdFrequency = models.FloatField(default=0)
	Source = models.CharField(max_length=20)

