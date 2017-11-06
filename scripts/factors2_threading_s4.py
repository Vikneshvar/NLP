import multiprocessing
from politicsApp.models import Ngram, Articles, ArticleNgram
import MySQLdb,re
from django.db import connection


def run():
    ngrams = Ngram.objects.all()
    articles = Articles.objects.all()
    ngramIds = []
    ngramValues = []
    articleIds = []
    processedText = []
    pT = []
    aI = []
    
    for ngram in ngrams:
        ngramIds.append(ngram.NgramId)
        ngramValues.append(ngram.Ngram)

    for article in articles:
        articleIds.append(article.ArticleId) 
        processedText.append(article.ProcessedText)   

    for i in range(len(ngramIds)):
        pT.append(processedText)

    for i in range(len(ngramIds)):
        aI.append(articleIds) 

#   print('ngramIds',ngramIds)
#   print('aI',aI)
    # Zip the parameters because pool.map() takes only one iterable
    params = zip(ngramIds, ngramValues, aI, pT)
    
    pool = multiprocessing.Pool()
    list_articleNgram = pool.map(runSimulation, params)
#   print('list_articleNgram',list_articleNgram)

    print('\n')

    cur = connection.cursor()
    for each in list_articleNgram:
        stmt= """INSERT INTO nlp2.politicsApp_articlengram (NgramId_id, ArticleId_id,Frequency,StdFrequency) 
                    VALUES (%s,%s,%s,%s)"""
        cur.executemany(stmt, each)
        connection.commit()
 #      print("affected rows {}".format(cur.rowcount))


def runSimulation(params):
    """This is the main processing function. It will contain whatever
    code should be run on multiple processors."""
    ngramId, ngram, aI, pT = params
    processedData = []
    print('ngramId',ngramId)
#    print('ngram',ngram)
#    print('aI',aI)

    list_articleNgram = []

    for eachProcessedText in pT:
        my_regex = r"\b" + ngram + r"\b"
#        print("my_regex: ",my_regex)
        matches = re.findall(my_regex,eachProcessedText)
#        print("match count: ", len(matches))
        t = (ngramId,pT.index(eachProcessedText)+1,len(matches),0)
        list_articleNgram.append(t)

    return list_articleNgram



"""
        dict_articleNgram = {}
        dict_articleNgram["NgramId"] = ngramId
        dict_articleNgram["ArticleId"] = pT.index(eachProcessedText)+1
        dict_articleNgram["Frequency"] = len(matches)
        dict_articleNgram["StdFrequency"] = 0
"""
