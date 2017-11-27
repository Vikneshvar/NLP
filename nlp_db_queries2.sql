# Select rows
select * from nlp2.politicsApp_articles
select count(*) from nlp2.politicsApp_articles
select * from nlp2.politicsApp_ngram
select count(*) from nlp2.politicsApp_ngram
select * from nlp2.politicsApp_articlengram
select count(*) from nlp2.politicsApp_articlengram
select * from nlp2.politicsApp_interaction
select count(*) from nlp2.politicsApp_interaction
select * from nlp2.politicsApp_nndata
select count(*) from nlp2.politicsApp_nndata
select * from nlp2.politicsApp_nndata_reduced
select count(*) from nlp2.politicsApp_nndata_reduced

# NgramSize Tables
select * from nlp2.politicsApp_nndata_ngram_size_1
select count(*) from nlp2.politicsApp_nndata_ngram_size_1
select count(*) from nlp2.politicsApp_nndata_ngram_size_2
select count(*) from nlp2.politicsApp_nndata_ngram_size_3

select count(*) from nlp2.politicsApp_ngram where NgramSize=2

select count(*) from nlp2.politicsApp_articlengram where Frequency=0
select count(*) from nlp2.politicsApp_articlengram where Frequency=1
select count(*) from nlp2.politicsApp_articlengram where Frequency=2
select count(*) from nlp2.politicsApp_articlengram where Frequency=4
select count(*) from nlp2.politicsApp_articlengram where Frequency=6
select count(*) from nlp2.politicsApp_articlengram where Frequency=8
select count(*) from nlp2.politicsApp_articlengram where Frequency=10
select count(*) from nlp2.politicsApp_articlengram where Frequency=14
select count(*) from nlp2.politicsApp_articlengram where Frequency=18
select count(*) from nlp2.politicsApp_articlengram where Frequency=22
select count(*) from nlp2.politicsApp_articlengram where Frequency=30
select count(*) from nlp2.politicsApp_articlengram where Frequency=40
select count(*) from nlp2.politicsApp_articlengram where Frequency=55
select * from nlp2.politicsApp_articlengram where Frequency=10

select * from nlp2.politicsApp_ngram where NgramId=18





# Check and Delete rows from table 

select * from nlp.django_migrations

select count(*) from nlp.django_migrations where app="politicsApp"

delete from nlp.django_migrations where id=17

# Drop table
drop table nlp2.politicsApp_interaction
drop table nlp2.`nlp.politicsApp_nndata`


# Delete all rows from table
truncate table nlp2.politicsApp_articles
truncate table nlp2.politicsApp_ngram
truncate table nlp2.politicsApp_articlengram
truncate table nlp2.politicsApp_interaction
truncate table nlp2.politicsApp_nndata
truncate table nlp2.politicsApp_nndata_latest

SET FOREIGN_KEY_CHECKS = 0
truncate table nlp.politicsApp_articles
truncate table nlp.politicsApp_ngram
SET FOREIGN_KEY_CHECKS = 1;

##### Create Table #####
create table nlp.politicsApp_interaction (
    ArticleNgramId int NOT NULL,
    ArticleId_id int NOT NULL,
    NgramId_id int NOT NULL,
    Frequency int NOT NULL,
    WordCount int NOT NULL,
    StdFrequency float NOT NULL,
    Source varchar(20) NOT NULL,
    primary key (ArticleNgramId)
)


######### Update Std frequency ###############

# create new table to get Std frequency
create table nlp.politicsApp_interaction as  
select arng.ArticleNgramId as 'ArticleNgramId', arng.NgramId_id as 'NgramId_id',
arng.Frequency as 'Frequency', arng.ArticleId_id as 'ArticleId_id',
article.WordCount as 'WordCount', round((arng.Frequency/article.WordCount),2) as 'SFrequency' 
from nlp.politicsApp_articlengram as arng, nlp.politicsApp_articles as article
where article.ArticleId=arng.ArticleId_id order by arng.ArticleNgramId asc 

# Not working - giving lock wait timeout error
# Different Update satements
SET SQL_SAFE_UPDATES = 0

# Using join update statement
update nlp.politicsApp_articlengram as AN join nlp.politicsApp_articles as A on 
AN.ArticleId_id=A.ArticleId set AN.StdFrequency=AN.Frequency/A.WordCount 
where AN.ArticleNgramId>=1 and AN.ArticleNgramId<=10501205

# Using select update statement
update nlp.politicsApp_articlengram as AN, (select ArticleId,WordCount from nlp.politicsApp_articles) as A 
set AN.StdFrequency=AN.Frequency/A.WordCount where AN.ArticleId_id=A.ArticleId  

SET SQL_SAFE_UPDATES = 1

 SHOW ENGINE INNODB STATUS
 SHOW PROCESSLIST
 KILL 15
 
 
 # Check std frequency of ngrams which is zero in all articles
  select count(*) from nlp2.politicsApp_nndata where `1`=0 and `2`= 0

create table nlp2.politicsApp_nndata_reduced as 
select * from nlp2.politicsApp_nndata where `1`!=0 or 
`2`!=0 or 
`3`!=0 or 
`4`!=0 or 
`5`!=0 or 
`6`!=0 or 
`7`!=0 or 
`8`!=0 or 
`9`!=0 or 
`10`!=0 or 
`11`!=0 or 
`12`!=0 or 
`13`!=0 or 
`14`!=0 or 
`15`!=0 or 
`16`!=0 or 
`17`!=0 or 
`18`!=0 or 
`19`!=0 or 
`20`!=0 or 
`21`!=0 or 
`22`!=0 or 
`23`!=0 or 
`24`!=0 or 
`25`!=0 or 
`26`!=0 or 
`27`!=0 or 
`28`!=0 or 
`29`!=0 or 
`30`!=0 or 
`31`!=0 or 
`32`!=0 or 
`33`!=0 or 
`34`!=0 or 
`35`!=0 or 
`36`!=0 or 
`37`!=0 or 
`38`!=0 or 
`39`!=0 or 
`40`!=0 or 
`41`!=0 or 
`42`!=0 or 
`43`!=0 or 
`44`!=0 or 
`45`!=0 or 
`46`!=0 or 
`47`!=0 or 
`48`!=0 or 
`49`!=0 or 
`50`!=0 or 
`51`!=0 or 
`52`!=0 or 
`53`!=0 or 
`54`!=0 or 
`55`!=0 or 
`56`!=0 or 
`57`!=0 or 
`58`!=0 or 
`59`!=0 or 
`60`!=0 or 
`61`!=0 or 
`62`!=0 or 
`63`!=0 or 
`64`!=0 or 
`65`!=0 or 
`66`!=0 or 
`67`!=0 or 
`68`!=0 or 
`69`!=0 or 
`70`!=0 or 
`71`!=0 or 
`72`!=0 or 
`73`!=0 or 
`74`!=0 or 
`75`!=0 or 
`76`!=0 or 
`77`!=0 or 
`78`!=0 or 
`79`!=0 or 
`80`!=0 or 
`81`!=0 or 
`82`!=0 or 
`83`!=0 or 
`84`!=0 or 
`85`!=0 or 
`86`!=0 or 
`87`!=0 or 
`88`!=0 or 
`89`!=0 or 
`90`!=0 or 
`91`!=0 or 
`92`!=0 or 
`93`!=0 or 
`94`!=0 or 
`95`!=0 or 
`96`!=0 or 
`97`!=0 or 
`98`!=0 or 
`99`!=0 or 
`100`!=0 or 
`101`!=0 or 
`102`!=0 or 
`103`!=0 or 
`104`!=0 or 
`105`!=0 or 
`106`!=0 or 
`107`!=0 or 
`108`!=0 or 
`109`!=0 or 
`110`!=0 or 
`111`!=0 or 
`112`!=0 or 
`113`!=0 or 
`114`!=0 or 
`115`!=0 or 
`116`!=0 or 
`117`!=0 or 
`118`!=0 or 
`119`!=0 or 
`120`!=0 or 
`121`!=0 or 
`122`!=0 or 
`123`!=0 or 
`124`!=0 or 
`125`!=0 or 
`126`!=0 or 
`127`!=0 or 
`128`!=0 or 
`129`!=0 or 
`130`!=0 or 
`131`!=0 or 
`132`!=0 or 
`133`!=0 or 
`134`!=0 or 
`135`!=0 or 
`136`!=0 or 
`137`!=0 or 
`138`!=0 or 
`139`!=0 or 
`140`!=0 or 
`141`!=0 or 
`142`!=0 or 
`143`!=0 or 
`144`!=0 or 
`145`!=0 or 
`146`!=0 or 
`147`!=0 or 
`148`!=0 or 
`149`!=0 or 
`150`!=0 or 
`151`!=0 or 
`152`!=0 or 
`153`!=0 or 
`154`!=0 or 
`155`!=0 or 
`156`!=0 or 
`157`!=0 or 
`158`!=0 or 
`159`!=0 or 
`160`!=0 or 
`161`!=0 or 
`162`!=0 or 
`163`!=0 or 
`164`!=0 or 
`165`!=0 or 
`166`!=0 or 
`167`!=0 or 
`168`!=0 or 
`169`!=0 or 
`170`!=0 or 
`171`!=0 or 
`172`!=0 or 
`173`!=0 or 
`174`!=0 or 
`175`!=0 or 
`176`!=0 or 
`177`!=0 or 
`178`!=0 or 
`179`!=0 or 
`180`!=0 or 
`181`!=0 or 
`182`!=0 or 
`183`!=0 or 
`184`!=0 or 
`185`!=0 or 
`186`!=0 or 
`187`!=0 or 
`188`!=0 or 
`189`!=0 or 
`190`!=0 or 
`191`!=0 or 
`192`!=0 or 
`193`!=0 or 
`194`!=0 or 
`195`!=0 or 
`196`!=0 or 
`197`!=0 or 
`198`!=0 or 
`199`!=0 or 
`200`!=0 or 
`201`!=0 or 
`202`!=0 or 
`203`!=0 or 
`204`!=0 or 
`205`!=0 or 
`206`!=0 or 
`207`!=0 or 
`208`!=0 or 
`209`!=0 or 
`210`!=0 or 
`211`!=0 or `212`!=0

select count(*) from nlp2.politicsApp_nndata where 0 in (select column_name from INFORMATION_SCHEMA.COLUMNS where table_name =  'politicsApp_nndata')
select column_name from INFORMATION_SCHEMA.columns where table_name =  'politicsApp_nndata'

select count(*) from nlp2.politicsApp_ngram where NgramSize = 1 or NgramSize=2
select count(*) from nlp2.politicsApp_ngramduplicates 
where NgramSize_D = 1 or NgramSize_D=2


select * from nlp2.politicsApp_nndata_latest