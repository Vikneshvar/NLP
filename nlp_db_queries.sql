
# Select rows
select * from nlp.politicsApp_ngram
 select * from nlp.politicsApp_articles
select * from nlp.politicsApp_articlengram
select count(*) from nlp.politicsApp_articlengram
select * from nlp.politicsApp_interaction
select count(*) from nlp.politicsApp_interaction
select * from nlp.politicsApp_nndata
select count(*) from nlp.politicsApp_nndata


select count(*) from nlp.politicsApp_articlengram where Frequency=1
select count(*) from nlp.politicsApp_articlengram where Frequency=2
select count(*) from nlp.politicsApp_articlengram where Frequency=4
select count(*) from nlp.politicsApp_articlengram where Frequency=6
select count(*) from nlp.politicsApp_articlengram where Frequency=8
select count(*) from nlp.politicsApp_articlengram where Frequency=10
select count(*) from nlp.politicsApp_articlengram where Frequency=14
select count(*) from nlp.politicsApp_articlengram where Frequency=18
select count(*) from nlp.politicsApp_articlengram where Frequency=22
select count(*) from nlp.politicsApp_articlengram where Frequency=30
select count(*) from nlp.politicsApp_articlengram where Frequency=40
select count(*) from nlp.politicsApp_articlengram where Frequency=55



# Check and Delete rows from table 

select * from nlp.django_migrations

select count(*) from nlp.django_migrations where app="politicsApp"

delete from nlp.django_migrations where id=17

# Drop table
drop table nlp.politicsApp_interaction
drop table nlp.`nlp.politicsApp_nndata`


# Delete all rows from table

truncate table nlp.politicsApp_articlengram
truncate table nlp.politicsApp_interaction

SET FOREIGN_KEY_CHECKS = 0; 
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
