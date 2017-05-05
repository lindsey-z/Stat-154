library(tidytext) 
library(dplyr)
library(tidyr)
library(ggplot2)
library(magrittr)
library(glmnet)
library(tm)
library(caret)
library(e1071)
library(party)
library(rminer)
library(lsa)
library(doMC)
library(doParallel)
library(MASS)
library(xgboost)  
library(archdata) 
library(rJava)
library(RWeka)
library(text2vec)
library(SnowballC)
library(stringr)

setwd("~/Desktop/Stat 154/Yelp/Rawdata")

reviews = read.csv2("yelp_academic_dataset_review_train.csv", header=TRUE, sep=",", stringsAsFactors = FALSE)
reviews$text = as.vector(reviews$text)
reviews$text = str_replace_all(reviews$text, "[\r\n]" , "")
reviews$text = gsub('[[:punct:] ]+',' ',reviews$text)
reviews$text = str_replace_all(reviews$text, "[\r\n]", "")
reviews$text = gsub('[[:punct:] ]+',' ',reviews$text)

business = read.csv2("yelp_academic_dataset_business_train.csv", header=TRUE, sep=",", stringsAsFactors = TRUE)
business_id_stars = business[,c(3,14)]
options(digits=2)
business_id_stars$stars = as.character(business_id_stars$stars)
business_id_stars$stars = as.numeric(business_id_stars$stars)


tips = read.csv2("yelp_academic_dataset_tip.csv", header=TRUE, sep=",", stringsAsFactors = FALSE)

#----------------------------------------------------------------------------------------------------------------------------------
mean_star = c()
word_count_star_uni = list()
word_count_star_bi = list()
word_count_star_tri = list()
for(i in 1:5){
  star = reviews %>% 
    filter(stars == i) %>% 
    dplyr::select(X,text)
  star_unnest = star %>% 
    unnest_tokens(word, text) 
  star_length = star_unnest %>%
    group_by(X) %>%
    count(X) 
  mean_star[i] = mean(star_length$n)
  word_count_star_uni[[i]] = star_unnest %>%
    filter(!word %in% stop_words$word) %>%
    count(word, sort = TRUE)
  word_count_star_bi[[i]] = star %>%
    unnest_tokens(bigram, text, token = "ngrams", n=2) %>%
    separate(bigram, c("word1", "word2"), sep = " ") %>%
    filter(!word1 %in% stop_words$word) %>%
    filter(!word2 %in% stop_words$word) %>%
    count(word1, word2, sort = TRUE)
  word_count_star_tri[[i]] = star %>%
    unnest_tokens(ngram, text, token = "ngrams", n=3) %>%
    separate(ngram, c("word1", "word2","word3"), sep = " ") %>%
    filter(!word1 %in% stop_words$word) %>%
    filter(!word2 %in% stop_words$word) %>%
    filter(!word3 %in% stop_words$word) %>%
    count(word1, word2, word3, sort = TRUE)
}

#----------------------------------------------------------------------------------------------------------------------------------

doc_length = reviews %>% 
  dplyr::select(X,text) %>%
  unnest_tokens(word, text) %>%
  group_by(X) %>%
  count(X) 

par(mfrow=c(2,2))

fit_length = glm(reviews$stars ~ doc_length$n)
summary(fit_length)
plot(fit_length$fitted.values,fit_length$residuals)

fit_length2 = glm(reviews$stars ~ doc_length$n + I(doc_length$n*doc_length$n))
summary(fit_length2)
plot(fit_length2$fitted.values,fit_length2$residuals)

fit_length3 = glm(reviews$stars ~ doc_length$n + I(doc_length$n*doc_length$n) +I(doc_length$n*doc_length$n*doc_length$n))
summary(fit_length3)
plot(fit_length3$fitted.values,fit_length3$residuals)

fit_length_sqrt = glm(reviews$stars ~ sqrt(doc_length$n))
summary(fit_length_sqrt)
plot(fit_length_sqrt$fitted.values,fit_length_sqrt$residuals)


#----------------------------------------------------------------------------------------------------------------------------------
AFINN <- get_sentiments("afinn")
reviews_unigrams_sentiment = reviews %>%
  dplyr::select(X,text,stars) %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word) %>%
  inner_join(AFINN, by = "word") %>%
  group_by(X, stars) %>%
  summarize(sentiment = mean(score))

ggplot(reviews_unigrams_sentiment, aes(stars, sentiment, group = stars)) +
  geom_boxplot() +
  ylab("Average sentiment score")


#----------------------------------------------------------------------------------------------------------------------------------
#count sentiment 

AFINN <- get_sentiments("afinn")
AFINN_sub = AFINN[abs(AFINN$score)>=1,1]
AFINN_sub = AFINN_sub$word

sentiment_word_count_star_uni = list()
sentiment_word_count_star_bi = list()

for(i in 1:5){
  star = reviews %>% 
    filter(stars == i) %>% 
    dplyr::select(X,text)
  star_unnest = star %>% 
    unnest_tokens(word, text) 
  sentiment_word_count_star_uni[[i]] = star_unnest %>%
    filter(!word %in% stop_words$word) %>%
    filter(word %in% AFINN_sub) %>%
    count(word, sort = TRUE)
  sentiment_word_count_star_bi[[i]] = star %>%
    unnest_tokens(bigram, text, token = "ngrams", n=2) %>%
    separate(bigram, c("word1", "word2"), sep = " ") %>%
    filter(!word1 %in% stop_words$word) %>%
    filter(!word2 %in% stop_words$word) %>%
    filter(word1 %in% AFINN_sub | word2 %in% AFINN_sub) %>%
    count(word1, word2, sort = TRUE)
}

#----------------------------------------------------------------------------------------------------------------------------------
table(reviews$useful)
table(reviews$cool)

table(reviews$stars[reviews$cool!=0])
table(reviews$stars[reviews$useful!=0])

for(i in 1:30){
  print(mean(reviews$stars[reviews$cool==i]))
}

reviews_cool = reviews[,c("cool","stars")]
reviews_cool$cool = ifelse(reviews_cool$cool>0,as.logical(1),as.logical(0))
fit_cool = glm(stars ~ ., data= reviews_cool)
summary(fit_cool)
plot(fit_cool$fitted.values,fit_cool$residuals)


reviews_useful = reviews[,c("useful","stars")]
reviews_useful$useful = ifelse(reviews_useful$useful>0,as.logical(1),as.logical(0))
fit_useful = glm(stars ~ ., data= reviews_useful)
summary(fit_useful)
plot(fit_useful$fitted.values,fit_useful$residuals)



#----------------------------------------------------------------------------------------------------------------------------------
detach(package:plyr)
tips_sentiment = tips %>%
  dplyr::select(business_id,text) %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word) %>%
  inner_join(AFINN, by = "word") %>%
  group_by(business_id) %>%
  summarize(sentiment = mean(score))

tips_sentiment_join = plyr::join(tips_sentiment,business_id_stars, by="business_id")
tips_sentiment_join = tips_sentiment_join[-which(is.na(tips_sentiment_join$stars)),]

quartz()
ggplot(tips_sentiment_join, aes(stars, sentiment, group = stars)) +
  geom_boxplot() +
  ylab("Average sentiment score")
