##whether we should include "cool" as a variable 

library(tidytext)  
library(plyr)
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

setwd("~/Desktop/Stat 154/Yelp/Rawdata")

reviews = read.csv2("yelp_academic_dataset_review_train.csv", header=TRUE, sep=",", stringsAsFactors = TRUE)
reviews$text = as.vector(reviews$text)
business = read.csv2("yelp_academic_dataset_business_train.csv", header=TRUE, sep=",", stringsAsFactors = TRUE)
business_id_stars = business[,c(3,14)]
options(digits=2)
business_id_stars$stars = as.character(business_id_stars$stars)
business_id_stars$stars = as.numeric(business_id_stars$stars)


tips = read.csv2("yelp_academic_dataset_tip.csv", header=TRUE, sep=",", stringsAsFactors = FALSE)
tips$text = as.vector(tips$text)

set.seed(200)
index = sample(1:2510, 2510*0.7)
train_business_id_stars = business_id_stars[index,]
test_business_id_stars = business_id_stars[-index,]

train_reviews = reviews[reviews$business_id %in% train_business_id_stars$business_id,]
test_reviews = reviews[!reviews$business_id %in% train_business_id_stars$business_id,]

train_reviews$text = as.vector(train_reviews$text)
test_reviews$text = as.vector(test_reviews$text)

#----------------------------------------------------------------------------------------------------------------------------------
stem_tokenizer <- function(x) {
  word_tokenizer(x) %>%
    lapply( function(x) SnowballC::wordStem(x, 'en'))
}

options(mc.cores=4)
train_token = itoken(train_reviews$text, 
                     preprocessor = tolower, 
                     tokenizer = stem_tokenizer, 
                     ids = train_reviews$X, 
                     progressbar = FALSE)

train_vocab = create_vocabulary(train_token, ngram = c(1L, 2L), sep_ngram = " ")

AFINN <- get_sentiments("afinn")
AFINN_sub = AFINN[abs(AFINN$score)>=1,1]
AFINN_sub = AFINN_sub$word
AFINN_sub = stem_tokenizer(AFINN_sub)
for(i in 1:length(AFINN_sub)){
  if(length(AFINN_sub[[i]])>1){
    AFINN_sub[[i]] = paste(AFINN_sub[[i]][1],AFINN_sub[[i]][2])
  }else{
    AFINN_sub[[i]] = AFINN_sub[[i]]
  }
}
AFINN_sub = do.call('rbind',AFINN_sub)
AFINN_sub = unique(AFINN_sub)

train_terms = train_vocab$vocab$terms
train_list = strsplit(train_terms," ")

dict_index = c()
registerDoMC(4)
registerDoParallel(4)
t1 = Sys.time()
for(i in 1:length(train_list)){
  dict_index[i] = any(train_list[[i]] %in% AFINN_sub)
}
print(difftime(Sys.time(), t1, units = 'min')) #17.47099 minutes

sentiment_terms = train_vocab$vocab$terms[dict_index]
non_sentiment_terms = train_vocab$vocab$terms[!dict_index]


train_vocab_sentiment = create_vocabulary(train_token, ngram = c(1L, 2L), stopwords = non_sentiment_terms , sep_ngram = " ")

train_pruned_vocab = prune_vocabulary(train_vocab_sentiment, 
                                      term_count_min = 15, 
                                      doc_proportion_max = 0.6,
                                      doc_proportion_min = 0.001)

train_vocab_vectorizer = vocab_vectorizer(train_pruned_vocab)

dtm_train = create_dtm(train_token, train_vocab_vectorizer)

test_token = itoken(test_reviews$text, 
                    preprocessor = tolower, 
                    tokenizer = stem_tokenizer, 
                    ids = test_reviews$X, 
                    progressbar = FALSE)

dtm_test = create_dtm(test_token, train_vocab_vectorizer)

#----------------------------------------------------------------------------------------------------------------------------------
## SVD with cool

SVD_bin_train = as.matrix(dtm_train)
SVD_bin_train = cbind(SVD_bin_train,sign(train_reviews$cool))
colnames(SVD_bin_train)[ncol(SVD_bin_train)] = "cool_variable"


SVD_bin_test= as.matrix(dtm_test)
SVD_bin_test = cbind(SVD_bin_test,sign(test_reviews$cool))
colnames(SVD_bin_test)[ncol(SVD_bin_test)] = "cool_variable"

SVD_bin_train = svd(SVD_bin_train)

options(digits=5)

plot(cumsum(SVD_bin_train$d^2)/sum(SVD_bin_train$d^2))
min_95 = min(which(cumsum(SVD_bin_train$d^2)/sum(SVD_bin_train$d^2)>0.95))

SVD_bin_reconstruct <- SVD_bin_train$u[,1:min_95] %*% diag(SVD_bin_train$d[1:min_95], min_95, min_95) %*% t(SVD_bin_train$v[,1:min_95])
colnames(SVD_bin_reconstruct) = colnames(SVD_bin_reconstruct)

SVD_bin_test = (SVD_bin_test %*% (SVD_bin_train$v[,1:min_95])) %*% t(SVD_bin_train$v[,1:min_95])

registerDoMC(4)
registerDoParallel(4)
t1 = Sys.time()
svd_bin_cv_multinomial_lasso<- cv.glmnet(SVD_bin_reconstruct, train_reviews$stars, family = "multinomial", nfolds = 5, alpha=1,
                                         type.multinomial = "grouped",
                                         # high value is less accurate, but has faster training
                                         thresh = 1e-4,
                                         # again lower number of iterations for faster training
                                         maxit = 1e4,
                                         parallel = TRUE)
print(difftime(Sys.time(), t1, units = 'min'))

svd_bin_multinomial_lasso_predict <-predict(svd_bin_cv_multinomial_lasso, s = svd_bin_cv_multinomial_lasso$lambda.min
                                            , SVD_bin_test, type="class")

svd_bin_multinomial_lasso_predict = as.numeric(as.factor(svd_bin_multinomial_lasso_predict))

reviews_with_cool_acc = sum(svd_bin_multinomial_lasso_predict == test_reviews$stars)/length(test_reviews$stars) #0.5463513
save(reviews_with_cool_acc, file="reviews_with_cool_acc.RData")

svd_bin_multinomial = data.frame(business_id = test_reviews[,c(6)], prediction = svd_bin_multinomial_lasso_predict)
mean_pred_svd_bin_multinomial = svd_bin_multinomial %>% 
  group_by(business_id)  %>% 
  dplyr::summarize(mean_pred = mean(prediction))
mean_pred_svd_bin_multinomial = as.data.frame(mean_pred_svd_bin_multinomial)
mean_pred_svd_bin_multinomial$business_id = as.character(mean_pred_svd_bin_multinomial$business_id)
mean_pred_svd_bin_multinomial_df = plyr::join(test_business_id_stars,mean_pred_svd_bin_multinomial, by="business_id")
reviews_with_cool_mse = mean((mean_pred_svd_bin_multinomial_df$stars-mean_pred_svd_bin_multinomial_df$mean_pred)^2) #0.23923

##------------------------------------------------------------------------------------------------------------------------------

## SVD without cool

SVD_bin_train = as.matrix(dtm_train)
SVD_bin_train = svd(SVD_bin_train)
SVD_bin_test= as.matrix(dtm_test)

options(digits=5)

plot(cumsum(SVD_bin_train$d^2)/sum(SVD_bin_train$d^2))
min_95 = min(which(cumsum(SVD_bin_train$d^2)/sum(SVD_bin_train$d^2)>0.95))

SVD_bin_reconstruct <- SVD_bin_train$u[,1:min_95] %*% diag(SVD_bin_train$d[1:min_95], min_95, min_95) %*% t(SVD_bin_train$v[,1:min_95])
colnames(SVD_bin_reconstruct) = colnames(SVD_bin_reconstruct)

SVD_bin_test = (SVD_bin_test %*% (SVD_bin_train$v[,1:min_95])) %*% t(SVD_bin_train$v[,1:min_95])

registerDoMC(4)
registerDoParallel(4)
t1 = Sys.time()
svd_bin_cv_multinomial_lasso<- cv.glmnet(SVD_bin_reconstruct, train_reviews$stars, family = "multinomial", nfolds = 5, alpha=1,
                                         type.multinomial = "grouped",
                                         # high value is less accurate, but has faster training
                                         thresh = 1e-4,
                                         # again lower number of iterations for faster training
                                         maxit = 1e4,
                                         parallel = TRUE)
print(difftime(Sys.time(), t1, units = 'min')) 

svd_bin_multinomial_lasso_predict <-predict(svd_bin_cv_multinomial_lasso, s = svd_bin_cv_multinomial_lasso$lambda.min
                                            , SVD_bin_test, type="class")

svd_bin_multinomial_lasso_predict = as.numeric(as.factor(svd_bin_multinomial_lasso_predict))

reviews_no_cool_acc = sum(svd_bin_multinomial_lasso_predict == test_reviews$stars)/length(test_reviews$stars) #0.54315
save(reviews_no_cool_acc, file="reviews_no_cool_acc.RData")

svd_bin_multinomial = data.frame(business_id = test_reviews[,c(6)], prediction = svd_bin_multinomial_lasso_predict)
mean_pred_svd_bin_multinomial = svd_bin_multinomial %>% 
  group_by(business_id)  %>% 
  dplyr::summarize(mean_pred = mean(prediction))
mean_pred_svd_bin_multinomial = as.data.frame(mean_pred_svd_bin_multinomial)
mean_pred_svd_bin_multinomial$business_id = as.character(mean_pred_svd_bin_multinomial$business_id)
mean_pred_svd_bin_multinomial_df = plyr::join(test_business_id_stars,mean_pred_svd_bin_multinomial, by="business_id")
reviews_no_cool_mse = mean((mean_pred_svd_bin_multinomial_df$stars-mean_pred_svd_bin_multinomial_df$mean_pred)^2) #0.24335
