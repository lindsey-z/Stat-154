#Final model (cool + doc length)

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

test_reviews = read.csv2("yelp_academic_dataset_review_test.csv", header=TRUE, sep=",", stringsAsFactors = TRUE)
test_reviews$text = as.vector(test_reviews$text)

#----------------------------------------------------------------------------------------------------------------------------------
stem_tokenizer <- function(x) {
  word_tokenizer(x) %>%
    lapply( function(x) SnowballC::wordStem(x, 'en'))
}

options(mc.cores=4)
dict_token = itoken(reviews$text, 
                    preprocessor = tolower, 
                    tokenizer = stem_tokenizer, 
                    ids = reviews$X, 
                    progressbar = FALSE)

dict_vocab = create_vocabulary(dict_token, ngram = c(1L, 2L), sep_ngram = " ")

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

dict_terms = dict_vocab$vocab$terms
dict_list = strsplit(dict_terms," ")

dict_index = c()
registerDoMC(4)
registerDoParallel(4)
t1 = Sys.time()
for(i in 1:length(dict_list)){
  dict_index[i] = any(dict_list[[i]] %in% AFINN_sub)
}
print(difftime(Sys.time(), t1, units = 'min')) #27 minutes

sentiment_terms = dict_vocab$vocab$terms[dict_index]
non_sentiment_terms = dict_vocab$vocab$terms[!dict_index]


dict_vocab_sentiment = create_vocabulary(dict_token, ngram = c(1L, 2L), stopwords = non_sentiment_terms , sep_ngram = " ")

dict_pruned_vocab = prune_vocabulary(dict_vocab_sentiment, 
                                     term_count_min = 15, 
                                     doc_proportion_max = 0.6,
                                     doc_proportion_min = 0.001)

dict_vocab_vectorizer = vocab_vectorizer(dict_pruned_vocab)

dtm_dict = create_dtm(dict_token, dict_vocab_vectorizer)


test_token = itoken(test_reviews$text, 
                    preprocessor = tolower, 
                    tokenizer = stem_tokenizer, 
                    ids = test_reviews$X, 
                    progressbar = FALSE)

dtm_test = create_dtm(test_token, dict_vocab_vectorizer)


doc_length_reviews = reviews %>% 
  dplyr::select(X,text) %>%
  unnest_tokens(word, text) %>%
  group_by(X) %>%
  count(X) 

doc_length_test = test_reviews %>% 
  dplyr::select(X,text) %>%
  unnest_tokens(word, text) %>%
  group_by(X) %>%
  count(X) 

#----------------------------------------------------------------------------------------------------------------------------------
#SVD Multinomial Lasso

mat_dict = as.matrix(dtm_dict)
mat_dict = cbind(mat_dict,sign(reviews$cool))
colnames(mat_dict)[ncol(mat_dict)] = "cool_variable"
mat_dict = cbind(mat_dict,sqrt(doc_length_reviews$n))
colnames(mat_dict)[ncol(mat_dict)] = "doc_length"
mat_dict[,ncol(mat_dict)] = scale(mat_dict[,ncol(mat_dict)])

mat_test= as.matrix(dtm_test)
mat_test = cbind(mat_test,sign(test_reviews$cool))
colnames(mat_test)[ncol(mat_test)] = "cool_variable"
mat_test = cbind(mat_test,sqrt(doc_length_test$n))
colnames(mat_test)[ncol(mat_test)] = "doc_length"
mat_test[,ncol(mat_test)] = scale(mat_test[,ncol(mat_test)])


SVD_bin_dict = svd(mat_dict)
SVD_bin_test = mat_test


options(digits=5)

plot(cumsum(SVD_bin_dict$d^2)/sum(SVD_bin_dict$d^2))
min_95 = min(which(cumsum(SVD_bin_dict$d^2)/sum(SVD_bin_dict$d^2)>0.95))

SVD_bin_reconstruct <- SVD_bin_dict$u[,1:min_95] %*% diag(SVD_bin_dict$d[1:min_95], min_95, min_95) %*% t(SVD_bin_dict$v[,1:min_95])
colnames(SVD_bin_reconstruct) = colnames(SVD_bin_reconstruct)

SVD_bin_test = (SVD_bin_test %*% (SVD_bin_dict$v[,1:min_95])) %*% t(SVD_bin_dict$v[,1:min_95])

registerDoMC(4)
registerDoParallel(4)
t1 = Sys.time()
svd_bin_cv_multinomial_lasso<- cv.glmnet(SVD_bin_reconstruct, reviews$stars, family = "multinomial", nfolds = 10, alpha=1,
                                         type.multinomial = "grouped",
                                         # high value is less accurate, but has faster training
                                         thresh = 1e-5,
                                         # again lower number of iterations for faster training
                                         maxit = 1e4,
                                         parallel = TRUE)
print(difftime(Sys.time(), t1, units = 'min')) #81.47 minutes

svd_bin_multinomial_lasso_predict <-predict(svd_bin_cv_multinomial_lasso, s = svd_bin_cv_multinomial_lasso$lambda.min
                                            , SVD_bin_test, type="class")
svd_bin_multinomial_lasso_predict = as.numeric(as.factor(svd_bin_multinomial_lasso_predict))

#aggregate prediction by business id

svd_bin_multinomial = data.frame(business_id = test_reviews[,c(6)], prediction = svd_bin_multinomial_lasso_predict)
detach(package:plyr)
mean_pred_svd_bin_multinomial = svd_bin_multinomial %>% 
  group_by(business_id)  %>% 
  dplyr::summarize(mean_pred = mean(prediction))
mean_pred_svd_bin_multinomial = as.data.frame(mean_pred_svd_bin_multinomial)
mean_pred_svd_bin_multinomial$business_id = as.character(mean_pred_svd_bin_multinomial$business_id)

write.csv(mean_pred_svd_bin_multinomial,"review_prediction_df.csv",row.names = FALSE)


#----------------------------------------------------------------------------------------------------------------------------------
#tips sentiment 

tips_sentiment = tips %>%
  dplyr::select(business_id,text) %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word) %>%
  inner_join(AFINN, by = "word") %>%
  group_by(business_id) %>%
  summarize(sentiment = mean(score))

#----------------------------------------------------------------------------------------------------------------------------------
#SVD Multinomial Lasso with tips In Sample


svd_bin_multinomial_lasso_predict_in_sample <-predict(svd_bin_cv_multinomial_lasso, s = svd_bin_cv_multinomial_lasso$lambda.min
                                                      , SVD_bin_reconstruct, type="class")
svd_bin_multinomial_lasso_predict_in_sample = as.numeric(as.factor(svd_bin_multinomial_lasso_predict_in_sample))
sum(svd_bin_multinomial_lasso_predict_in_sample == reviews$stars)/length(reviews$stars) #0.54663

svd_bin_multinomial_in_sample = data.frame(business_id = reviews$business_id, prediction = svd_bin_multinomial_lasso_predict_in_sample)
mean_pred_svd_bin_multinomial_in_sample = svd_bin_multinomial_in_sample %>% 
  group_by(business_id)  %>% 
  dplyr::summarize(mean_pred = mean(prediction))
mean_pred_svd_bin_multinomial_in_sample = as.data.frame(mean_pred_svd_bin_multinomial_in_sample)
mean_pred_svd_bin_multinomial_in_sample$business_id = as.character(mean_pred_svd_bin_multinomial_in_sample$business_id)
mean_pred_svd_bin_multinomial_in_sample = join(business_id_stars,mean_pred_svd_bin_multinomial_in_sample, by="business_id")
mean((mean_pred_svd_bin_multinomial_in_sample$stars-mean_pred_svd_bin_multinomial_in_sample$mean_pred)^2) #0.25538

pred_review_tips_df = join(mean_pred_svd_bin_multinomial_in_sample, tips_sentiment, by= "business_id")
pred_review_tips_df$sentiment = ifelse(is.na(pred_review_tips_df$sentiment),pred_review_tips_df$mean_pred,pred_review_tips_df$sentiment)

lm_pred_review_tips_df = lm(stars ~ mean_pred + sentiment, data=pred_review_tips_df)
plot(pred_review_tips_df$sentiment,lm_pred_review_tips_df$residuals)
plot(pred_review_tips_df$mean_pred,lm_pred_review_tips_df$residuals)
res1 = lm(stars ~ mean_pred, data=pred_review_tips_df)
res2 = lm(sentiment ~ mean_pred, data=pred_review_tips_df)
plot(res1$residuals,res2$residuals)
res1 = lm(stars ~ sentiment, data=pred_review_tips_df)
res2 = lm(mean_pred ~ sentiment, data=pred_review_tips_df)
plot(res1$residuals,res2$residuals)


pred_review_tips_df_in_sample <-predict(lm_pred_review_tips_df, pred_review_tips_df)
mean((mean_pred_svd_bin_multinomial_in_sample$stars-pred_review_tips_df_in_sample)^2) #0.17639

# Out of Sample

review_tips_out_sample = join(mean_pred_svd_bin_multinomial, tips_sentiment, by="business_id")
review_tips_out_sample$sentiment = ifelse(is.na(review_tips_out_sample$sentiment),review_tips_out_sample$mean_pred
                                          ,review_tips_out_sample$sentiment)
pred_review_tips_out_sample <-predict(lm_pred_review_tips_df, review_tips_out_sample)
lm_prediction_df = data.frame(business_id = review_tips_out_sample$business_id, stars = pred_review_tips_out_sample)


#Robust reg

rlm_pred_review_tips_df = rlm(stars ~ mean_pred + sentiment, data=pred_review_tips_df)

rlm_pred_review_tips_df_in_sample <-predict(rlm_pred_review_tips_df, pred_review_tips_df)
mean((mean_pred_svd_bin_multinomial_in_sample$stars-rlm_pred_review_tips_df_in_sample)^2) #0.17751

# Out of Sample

rlm_pred_review_tips_out_sample <-predict(rlm_pred_review_tips_df, review_tips_out_sample)

rlm_prediction_df = data.frame(business_id = review_tips_out_sample$business_id, stars = rlm_pred_review_tips_out_sample)

#----------------------------------------------------------------------------------------------------------------------------------
business_test = read.csv2("yelp_academic_dataset_business_test", header=TRUE, sep=",", stringsAsFactors = TRUE)
business_id_test = unique(business_test$business_id)

id_not_in_test = which(!lm_prediction_df$business_id %in% business_id_test)

lm_prediction_df = lm_prediction_df[-id_not_in_test,]

rlm_prediction_df = rlm_prediction_df[-id_not_in_test,]

write.csv(lm_prediction_df,"lm_prediction_df.csv",row.names = FALSE) #lm better
write.csv(rlm_prediction_df,"rlm_prediction_df.csv",row.names = FALSE)

