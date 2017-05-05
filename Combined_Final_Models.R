# Combine all models

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
## cnn is the random forest train model
cnn = read.csv('cnn.csv', header=TRUE, sep=",", stringsAsFactors = TRUE)
cnn = as.matrix(cnn)
rf_train = randomForest(stars ~ .,cnn, ntree=500)

cn = read.csv('cn.csv', header=TRUE, sep=",", stringsAsFactors = TRUE)
rf_prediction_df = data.frame(business_id=cn$business_id, rf_prediction = rf_train$predicted)


#----------------------------------------------------------------------------------------------------------------------------------

combined = plyr::join(mean_pred_svd_bin_multinomial_in_sample,tips_sentiment,by="business_id")
combined$sentiment = ifelse(is.na(combined$sentiment),
                            combined$mean_pred ,combined$sentiment)


combined = plyr::join(combined,rf_prediction_df,by="business_id")
combined$rf_prediction = ifelse(is.na(combined$rf_prediction),
                                combined$mean_pred,
                                combined$rf_prediction)

lm_fit = lm(stars ~ . ,combined[,-1])


#----------------------------------------------------------------------------------------------------------------------------------
tes_business = read.csv('test.csv',header=TRUE,stringsAsFactors = TRUE)
hours_pred = read.csv('test_business_pred_hours.csv', header=TRUE,stringsAsFactors = TRUE)
hours = read.csv('test_business_hours.csv', header=TRUE,stringsAsFactors = TRUE)
hours = hours[,-1]
hours_bind = rbind(hours,hours_pred)

test_bus_att = read.csv2('business_attributes_test_clean.csv', header=TRUE, sep=",", stringsAsFactors = TRUE)
test_bus_att = test_bus_att[,-1]

dmy1 = dummyVars(" ~ .", data = test_bus_att[,-1])
test_bus_att_dummy = data.frame(predict(dmy1, newdata=test_bus_att[,-1]))
test_bus_att_dummy$business_id = test_bus_att_dummy$business_id
test_bus_att_dummy = test_bus_att_dummy[,-1]
test_bus_att_dummy$business_id = test_bus_att$business_id
test_bus_att_dummy = as.data.frame(test_bus_att_dummy)

tes_business = tes_business[,c(4,10,11)]

tes_business = plyr::join(tes_business,hours_bind,by="business_id")
test_categories = read.csv2("Business_Category_Test2.csv", header=TRUE, sep=",", stringsAsFactors = TRUE)

tes_business = plyr::join(tes_business,test_categories,by="business_id")

dmy2 = dummyVars(" ~ .", data = tes_business[,-1])
test_cities_dummy = data.frame(predict(dmy2, newdata=tes_business[,-1]))
test_cities_dummy$business_id = tes_business$business_id


test_attr_cities = plyr::join(test_bus_att_dummy,test_cities_dummy,by="business_id")

test_attr_cities = test_attr_cities[,-29]

colnames(test_attr_cities)[c(17:26,28,29,30,31)] = colnames(cnn)[c(1:10,12,14,15,13)]


rf_test = predict(rf_train, test_attr_cities)
rf_df = data.frame(business_id = test_attr_cities$business_id, rf_test)
#----------------------------------------------------------------------------------------------------------------------------------
reviews_test = mean_pred_svd_bin_multinomial
test_review_tips = plyr::join(reviews_test,tips_sentiment, by="business_id")
test_review_tips_rf = plyr::join(test_review_tips,rf_df, by="business_id")

test_review_tips_rf$sentiment = ifelse(is.na(test_review_tips_rf$sentiment),
                                       test_review_tips_rf$mean_pred ,test_review_tips_rf$sentiment)


test_review_tips_rf$rf_test = ifelse(is.na(test_review_tips_rf$rf_test),
                                       test_review_tips_rf$mean_pred ,test_review_tips_rf$rf_test)
colnames(test_review_tips_rf)[4] = "rf_prediction"

predict_combined_all = predict(lm_fit,test_review_tips_rf)

index = which(!test_review_tips_rf$business_id %in% tes_business$business_id)
predict_combined_all = predict_combined_all[-index]
predict_combined_all_df = data.frame(business_id = test_review_tips_rf$business_id[-index], stars = predict_combined_all)

write.csv(predict_combined_all_df,"prediction_combined_all.csv", row.names=FALSE)
