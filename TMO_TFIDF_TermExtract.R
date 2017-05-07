install.packages('RMySQL', repos='http://cran.us.r-project.org')
install.packages("tm", repos='http://cran.us.r-project.org')
install.packages("reshape2", repos='http://cran.us.r-project.org')
install.packages("data.table", repos='http://cran.us.r-project.org')
install.packages("RWeka", repos='http://cran.us.r-project.org')
install.packages("data.table", repos='http://cran.us.r-project.org')
install.packages("stringr", repos='http://cran.us.r-project.org')
install.packages("qdap", repos='http://cran.us.r-project.org')
require(RWeka)
require(tm)
require(reshape2)
require(data.table)
require(stringr)
library("plyr")
require("qdap")
options(scipen=999)

#QA file count
(n_fields <- count.fields("/Users/tganka/Desktop/twitter/att_20160501_20160507.csv"))

#Load file and stopwords
text <- read.csv("/Users/tganka/Desktop/twitter/att_20160501_20160507.csv", sep=",", header=TRUE, quote = "")
text$tweet<-bracketX(text$tweet)
text$tweet<-rm_url(text$tweet)
text$tweet<-rm_twitter_url(text$tweet)
text.cl <- na.omit(text)
stopWords <- scan("/Users/tganka/Desktop/twitter/stopWords_v2.txt", character(0))
myReader <- readTabular(mapping=list(content="tweet", id="id"))
Corpus <- Corpus(DataframeSource(text.cl), readerControl=list(reader=myReader))

#pre-process corpus

corp <- tm_map(Corpus, content_transformer(tolower)) 
corp <- tm_map(corp, removePunctuation)
corp <- tm_map(corp, removeWords, stopWords)

#Create a term document matrix
dtm <- DocumentTermMatrix(corp, control = list(weighting = weightTfIdf))
rowTotals <- apply(dtm , 1, sum)
dtm   <- dtm[rowTotals> 0, ]  

#remove sparseTerms
dtm <- removeSparseTerms(dtm, 0.999)

#Transform to DF and reshape
m.dtm <- as.matrix(dtm)
melt.dtm <- melt(m.dtm)
df.melt.dtm <- as.data.frame(melt.dtm)
head(df.melt.dtm,n=10)
colnames(df.melt.dtm) <- c("id","term","tfidf")

#Grab top 3 TFIDF per session Unigram
tfidf<-setDT(df.melt.dtm)[order(session_id,-tfidf), .SD[1:2], by=session_id]
tfidf.cl <- tfidf[!tfidf == 0]
termCnt <- count(tfidf.cl, c("term"))
head(arrange(termCnt, desc(freq)),n=10)

#add bigrams to TDM
options(mc.cores=1)
bigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
bi.dtm <- DocumentTermMatrix(corp, control = list(weighting = weightTfIdf, tokenize = bigramTokenizer))
inspect(bi.dtm[1000:1005, 10:15])
bi.dtm <- removeSparseTerms(bi.dtm, 0.999)

#Bigram top 3 TFIDF

findFreqTerms(bi.dtm, lowfreq=14, highfreq=Inf)

m.bi.dtm <- as.matrix(bi.dtm)
melt.bi.dtm <- melt(m.bi.dtm)
df.melt.bi.dtm <- as.data.frame(melt.bi.dtm)
head(df.melt.bi.dtm,n=10)
colnames(df.melt.bi.dtm) <- c("id","term","tfidf")

bi.tfidf<-setDT(df.melt.bi.dtm)[order(id,-tfidf), .SD[1:1], by=id]
bi.tfidf.cl <- bi.tfidf[!tfidf == 0]
bi.termCnt <- count(bi.tfidf.cl, c("term"))
head(arrange(bi.termCnt, desc(freq)),n=10)

#Create Top TFIDF for Uni/Bi per session
loc <- text.cl[,c("id","tweet")]
bi.tfidf.sess <- merge(loc,bi.tfidf,by="id")
write.table(bi.tfidf.sess, "/Users/tganka/Desktop/twitter/att_tfidf_20160501_20160507.csv", row.names=FALSE, sep=",")

#K-means clustering terms by day
k.bi.dtm <- removeSparseTerms(bi.dtm, 0.99)
km.bi.dtm <- as.matrix(k.bi.dtm)
cl <- kmeans(km.bi.dtm, sqrt(nrow(km.bi.dtm) / 2))


#Correlated words by top terms
toi <- "card reader"
corlimit <- 0.1
cardreader_0.1 <- as.data.frame(findAssocs(bi.dtm, toi, corlimit))

#SVM Classifier (Calls)
#training
text <- read.csv("C:/Users/tganka.TOWN/Documents/Classification/RBS/v0.2/Test/Customer/rbs_cus_test_callflg_20150413.txt", sep="\t", stringsAsFactors=FALSE, quote="")
colnames(text) <- c("session_id","text","call_flag")
stopWords_v1 <- scan("D:/tganka/stopWords_v1.txt", character(0))
text$text <- gsub("[^a-zA-Z]+", " ", text$text)
text <- as.vector(text)

#count calls
as.data.frame(table(text$call_flag))

createCorpus <- function(x){
	Corpus <- Corpus(VectorSource(text$text))
	tmp.corp <- tm_map(Corpus, content_transformer(tolower)) 
	tmp.corp <- tm_map(tmp.corp, content_transformer(removePunctuation))
	tmp.corp <- tm_map(tmp.corp, content_transformer(stripWhitespace))
	tmp.corp <- tm_map(tmp.corp, removeWords, stopWords_v1)
	final.corp <- tm_map(corp, stemDocument)
	dtm <- DocumentTermMatrix(corp, control = list(weighting = weightTf, normalize = TRUE))
	bigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
	bi.dtm <- DocumentTermMatrix(corp, control = list(weighting = weightTfIdf, tokenize = bigramTokenizer, normalize = TRUE))
	dtm <- removeSparseTerms(dtm, 0.998)
	bi.dtm <- removeSparseTerms(bi.dtm, 0.998)
	return(dtm)
}

createCorpus(text)
inspect(dtm[155:160,1:5])
#Modeling
require(RTextTools)
container <- create_container(dtm, text$call_flag, trainSize=1:70, testSize=71:100, virgin=FALSE)
svm <- train_model(container, "SVM", type='one-classification', nu=0.10, scale=TRUE, kernel="linear", cost=0.5, gamma=0.001)
svm_results <- classify_model(container, svm)
tree <- train_model(container, "TREE")
tree_results <- classify_model(container, tree)



#analytics
analytics <- create_analytics(container, cbind(svm_results,tree_results))

#QA
text[text$session_id == "4875984710481962624",]









#Testing
new_cohort_v1 <- as.data.frame(tf)
nlp.sub<-subset(new_cohort_v1, select = -c(Title,Language,Description,Keywords,Robots))
nlp_re<-melt(nlp.sub, id.var=c("session_id"))
tf<-setDT(nlp_re)[order(session_id,-value), .SD[1:1], by=session_id]
tf<-as.data.frame(tf)
tfidf<-as.data.frame(tfidf)
nlp.tf2<-merge(tfidf,tf,by="session_id")
colnames(nlp.tf2) <- c("session_id","tfidf_term","tfidf","tf_term","tf")
nlp.tf2<-nlp.tf2[c(1,2,4,3,5)]


library("openNLP")

acq <- "Gulf Applied Technologies Inc said it sold its subsidiaries engaged in
        pipeline and terminal operations for 12.2 mln dlrs. The company said 
        the sale is subject to certain post closing adjustments, 
        which it did not explain. Reuter." 

tagPOS <-  function(x, ...) {
  s <- as.String(x)
  word_token_annotator <- Maxent_Word_Token_Annotator()
  a2 <- Annotation(1L, "sentence", 1L, nchar(s))
  a2 <- annotate(s, word_token_annotator, a2)
  a3 <- annotate(s, Maxent_POS_Tag_Annotator(), a2)
  a3w <- a3[a3$type == "word"]
  POStags <- unlist(lapply(a3w$features, `[[`, "POS"))
  POStagged <- paste(sprintf("%s/%s", s[a3w], POStags), collapse = " ")
  list(POStagged = POStagged, POStags = POStags)
}

acqTag <- as.character(tagPOS(acq))
acqTagNNP <- sapply(strsplit(acqTag,"[[:punct:]]*/NNP.?"),function(x) {res = sub("(^.*\\s)(\\w+$)", "\\2", x); res[!grepl("\\s",res)]} ) 
