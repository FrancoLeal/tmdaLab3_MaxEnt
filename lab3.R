require('maxent')
require('tm')
require('SnowballC')
require('wordcloud')
pathroot = "/home/nicolas/Documents/USACH/TMDA/tmdaLab3_MaxEnt/"
#pathroot ="~ /Documentos/2-2019/TMDA/tmdaLab3/"
path1 = paste(pathroot,"data/amazon_cells_labelled.txt",sep="")
path2 = paste(pathroot,"data/imdb_labelled.txt",sep="")
path3 = paste(pathroot,"data/yelp_labelled.txt",sep="")
#path = "~/Escritorio/USACH/Topicos/Taller de mineria de datos avanzada/tmdaLab2_randomForest/breast-cancer-wisconsin.data"

data = read.delim(path1,header = FALSE)
colnames = c('text','class')
colnames(data) = colnames

dataAux = read.delim(path2,header=FALSE)
colnames(dataAux) = colnames

data = rbind(data,dataAux)
dataAux = read.delim(path3,header=FALSE)
  colnames(dataAux) = colnames

data = na.omit(rbind(data,dataAux))

data.positive = data[which(data$class == 1),]
data.negative = data[which(data$class == 0),]

transform_corpus <- function(corpusF) {
  corpusF = tm_map(corpusF,content_transformer(removePunctuation))
  
  corpusF = tm_map(corpusF,content_transformer(tolower))
  
  corpusF = tm_map(corpusF,content_transformer(removeWords),stopwords("english"))
  
  corpusF = tm_map(corpusF,stemDocument)
  
  corpusF = tm_map(corpusF,stripWhitespace)
  
  corpusF = tm_map(corpusF,content_transformer(removeNumbers))
  
  return(corpusF)
}


corpus = Corpus(VectorSource(data$text))


summary(corpus)


corpus = transform_corpus(corpus)

matrix = DocumentTermMatrix(corpus)
sparse <- as.compressed.matrix(matrix)

#Palabras frecuentes
matrix.aux = TermDocumentMatrix(corpus)

matrix.m <- as.matrix(matrix.aux)
matrix.sorted <- sort(rowSums(matrix.m),decreasing=TRUE)
#3963 tokens
data.df <- data.frame(word = names(matrix.sorted),freq=matrix.sorted)
wordcloud(data.df$word,data.df$freq,min.freq=30,scale=c(2.0,0.5), colors =brewer.pal(9,"Set1"),max.words =30)


corpus.positive = Corpus(VectorSource(data.positive$text))


corpus.positive = transform_corpus(corpus.positive)

matrix.positive = DocumentTermMatrix(corpus.positive)
sparse.positive <- as.compressed.matrix(matrix.positive)

#Palabras frecuentes
matrix.aux.positive = TermDocumentMatrix(corpus.positive)

matrix.m.positive <- as.matrix(matrix.aux.positive)
matrix.sorted.positive <- sort(rowSums(matrix.m.positive),decreasing=TRUE)
#3963 tokens
data.df.positive <- data.frame(word = names(matrix.sorted.positive),freq=matrix.sorted.positive)
wordcloud(data.df.positive$word,data.df.positive$freq,min.freq=35,scale=c(2.0,0.5), colors = brewer.pal(9,"Blues"),max.words =80)


corpus.negative = Corpus(VectorSource(data.negative$text))



corpus.negative = transform_corpus(corpus.negative)

matrix.negative = DocumentTermMatrix(corpus.negative)
sparse.negative <- as.compressed.matrix(matrix.negative)

#Palabras frecuentes
matrix.aux.negative = TermDocumentMatrix(corpus.negative)

matrix.m.negative <- as.matrix(matrix.aux.negative)
matrix.sorted.negative <- sort(rowSums(matrix.m.negative),decreasing=TRUE)
#3963 tokens
data.df.negative <- data.frame(word = names(matrix.sorted.negative),freq=matrix.sorted.negative)
wordcloud(data.df.negative$word,data.df.negative$freq,min.freq=35,scale=c(2.0,0.5), colors = brewer.pal(8,"Reds"),max.words =80)



smp_size = floor(0.65*nrow(data))
set.seed(92347)

train_ind = sample(seq_len(nrow(data)),size = smp_size)

data.train = data[train_ind,]
data.test = data[-train_ind,]

corpus.train.np = Corpus(VectorSource(data.train$text))

matrix.train.np = DocumentTermMatrix(corpus.train.np)
sparse.train.np <- as.compressed.matrix(matrix.train.np)

data.me.t.np <- tune.maxent(sparse.train.np,data.train$class,nfold=3,showall=TRUE, verbose=TRUE)
print(data.me.t.np)

model.np<-maxent(sparse.train.np,data.train$class, l2_regularizer=0.2, use_sgd=FALSE, set_heldout=0, verbose=TRUE)


corpus.train = transform_corpus(corpus.train.np)

matrix.train = DocumentTermMatrix(corpus.train)
sparse.train <- as.compressed.matrix(matrix.train)

data.me.t <- tune.maxent(sparse.train,data.train$class,nfold=3,showall=TRUE, verbose=TRUE)
print(data.me.t)

model<-maxent(sparse.train,data.train$class, l2_regularizer=0.2, use_sgd=FALSE, set_heldout=0, verbose=TRUE)


corpus.test.np = Corpus(VectorSource(data.test$text))

matrix.test.np = DocumentTermMatrix(corpus.test.np)
sparse.test.np <- as.compressed.matrix(matrix.test.np)

results.np <- as.data.frame(predict(model.np,sparse.test.np))

options(digits=15)
results.np$relevant <- "1.relevante"
results.np$relevant[which(results.np$label == 0)] <- "2.no relevante"
results.np$recuperado <- "2.no recuperado"
results.np$recuperado[which(data.test$class == results.np$label)] <- "1.recuperado"





table(results.np$recuperado,results.np$relevant)



corpus.test = transform_corpus(corpus.test.np)

matrix.test = DocumentTermMatrix(corpus.test)
sparse.test <- as.compressed.matrix(matrix.test)

results <- as.data.frame(predict(model,sparse.test))

options(digits=15)
results$relevant <- "1.relevante"
results$relevant[which(results$label == 0)] <- "2.no relevante"
results$recuperado <- "2.no recuperado"
results$recuperado[which(data.test$class == results$label)] <- "1.recuperado"





table(results$recuperado,results$relevant)

