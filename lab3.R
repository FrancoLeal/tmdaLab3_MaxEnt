require('maxent')
require('tm')
require('SnowballC')
pathroot = "/home/nicolas/Escritorio/USACH/Topicos/Taller de mineria de datos avanzada/tmdaLab3_MaxEnt/"
#pathroot =~"/Documentos/2-2019/TMDA/tmdaLab3/"
path1 = paste(pathroot,"data/amazon_cells_labelled.txt",sep="")
path2 = paste(pathroot,"/data/imdb_labelled.txt",sep="")
path3 = paste(pathroot,"/data/yelp_labelled.txt",sep="")
#path = "~/Escritorio/USACH/Topicos/Taller de mineria de datos avanzada/tmdaLab2_randomForest/breast-cancer-wisconsin.data"

data = read.delim(path1,header = FALSE)
colnames = c('text','class')
colnames(data) = colnames

dataAux = read.delim(path2,header=FALSE)
colnames(dataAux) = colnames

data = rbind(data,dataAux)
dataAux = read.delim(path3,header=FALSE)
colnames(dataAux) = colnames

data = rbind(data,dataAux)

smp_size = floor(0.65*nrow(data))
set.seed(12345)

train_ind = sample(seq_len(nrow(data)),size = smp_size)

data.train = data[train_ind,]
data.test = data[-train_ind,]

corpus = Corpus(VectorSource(data.train$text))


summary(corpus)

for(i in 1:10) print(corpus[[i]]$content)

corpus = tm_map(corpus,content_transformer(removePunctuation))
for(i in 1:10) print(corpus[[i]]$content)

corpus = tm_map(corpus,content_transformer(tolower))
for(i in 1:10) print(corpus[[i]]$content)

corpus = tm_map(corpus,content_transformer(removeWords),stopwords("english"))
for(i in 1:10) print(corpus[[i]]$content)

corpus = tm_map(corpus,stemDocument)
for(i in 1:10) print(corpus[[i]]$content)

corpus = tm_map(corpus,stripWhitespace)
for(i in 1:10) print(corpus[[i]]$content)

corpus = tm_map(corpus,content_transformer(removeNumbers))
for(i in 1:10) print(corpus[[i]]$content)

matrix = DocumentTermMatrix(corpus)
sparse <- as.compressed.matrix(matrix)

data.me.t <- tune.maxent(sparse,data.train$class,nfold=3,showall=TRUE, verbose=TRUE)
print(data.me.t)

model<-maxent(sparse,data$class, l1_regularizer=0,6, use_sgd=FALSE, set_heldout=0, verbose=TRUE)


corpus.test = Corpus(VectorSource(data.test$text))


summary(corpus.test)

for(i in 1:10) print(corpus.test[[i]]$content)

corpus.test = tm_map(corpus.test,content_transformer(removePunctuation))
for(i in 1:10) print(corpus.test[[i]]$content)

corpus.test = tm_map(corpus.test,content_transformer(tolower))
for(i in 1:10) print(corpus[[i]]$content)

corpus.test = tm_map(corpus.test,content_transformer(removeWords),stopwords("english"))
for(i in 1:10) print(corpus[[i]]$content)

corpus.test = tm_map(corpus.test,stemDocument)
for(i in 1:10) print(corpus[[i]]$content)

corpus.test = tm_map(corpus.test,stripWhitespace)
for(i in 1:10) print(corpus[[i]]$content)

corpus.test = tm_map(corpus.test,content_transformer(removeNumbers))
for(i in 1:10) print(corpus.test[[i]]$content)

matrix.test = DocumentTermMatrix(corpus.test)
sparse.test <- as.compressed.matrix(matrix.test)

results <- as.data.frame(predict(model,sparse.test))

options(digits=15)
results$predicted <- 0
results$predicted[as.double(results$"0") > as.double(results$"1")] <- 0
results$predicted[as.double(results$"0") < as.double(results$"1")] <- 1
print(results)
