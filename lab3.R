require('maxent')
require('tm')
require('SnowballC')

path1 = "~/Documentos/2-2019/TMDA/tmdaLab3/data/amazon_cells_labelled.txt"
path2 = "~/Documentos/2-2019/TMDA/tmdaLab3/data/imdb_labelled.txt"
path3 = "~/Documentos/2-2019/TMDA/tmdaLab3/data/yelp_labelled.txt"
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

corpus = Corpus(VectorSource(data$text))

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
