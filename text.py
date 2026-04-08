library(tm)
library(topicmodels)
library(syuzhet)
library(slam)

data <- read.csv("C:/Users/HOME/Downloads/ADA/Shakespeare_data.csv", stringsAsFactors = FALSE)

docs <- na.omit(data$PlayerLine)

docs <- docs[1:5000]

corpus <- Corpus(VectorSource(docs))

corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)

dtm <- DocumentTermMatrix(corpus)

dtm <- removeSparseTerms(dtm, 0.99)

dtm <- dtm[row_sums(dtm) > 0, ]

lda_model <- LDA(dtm, k = 3)

print(terms(lda_model, 5))

sentiment <- get_nrc_sentiment(docs)

barplot(colSums(sentiment),
        las = 2,
        col = rainbow(10),
        main = "Shakespeare Sentiment")
