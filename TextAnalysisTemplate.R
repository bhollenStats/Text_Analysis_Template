######################################################################################
# Basic End-to-End Text Analysis Template with Usenet News Data
# Data for this example available at http://qwone.com/~jason/20Newsgroups
# The data was extracted to a parallel folder to this project named in the 
# variable training_folder within the code
######################################################################################
library(dplyr)
library(tidyr)
library(purrr)
library(readr)
library(ggplot2)
library(stringr)
library(tidytext)
library(widyr)
library(ggraph)
library(igraph)
library(topicmodels)

######################################################################################
# 1. Input your data, clean it, and then tidy it up for analysis
######################################################################################
read_folder <- function(infolder) {
  data_frame(file=dir(infolder, full.names = TRUE)) %>%
    mutate(text = map(file, read_lines)) %>%
    transmute(id = basename(file), text) %>%
    unnest(text)
}

training_folder <- '../data/20news-bydate-train'

raw_text <- data_frame(folder = dir(training_folder, full.names = TRUE)) %>%
  unnest(map(folder, read_folder)) %>%
  transmute(newsgroup = basename(folder), id, text)

cleaned_text <- raw_text %>%
  group_by(newsgroup, id) %>%
  filter(cumsum(text=='') > 0,
         cumsum(str_detect(text,"^--")) == 0) %>%
  ungroup()

cleaned_text <- cleaned_text %>%
  filter(str_detect(text, "^[^>]+[A-Za-z\\d]"),
         str_detect(text, ""),
         !str_detect(text, "writes(:|\\.\\.\\.)$"),
         !str_detect(text, "^In article <"),
         !id %in% c(9704,9985))

usenet_words <- cleaned_text %>%
  unnest_tokens(word, text) %>%
  filter(str_detect(word, "[a-z']$"), 
         !word %in% stop_words$word)

######################################################################################
# Tidy'd body of words: newsgroup [attribute], id of message [attribute], word 
######################################################################################
# > usenet_words
# # A tibble: 710,358 x 3
#    newsgroup   id    word     
#    <chr>       <chr> <chr>    
#  1 alt.atheism 49960 archive  
#  2 alt.atheism 49960 atheism  
#  3 alt.atheism 49960 resources
#  4 alt.atheism 49960 alt      
#  5 alt.atheism 49960 atheism  
#  6 alt.atheism 49960 archive  
#  7 alt.atheism 49960 resources
#  8 alt.atheism 49960 modified 
#  9 alt.atheism 49960 december 
# 10 alt.atheism 49960 version  
# # ... with 710,348 more rows
######################################################################################

######################################################################################
# 2. Determining tf-idf within values of an attribute [newsgroup]
######################################################################################
words_by_newsgroup <- usenet_words %>%
  count(newsgroup, word, sort=TRUE) %>% # [newsgroup] == attribute
  ungroup()

tf_idf <- words_by_newsgroup %>%
  bind_tf_idf(word, newsgroup, n) %>%
  arrange(desc(tf_idf))

######################################################################################
# 2.a. Single out one attribute [newsgroup] and graph the top words
######################################################################################
tf_idf %>%
  filter(str_detect(newsgroup, "^sci\\.")) %>%
  group_by(newsgroup) %>%
  top_n(12, tf_idf) %>%
  ungroup() %>%
  ggplot(aes(word, tf_idf, fill=newsgroup)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~newsgroup, scales = "free") +
  ylab('tf_idf') + 
  coord_flip()

######################################################################################
# 3. Pairwise Correlation of Words
######################################################################################
newsgroup_cors <- words_by_newsgroup %>%
  pairwise_cor(newsgroup, word, n, sort = TRUE)

######################################################################################
# 3.a. Search for stronger correlations among attributes and view them in a network
######################################################################################
set.seed(2017)
newsgroup_cors %>%
  filter(correlation > 0.4) %>% # graph only data in this data with a correlation greater than 0.4
  graph_from_data_frame() %>%
  ggraph(layout = "fr") +
  geom_edge_link(aes(alpha = correlation, width = correlation)) +
  geom_node_point(size = 6, color = 'lightblue') +
  geom_node_text(aes(label = name), repel = TRUE) +
  theme_void()

######################################################################################
# 4. Topic Modeling with the Latent Dirichlet Allocation ("LDA")
######################################################################################

######################################################################################
# 4.a. Divide the messages into a series of groups ()
######################################################################################
word_sci_newsgroups <- usenet_words %>%
  filter(str_detect(newsgroup, "^sci")) %>%
  group_by(word) %>%
  mutate(word_total = n()) %>%
  ungroup() %>%
  filter(word_total > 50)

######################################################################################
# 4.b. Convert results into a document-term matrix
######################################################################################
sci_dtm <- word_sci_newsgroups %>%
  unite(document, newsgroup, id) %>%
  count(document, word) %>%
  cast_dtm(document, word, n)

######################################################################################
# 4.c. Divide input using the LDA algoritm into n[4] topics
######################################################################################
sci_lda <- LDA(sci_dtm, k = 4, control = list(seed = 2016))

######################################################################################
# 4.d. Visualize the output from the algorithm to determine how well it fits
######################################################################################
sci_lda %>%
  tidy() %>% # default matrix = 'beta'
  group_by(topic) %>%
  top_n(8, beta) %>%
  ungroup() %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) + 
  facet_wrap(~topic, scales = "free_y") +
  coord_flip()

######################################################################################
# 4.e. Box plot the gamma matrix from each topic for further insight
######################################################################################
sci_lda %>%
  tidy(matrix = "gamma") %>%
  separate(document, c("newsgroup", "id"), sep = "_") %>%
  mutate(newsgroup = reorder(newsgroup, gamma * topic)) %>%
  ggplot(aes(factor(topic), gamma)) +
  geom_boxplot() + 
  facet_wrap(~newsgroup) +
  labs(x = "Topic",
       y = "Number of messages where this was the highest percent topic")

######################################################################################
# 5.a. Sentiment analysis
######################################################################################
newsgroup_sentiments <- words_by_newsgroup %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(newsgroup) %>%
  summarize(score = sum(score * n) / sum(n))

newsgroup_sentiments %>%
  mutate(newsgroup = reorder(newsgroup, score)) %>%
  ggplot(aes(newsgroup, score, fill = score > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  ylab("Average sentiment score")

######################################################################################
# 5.b. Sentiment analysis by word
######################################################################################
contributions <- usenet_words %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(word) %>%
  summarize(occurences = n(),
            contribution = sum(score))

contributions %>%
  top_n(25, abs(contribution)) %>%
  mutate(word = reorder(word, contribution)) %>%
  ggplot(aes(word, contribution, fill = contribution > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip()
  
######################################################################################
# 5.c. Sentiment analysis per newsgroup
######################################################################################
top_sentiment_words <- words_by_newsgroup %>%
  inner_join(get_sentiments("afinn"), by="word") %>%
  mutate(contribution = score * n / sum(n))

target_newsgroups <- c('talk.politics.guns',
                        'talk.politics.mideast',
                        'talk.politics.misc',
                        'alt.atheism',
                        'talk.religion.misc',
                        'misc.forsale')

top_sentiment_words %>%
  filter(newsgroup %in% target_newsgroups) %>%
  group_by(newsgroup) %>%
  top_n(10, abs(contribution)) %>%
  ungroup() %>%
  mutate(word = reorder(word, contribution)) %>%
  ggplot(aes(word, contribution, fill = contribution > 0)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~newsgroup, scales = "free_y") +
  coord_flip()
