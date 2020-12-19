# download if need
# nltk.download('punkt')
# pip install gensim
# pip install tensorflow==2.2
# pip install pyLDAvis
# import the libraries
import pandas as pd
import numpy as np
from datetime import datetime as dt
import pytz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import snscrape.modules.twitter as tw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
from pandas_datareader import data
import seaborn as sns
import nltk, re, string
from nltk.corpus import stopwords
import string
from nltk.tokenize import TweetTokenizer
from nltk.stem import *
from sklearn.preprocessing import normalize
import ngram
import pyLDAvis.gensim
import keras
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
import gensim as gs
from nltk.stem.wordnet import WordNetLemmatizer
import stylecloud
from itertools import chain
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# Set Input
# the query you want to search on twitter
twitterquery = 'from:#dowjones since:2015-12-01 until:2020-06-01'
# #dowjones is the tag you interested in, since and until is the time period you want

ric = 'DJIA'  # the RIC of the equity/ETF/index you want
# the date you select
start_date = '2015-12-01'
end_date = '2020-06-01'

# Save Twitter data to external file
twt_content = open('project.txt', 'w', encoding='utf-8')
for tweet in tw.TwitterSearchScraper(twitterquery).get_items():
    # Possible outputs: url, date, content, id, username, outlinks, outlinksss, tcooutlinks, tcooutlinksss
    date_str = tweet.date.strftime("%Y-%m-%d %H:%M:%S%z")
    date_str = date_str[:-2] + ":" + date_str[-2:]
    twt_content.write(date_str + "|" + tweet.content + "\n")
twt_content.close()

# Read download Twitter data
twet = []
dates = []
twt_content = open("project.txt", "r", encoding="utf-8")
for l in twt_content:
    line = l.split("|")
    date_str = line[0]
    try:
        date_time = dt.fromisoformat(date_str)
        date_time = date_time.astimezone(pytz.timezone("US/Eastern"))
        line[0] = date_time
        line[1] = line[1][:-1]
        twet.append(line)
        dates.append(date_time.date())
    except:
        twet[-1][1] += " " + l[:-1]
twt_content.close()

df = pd.DataFrame(data=twet, columns=['Time', 'Tweet', 'a', 'b', 'c', 'd', 'e', 'f', 'g'])
df1 = df.iloc[:, 0:2]  # there are 9 columns in df, we only need the first two columns
df2 = df.iloc[:, 0:2]
df1['Dates'] = dates
df2['Dates'] = dates

# Download stock data
stock_data = data.DataReader(ric, "yahoo", start_date, end_date)
equity = pd.DataFrame(data=stock_data['Adj Close'])
equity['log_ret'] = np.log(equity['Adj Close'].shift(-1)) - np.log(equity['Adj Close'])
equity = equity.iloc[:-1, ]

# Visualize the daily close price & log return
equity['Adj Close'].plot()
plt.grid()
plt.title('Daily Close Price')
plt.legend()

equity['log_ret'].plot()
plt.grid()
plt.title('Log Return')
plt.legend()

# Construct average sentiment on each day
prev_d = dt.fromisoformat(start_date).date()
sent_mean = []
for date in equity.index.values:
    d = pd.to_datetime(date).date()
    avg = np.mean(df1.loc[(df1.Dates <= d) & (df1.Dates > prev_d)].Sentiment)
    if np.isnan(avg):
        sent_mean.append(0)
    else:
        sent_mean.append(avg)
    prev_d = d
equity['sentiment'] = sent_mean

# Visualize sentiment mean value
equity['sentiment'].plot(figsize=(20, 5))
plt.title('Daily sentiment mean value')
plt.grid()

# Data clean
remove_handles = lambda x: re.sub('@[^\s]+', '', x)
remove_urls = lambda x: re.sub('http[^\s]+', '', x)
remove_hashtags = lambda x: re.sub('#[^\s]*', '', x)

df1['Tweet'] = df1['Tweet'].apply(remove_handles)
df1['Tweet'] = df1['Tweet'].apply(remove_urls)
df1['Tweet'] = df1['Tweet'].apply(remove_hashtags)

# Calculate the average length of tweets
df1['text_len'] = df1['Tweet'].map(lambda x: len(str(x)))
print('The average length of tweets is:', df1['text_len'].mean())

twt_len = pd.DataFrame(df1, columns=('Tweet', 'text_len'))

# Plot the distribution of tweet text length
sns.displot(twt_len['text_len'], bins=200)
plt.title("Tweet length distribution", size=20)

# Delete stop words
stop_words = stopwords.words('english')
stop_words += ["The", "I", "...", "S", "'", "It", "..", " ", "‘"]


def get_tokens(doc):
    tokens = [token.strip() \
              for token in nltk.word_tokenize(str(doc).lower()) \
              if token.strip() not in stop_words and \
              token.strip() not in string.punctuation]
    return tokens


def tokenize(text):
    tokens = None
    pattern = r'[a-zA-Z][-._a-zA-Z]*[a-zA-Z]'
    tokens = [token for token in nltk.regexp_tokenize(str(text).lower(), pattern) if
              token not in stop_words]
    str1 = " ".join(tokens)
    return str1


df1['text'] = df1.Tweet.apply(tokenize)


# Use N-Grams

def get_ngrams(doc, n=None):
    vectorizer = CountVectorizer(ngram_range=(n, n)).fit(doc)
    bag_of_words = vectorizer.transform(doc)
    sum_of_words = bag_of_words.sum(axis=0)
    word_counts = [(word, sum_of_words[0, index])
                   for word, index in vectorizer.vocabulary_.items()
                   ]
    word_counts = sorted(word_counts, key=lambda x: x[1], reverse=True)
    return word_counts


# Get n-grams
top_bigrams = get_ngrams(df1['text'], 2)[:20]
top_trigrams = get_ngrams(df1['text'], 3)[:20]


# Hot word visualization
def preprocess_tweet(df: pd.DataFrame, stop_words: None):
    processed_tweets = []
    tokenizer = TweetTokenizer()
    for text in df['Tweet']:
        words = [w for w in tokenizer.tokenize(text) if (w not in stop_words and \
                                                         w not in string.punctuation)]
        processed_tweets.append(words)
    return processed_tweets


df3 = pd.DataFrame(df1)
pro_twt = preprocess_tweet(df3, stop_words)
word = chain.from_iterable(pro_twt)
stylecloud.gen_stylecloud(' '.join(word), icon_name="fab fa-twitter")

hot_word = mpimg.imread('stylecloud.png')
plt.figure(figsize=(20, 20))
plt.axis('off')
plt.imshow(hot_word)
plt.show()

# LDA analysis
twt_dict = gs.corpora.Dictionary(pro_twt)
vec_twt = [twt_dict.doc2bow(doc) for doc in pro_twt]

twt_model = gs.models.LdaMulticore(
    vec_twt,
    num_topics=5,
    id2word=twt_dict,
    passes=10,
    workers=2)
twt_model.show_topics()  # show model result

pyLDAvis.enable_notebook()
topic_vis = pyLDAvis.gensim.prepare(twt_model, vec_twt, twt_dict)
topic_vis

# Compute Coherence Score
coherence_model = gs.models.CoherenceModel(
    model=twt_model,
    texts=pro_twt,
    dictionary=twt_dict,
    coherence='c_v')
coherence_score = coherence_model.get_coherence()
print(f'Coherence Score: {coherence_score}')

# Sentiment analysis
# The compound score is computed by summing the valence scores of each word in the lexicon,
# adjusted according to the rules, and then normalized to be between -1 (most extreme negative)
# and +1 (most extreme positive). This is the most useful metric if you want a single unidimensional
# measure of sentiment for a given sentence. Calling it a 'normalized, weighted composite score' is accurate.
# It is also useful for researchers who would like to set standardized thresholds for classifying sentences
# as either positive, neutral, or negative. Typical threshold values (used in the literature cited on this page) are:
#    positive sentiment: compound score >= 0.05
#    neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
#    negative sentiment: compound score <= -0.05
# The pos, neu, and neg scores are ratios for proportions of text that fall in each category (so these should all
# add up to be 1... or close to it with float operation). These are the most useful metrics if you want
# multidimensional measures of sentiment for a given sentence.


df2 = pd.DataFrame(df2, columns=['Dates', 'Tweet'])
analyser = SentimentIntensityAnalyzer()
sentiment_score = []
for tweet in df2['Tweet']:
    sentiment_score.append(analyser.polarity_scores(tweet))

sentiment_negative = []
sentiment_positive = []
sentiment_neutral = []
sentiment_compound = []
for item in sentiment_score:
    sentiment_negative.append(item['neg'])
    sentiment_positive.append(item['pos'])
    sentiment_neutral.append(item['neu'])
    sentiment_compound.append(item['compound'])

df2['sentiment_negative'] = sentiment_negative
df2['sentiment_positive'] = sentiment_positive
df2['sentiment_neutral'] = sentiment_neutral
df2['sentiment_score_compound'] = sentiment_compound

stock_sent = pd.DataFrame(equity, columns=['Adj Close', 'log_ret'])

# sentiment score
prev_d = dt.fromisoformat(start_date).date()
mean_pos = []
mean_neg = []
mean_comp = []
for date in stock_sent.index.values:
    d = pd.to_datetime(date).date()
    avg_p = np.mean(df2.loc[(df2.Dates <= d) & (df2.Dates > prev_d)].sentiment_positive)
    if np.isnan(avg_p):
        mean_pos.append(0)
    else:
        mean_pos.append(avg_p)
    prev_d = d

for date in stock_sent.index.values:
    d = pd.to_datetime(date).date()
    avg_n = np.mean(df2.loc[(df2.Dates <= d) & (df2.Dates > prev_d)].sentiment_negative)
    if np.isnan(avg_n):
        mean_neg.append(0)
    else:
        mean_neg.append(avg_n)
    prev_d = d

for date in stock_sent.index.values:
    d = pd.to_datetime(date).date()
    avg_c = np.mean(df2.loc[(df2.Dates <= d) & (df2.Dates > prev_d)].sentiment_score_compound)
    if np.isnan(avg_c):
        mean_comp.append(0)
    else:
        mean_comp.append(avg_c)
    prev_d = d

stock_sent['sentiment_positive_score'] = mean_pos
stock_sent['sentiment_negative_score'] = mean_neg
stock_sent['sentiment_compound_score'] = mean_comp

# save as csv
stock_sent.to_csv('stock_sentiment.csv')

stock_sent['sentiment_positive_score'].plot(figsize=(20, 5))
plt.title('Positive sentiment score of tweets')

stock_sent['sentiment_negative_score'].plot(figsize=(20, 5))
plt.title('Negative sentiment score of tweets')

stock_sent['sentiment_compound_score'].plot(figsize=(20, 5))
plt.title('Compound sentiment score of tweets')

# Sentiment score & close price
rcParams['figure.figsize'] = 16, 4
x = stock_sent.index
y1 = equity['Adj Close']
y2 = stock_sent['sentiment_compound_score']
fig, ax1 = plt.subplots()
ax1.plot(x, y1, color='r', linewidth=1, label='Close Price')
ax2 = ax1.twinx()
ax2.plot(x, y2, color='g', linewidth=1, label='Sentiment Score')
ax1.set_xlabel('Date')
ax1.set_ylabel('The Daily Close Price')
ax2.set_ylabel('Sentiment Compound Score')
ax1.legend(loc=0)
ax2.legend(loc=1)
plt.show()

# rolling-30-average compound score & daily close price
stock_sent['roll_30_compound'] = stock_sent['sentiment_compound_score'].rolling(30).mean()
rcParams['figure.figsize'] = 16, 4
x = stock_sent.index
y1 = equity['Adj Close']
y2 = stock_sent['roll_30_compound']
fig, ax1 = plt.subplots()
ax1.plot(x, y1, color='r', linewidth=1, label='close price')
ax2 = ax1.twinx()
ax2.plot(x, y2, color='g', linewidth=1, label='sentiment score')
ax1.set_xlabel('Date')
ax1.set_ylabel('The Daily Close Price')
ax2.set_ylabel('Rolling-30day-average Compound Score')
ax1.legend(loc=0)
ax2.legend(loc=1)
plt.show()

# Training linear model
data_X = np.array(stock_sent['roll_30_compound'][30::]).reshape(-1, 1)
data_y = np.array(equity['Adj Close'][30::]).reshape(-1, 1)

model = LinearRegression()
model.fit(data_X, data_y)

# Testing data
twitterquery = 'from:#dowjones since:2020-07-02 until:2020-12-17'
ric = 'DJIA'
start_date = '2020-07-02'
end_date = '2020-12-17'

twt_content = open('project2222.txt', 'w', encoding='utf-8')

for tweet in tw.TwitterSearchScraper(twitterquery).get_items():
    date_str = tweet.date.strftime("%Y-%m-%d %H:%M:%S%z")
    date_str = date_str[:-2] + ":" + date_str[-2:]
    twt_content.write(date_str + "|" + tweet.content + "\n")
twt_content.close()
twet = []
dates = []
twt_content = open("project2222.txt", "r", encoding="utf-8")
for l in twt_content:
    line = l.split("|")
    date_str = line[0]  # +"+00:00"
    try:
        date_time = dt.fromisoformat(date_str)
        date_time = date_time.astimezone(pytz.timezone("US/Eastern"))
        line[0] = date_time
        line[1] = line[1][:-1]
        twet.append(line)
        dates.append(date_time.date())
    except:
        twet[-1][1] += " " + l[:-1]
twt_content.close()
df = pd.DataFrame(data=twet,
                  columns=['Time', 'Tweet', 'a', 'b', 'c', 'd'])  # number of column names up to the actual situation
df1 = df.iloc[:, 0:2]
df1['Dates'] = dates
# Download testing stock data
stock_data = data.DataReader(ric, "yahoo", start_date, end_date)
equity = pd.DataFrame(data=stock_data['Adj Close'])
equity['log_ret'] = np.log(equity['Adj Close'].shift(-1)) - np.log(equity['Adj Close'])
equity = equity.iloc[:-1, ]
df1 = df.iloc[:, 0:2]
df1['Dates'] = dates
df1 = pd.DataFrame(df1, columns=['Dates', 'Tweet'])
df2 = pd.DataFrame(df2, columns=['Dates', 'Tweet'])
analyser = SentimentIntensityAnalyzer()
sentiment_score = []
for tweet in df2['Tweet']:
    sentiment_score.append(analyser.polarity_scores(tweet))

sentiment_compound = []
for item in sentiment_score:
    sentiment_compound.append(item['compound'])

df2['sentiment_score_compound'] = sentiment_compound

stock_sent = pd.DataFrame(equity, columns=['Adj Close', 'log_ret'])

prev_d = dt.fromisoformat(start_date).date()
mean_comp = []

for date in stock_sent.index.values:
    d = pd.to_datetime(date).date()
    avg_c = np.mean(df2.loc[(df2.Dates <= d) & (df2.Dates > prev_d)].sentiment_score_compound)
    if np.isnan(avg_c):
        mean_comp.append(0)
    else:
        mean_comp.append(avg_c)
    prev_d = d

stock_sent['sentiment_compound_score'] = mean_comp

stock_sent['roll_30_compound'] = stock_sent['sentiment_compound_score'].rolling(30).mean()
test_score = np.array(stock_sent['roll_30_compound'][30::]).reshape(-1, 1)
test_price = equity['Adj Close'][30::]

# Show the prediction & actual price
rcParams['figure.figsize'] = 16, 4
x = test_price.index
y1 = test_price
y2 = model.predict(test_score[:, 0::])
fig, ax1 = plt.subplots()
ax1.plot(x, y1, color='r', linewidth=1, label='Actual price')
ax2 = ax1.twinx()
ax2.plot(x, y2, color='g', linewidth=1, label='prediction')
ax1.set_xlabel('Date')
ax1.set_ylabel('The Daily Close Price')
ax2.set_ylabel('Prediction Price')
ax1.legend(loc=0)
ax2.legend(loc=1)
plt.show()
