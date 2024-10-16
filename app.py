from flask import Flask, render_template, request
import tweepy
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import re
import os
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Set up Tweepy
consumer_key = os.getenv('CONSUMER_KEY')
consumer_secret = os.getenv('CONSUMER_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')

# Log the API keys to ensure they are being fetched correctly
logging.debug(f"Consumer Key: {consumer_key}")
logging.debug(f"Consumer Secret: {consumer_secret}")
logging.debug(f"Access Token: {access_token}")
logging.debug(f"Access Token Secret: {access_token_secret}")

# Check if any of the keys are None
if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
    logging.error("One or more Twitter API keys are missing. Please set the environment variables correctly.")
else:
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        # Verify credentials
        api.verify_credentials()
        logging.debug("Twitter API authentication successful.")
    except Exception as e:
        logging.error(f"Error during Twitter API authentication: {e}")

# Set up News API
news_api_key = os.getenv('NEWS_API_KEY')

# Log the News API key to ensure it is being fetched correctly
logging.debug(f"News API Key: {news_api_key}")

# Check if the News API key is None
if not news_api_key:
    logging.error("News API key is missing. Please set the environment variable correctly.")

# Set up NLTK
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        stock = request.form['stock']
        logging.debug(f"Analyzing stock: {stock}")
        tweets = fetch_tweets(stock)
        news_articles = fetch_news_articles(stock)
        logging.debug(f"Fetched {len(tweets)} tweets and {len(news_articles)} news articles for stock: {stock}")
        tweet_sentiments = analyze_sentiment(tweets)
        news_sentiments = analyze_sentiment(news_articles)
        logging.debug(f"Tweet Sentiments: {tweet_sentiments}")
        logging.debug(f"News Sentiments: {news_sentiments}")
        print(f"Tweet Sentiments: {tweet_sentiments}")
        print(f"News Sentiments: {news_sentiments}")
        create_sentiment_chart(tweet_sentiments, news_sentiments, stock)
        return render_template('result.html', stock=stock, tweet_sentiments=tweet_sentiments, news_sentiments=news_sentiments)
    except Exception as e:
        logging.error(f"Error in analyze route: {e}")
        return "Internal Server Error", 500

def fetch_tweets(stock):
    try:
        tweets = api.search(q=stock, count=100, lang='en')
        logging.debug(f"Raw Tweets: {[tweet.text for tweet in tweets]}")
        cleaned_tweets = [clean_text(tweet.text) for tweet in tweets]
        logging.debug(f"Cleaned Tweets: {cleaned_tweets}")
        print(f"Fetched Tweets: {cleaned_tweets}")
        return cleaned_tweets
    except Exception as e:
        logging.error(f"Error fetching tweets: {e}")
        return []

def fetch_news_articles(stock):
    try:
        url = f'https://newsapi.org/v2/everything?q={stock}&apiKey={news_api_key}'
        response = requests.get(url)
        articles = response.json().get('articles', [])
        cleaned_articles = [clean_text(article['title']) for article in articles]
        logging.debug(f"Fetched News Articles: {cleaned_articles}")
        print(f"Fetched News Articles: {cleaned_articles}")
        return cleaned_articles
    except Exception as e:
        logging.error(f"Error fetching news articles: {e}")
        return []

def clean_text(text):
    original_text = text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove special characters
    logging.debug(f"Original Text: {original_text}, Cleaned Text: {text}")
    print(f"Original Text: {original_text}, Cleaned Text: {text}")
    return text

def analyze_sentiment(texts):
    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    for text in texts:
        score = sia.polarity_scores(text)
        logging.debug(f"Text: {text}, Sentiment Score: {score}")
        print(f"Text: {text}, Sentiment Score: {score}")
        if score['compound'] > 0.05:
            sentiments['positive'] += 1
        elif score['compound'] < -0.05:
            sentiments['negative'] += 1
        else:
            sentiments['neutral'] += 1
    return sentiments

def create_sentiment_chart(tweet_sentiments, news_sentiments, stock):
    try:
        labels = ['Positive', 'Neutral', 'Negative']
        tweet_values = [tweet_sentiments['positive'], tweet_sentiments['neutral'], tweet_sentiments['negative']]
        news_values = [news_sentiments['positive'], news_sentiments['neutral'], news_sentiments['negative']]

        x = range(len(labels))

        plt.figure(figsize=(10, 5))
        plt.bar(x, tweet_values, width=0.4, label='Tweets', align='center')
        plt.bar(x, news_values, width=0.4, label='News', align='edge')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title(f'Sentiment Analysis for {stock}')
        plt.xticks(x, labels)
        plt.legend()
        plt.savefig('static/sentiment_chart.png')
        plt.close()
        logging.debug(f"Sentiment chart created for stock: {stock}")
        print(f"Sentiment chart created for stock: {stock}")
    except Exception as e:
        logging.error(f"Error creating sentiment chart: {e}")

if __name__ == '__main__':
    app.run(debug=True)