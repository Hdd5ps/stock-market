from flask import Flask, request, render_template
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import os
import requests
import tweepy

# Download the VADER lexicon
nltk.download('vader_lexicon')

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

# Set up API keys
NEWS_API_KEY = 'your_news_api_key'
TWITTER_API_KEY = 'your_twitter_api_key'
TWITTER_API_SECRET_KEY = 'your_twitter_api_secret_key'
TWITTER_ACCESS_TOKEN = 'your_twitter_access_token'
TWITTER_ACCESS_TOKEN_SECRET = 'your_twitter_access_token_secret'

# Set up Twitter API client
auth = tweepy.OAuth1UserHandler(
    TWITTER_API_KEY, TWITTER_API_SECRET_KEY,
    TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET
)
twitter_api = tweepy.API(auth)

def fetch_news_articles(query):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles

def fetch_tweets(query):
    try:
        tweets = twitter_api.search_tweets(q=query, count=10, lang='en')
        return tweets
    except tweepy.errors.Unauthorized as e:
        print(f"Unauthorized error: {e}")
        return []
    except tweepy.errors.TweepyException as e:
        print(f"Error fetching tweets: {e}")
        return []

def create_sentiment_chart(sentiment, text):
    labels = ['Positive', 'Neutral', 'Negative', 'Compound']
    scores = [sentiment['pos'], sentiment['neu'], sentiment['neg'], sentiment['compound']]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, scores, color=['green', 'blue', 'red', 'purple'])
    plt.xlabel('Sentiment')
    plt.ylabel('Scores')
    plt.title(f'Sentiment Analysis for: {text[:50]}...')
    plt.ylim(0, 1)
    
    # Save the plot as an image file
    chart_path = os.path.join('static', 'sentiment_chart.png')
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    query = request.form['text']
    news_articles = fetch_news_articles(query)
    tweets = fetch_tweets(query)
    
    combined_text = ' '.join([article['title'] for article in news_articles] + [tweet.text for tweet in tweets])
    sentiment = sia.polarity_scores(combined_text)
    chart_path = create_sentiment_chart(sentiment, combined_text)
    
    return render_template('result.html', sentiment=sentiment, text=query, chart_path=chart_path, news_articles=news_articles, tweets=tweets)

if __name__ == '__main__':
    app.run(debug=True)