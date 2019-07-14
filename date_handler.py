import got3
import arrow
from textblob import TextBlob
import numpy as np
from proxy_selector import get_proxy
import os

#date has to be a day (no ms) as the got3/Twitter api doesn't support ms
def dates_to_sentiment(dates, ticker, max_tweets, has_hashtag=True):

    dir_path = os.path.dirname(os.path.abspath(__file__))

    proxy_log = open(os.path.join(dir_path, 'logs/proxy_log.txt'), 'a')
    proxy_log.write("\n\n")

    memoization_sentiment = {}

    sentiments = []
    for date in dates:

        if date in memoization_sentiment:
            sentiments.append(memoization_sentiment[date])

        else:
            print("Getting tweets from: " + str(date))

            arrow_date = arrow.get(date, "YYYY-MM-DD")

            tweetCriteria = None
            if has_hashtag:
                tweetCriteria = got3.manager.TweetCriteria().setQuerySearch("#"+ticker).setMaxTweets(max_tweets) \
                    .setSince(arrow_date.replace(days=-1).format("YYYY-MM-DD")) \
                    .setUntil(arrow_date.format("YYYY-MM-DD"))
            else:
                tweetCriteria = got3.manager.TweetCriteria().setQuerySearch(ticker).setMaxTweets(max_tweets) \
                    .setSince(arrow_date.replace(days=-1).format("YYYY-MM-DD")) \
                    .setUntil(arrow_date.format("YYYY-MM-DD"))


            tweets = None
            proxies_used = set()
            blocked_proxies = set()
            proxy = None
            for i in range(0,65):
                proxy = None
                try:
                    count = 0
                    while (proxy == None or proxy in blocked_proxies) and count < 100:
                        print("Selecting new proxy. Proxy was either not set or blocked.")
                        proxy = get_proxy()
                        count += 1
                    proxies_used.add(proxy)
                    tweets = got3.manager.TweetManager.getTweets(tweetCriteria, proxy=proxy)
                    if len(tweets) == 0:
                        print("ERROR: TWEETS OF LENGTH 0")
                        raise ValueError("No tweets returned.")
                    else: 
                        break
                except: #don't want to expicitly declare all the anti bot measures twitter will throw at us
                    print("Attempting to bypass twitter block")
                    blocked_proxies.add(proxy)


            proxy_log.write(str(proxy)+ "\n")

            if len(tweets) == 0:
                raise ValueError("Proxies: \n{} \n\nBlocked: \n{}".format(proxies_used, blocked_proxies))

            polarity_per_tweet = []
            subjectivity_per_tweet = []
            for t in tweets:
                blob = TextBlob(t.text)
                polarity = blob.sentiment[0]
                polarity_per_tweet.append(polarity)
                subjectivity = blob.sentiment[1]
                subjectivity_per_tweet.append(subjectivity)

            mean_sentiment = [sum(polarity_per_tweet) / len(polarity_per_tweet)]
            mean_sentiment.append(sum(subjectivity_per_tweet) / len(subjectivity_per_tweet))

            memoization_sentiment[date] = mean_sentiment

            sentiments.append(mean_sentiment)


    sentiments = np.asarray(sentiments)
    proxy_log.close()

    return np.asarray(sentiments)


#Example

if __name__ == "__main__":
    dates = ['2018-03-01', '2018-03-01', '2018-03-02', '2017-03-02']
    dates = ['2017-03-02']
    ticker = 'Ethereum'
    max_tweets = 5
    print(dates_to_sentiment(dates, ticker, max_tweets))

