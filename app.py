# Import necessary packages and functions
import pandas as pd
from flask import Flask, render_template, request
from alpha_vantage.timeseries import TimeSeries
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from datetime import datetime
import yfinance as yf
from textblob import TextBlob
from newsapi import NewsApiClient
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import threading

# Load the saved TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = joblib.load(f)

# Load the saved model
with open('sentiment_model.pkl', 'rb') as f:
    model = joblib.load(f)

# Load the CSV file containing stock symbols, names, categories, and countries
stock_symbols = pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')

# Function to map stock symbol to its corresponding name, category, and country
def map_symbol_to_details(symbol):
    symbol = symbol.upper()
    if symbol in stock_symbols['Ticker'].values:
        details = stock_symbols.loc[stock_symbols['Ticker'] == symbol].iloc[0]
        return details['Name'], details['Category Name'], details['Country']
    else:
        return None, None, None

# Function to filter news articles related to the stock symbol name, category, or country
def filter_stock_related_articles(articles, symbol, name, category, country):
    filtered_articles = []
    for article in articles.get('articles', []):
        title = article.get('title', '')
        description = article.get('description')  # Do not provide a default value
        if description is not None:  # Check if description is not None
            description = description.lower()  # Convert to lowercase
        if any(entity.lower() in title.lower() or (description and entity.lower() in description)
               for entity in (symbol, name, category, country)):
            filtered_articles.append({
                'title': title,
                'description': article.get('description', ''),  # Provide default value if description is None
                'url': article['url']
            })
    return filtered_articles


# Function to fetch news articles related to the stock symbol name, category, or country
def get_news_articles(symbol):
    newsapi = NewsApiClient(api_key='9cc0aa97df9d4113ac08507a4a003915')
    name, category, country = map_symbol_to_details(symbol)
    if name:
        articles = newsapi.get_everything(q=name, language='en', sort_by='publishedAt')
        filtered_articles = filter_stock_related_articles(articles, symbol, name, category, country)
        return filtered_articles
    else:
        print("Symbol not found.")
        return []

def preprocess_text(text):
    # You can add more preprocessing steps as needed
    return text.lower().strip()

def get_combined_sentiment(model_sentiment, vader_sentiment, textblob_sentiment):
    # Implement your logic for combining sentiments here
    # For simplicity, let's take a simple voting approach
    sentiments = [model_sentiment, vader_sentiment, textblob_sentiment]
    positive_count = sentiments.count('Positive')
    negative_count = sentiments.count('Negative')
    if positive_count > negative_count:
        return 'Positive'
    elif positive_count < negative_count:
        return 'Negative'
    else:
        return 'Neutral'

def perform_sentiment_analysis(news_data, model):
    sentiments = []
    article_titles = []
    model_sentiments = []
    vader_sentiments = []
    textblob_sentiments = []
  
    for article in news_data:
        text = preprocess_text((article['title'] or '') + ' ' + (article['description'] or ''))
        
        # Perform sentiment analysis using the trained model
        text_vectorized = vectorizer.transform([text])
        model_prediction = model.predict(text_vectorized)[0]
        model_sentiments.append(model_prediction)
        
        # Perform sentiment analysis using VADER
        analyzer = SentimentIntensityAnalyzer()
        vader_score = analyzer.polarity_scores(text)['compound']
        vader_sentiment = 'Positive' if vader_score > 0 else 'Negative'
        vader_sentiments.append(vader_sentiment)
        
        # Perform sentiment analysis using TextBlob
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity
        textblob_sentiment = 'Positive' if textblob_score > 0 else 'Negative'
        textblob_sentiments.append(textblob_sentiment)
        
        article_titles.append(article['title'])

    for model_prediction, vader_sentiment, textblob_sentiment in zip(model_sentiments, vader_sentiments, textblob_sentiments):
        # Combine sentiments from all methods
        combined_sentiment = get_combined_sentiment(model_prediction, vader_sentiment, textblob_sentiment)
        sentiments.append(combined_sentiment)

    return article_titles, sentiments



# Initialize Flask app
app = Flask(__name__)

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/insertintotable', methods=['POST'])
def insertintotable():
    nm = request.form['nm']
    # Functions for fetching historical data and performing ARIMA and LSTM predictions
    def get_historical(quote):
        end = datetime.now()
        start = datetime(end.year-2,end.month,end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        df.to_csv(''+quote+'.csv')
        if(df.empty):
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            #Format df
            #Last 2 yrs rows => 502, in ascending order => ::-1
            data=data.head(503).iloc[::-1]
            data=data.reset_index()
            #Keep Required cols only
            df=pd.DataFrame()
            df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']
            df.to_csv(''+quote+'.csv',index=False)
        print(df)
        return df

    def ARIMA_ALGO(df):
        # Remove this part related to 'Code' column
    # uniqueVals = df["Code"].unique()  
    # len(uniqueVals)
    # df = df.set_index("Code")

    # Remaining function code remains unchanged...

        #def parser(x):
            #return datetime.strptime(x, '%Y-%m-%d')

        def arima_model(train, test):
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(6, 1, 0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            return predictions

        for company in df.columns[:10]:  # Loop over columns instead of unique values
            data = df.reset_index()
            data['Price'] = data['Close']
            Quantity_date = data[['Price', 'Date']]
            #Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
            Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
            Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
            Quantity_date = Quantity_date.drop(['Date'], axis=1)

            fig = plt.figure(figsize=(10, 7), dpi=65)
            plt.plot(Quantity_date)
            plt.savefig('static/Trends.png')
            plt.close(fig)

            quantity = Quantity_date.values
            size = int(len(quantity) * 0.80)
            train, test = quantity[0:size], quantity[size:len(quantity)]

            predictions = arima_model(train, test)

            fig = plt.figure(figsize=(10, 7), dpi=65)
            plt.plot(test, label='Actual Price')
            plt.plot(predictions, label='Predicted Price')
            plt.legend(loc=4)
            plt.savefig('static/ARIMA.png')
            plt.close(fig)

            print()
            print("##############################################################################")
            arima_pred = predictions[-2]
            print("Tomorrow's", quote, " Closing Price Prediction by ARIMA:", arima_pred)
            error_arima = math.sqrt(mean_squared_error(test, predictions))
            print("ARIMA RMSE:", error_arima)
            print("##############################################################################")
            return arima_pred, error_arima

        

    def LSTM_ALGO(df):
        #Split data into training set and test set
        dataset_train=df.iloc[0:int(0.8*len(df)),:]
        dataset_test=df.iloc[int(0.8*len(df)):,:]
        ############# NOTE #################
        #TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
        # HERE N=7
        ###dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
        training_set=df.iloc[:,4:5].values# 1:2, to store as numpy array else Series obj will be stored
        #select cols using above manner to select as float64 type, view in var explorer

        #Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc=MinMaxScaler(feature_range=(0,1))#Scaled values btween 0,1
        training_set_scaled=sc.fit_transform(training_set)
        #In scaling, fit_transform for training, transform for test
        
        #Creating data stucture with 7 timesteps and 1 output. 
        #7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
        X_train=[]#memory with 7 days from day i
        y_train=[]#day i
        for i in range(7,len(training_set_scaled)):
            X_train.append(training_set_scaled[i-7:i,0])
            y_train.append(training_set_scaled[i,0])
        #Convert list to numpy arrays
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        X_forecast=np.array(X_train[-1,1:])
        X_forecast=np.append(X_forecast,y_train[-1])
        #Reshaping: Adding 3rd dimension
        X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))#.shape 0=row,1=col
        X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
        #For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features)
        
        #Building RNN
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        from keras.layers import LSTM
        
        #Initialise RNN
        regressor=Sequential()
        
        #Add first LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
        #units=no. of neurons in layer
        #input_shape=(timesteps,no. of cols/features)
        #return_seq=True for sending recc memory. For last layer, retrun_seq=False since end of the line
        regressor.add(Dropout(0.1))
        
        #Add 2nd LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
        #Add 3rd LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
        #Add 4th LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        
        #Add o/p layer
        regressor.add(Dense(units=1))
        
        #Compile
        regressor.compile(optimizer='adam',loss='mean_squared_error')
        
        #Training
        regressor.fit(X_train,y_train,epochs=25,batch_size=32 )
        #For lstm, batch_size=power of 2
        
        #Testing
        ###dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
        real_stock_price=dataset_test.iloc[:,4:5].values
        
        #To predict, we need stock prices of 7 days before the test set
        #So combine train and test set to get the entire data set
        dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0) 
        testing_set=dataset_total[ len(dataset_total) -len(dataset_test) -7: ].values
        testing_set=testing_set.reshape(-1,1)
        #-1=till last row, (-1,1)=>(80,1). otherwise only (80,0)
        
        #Feature scaling
        testing_set=sc.transform(testing_set)
        
        #Create data structure
        X_test=[]
        for i in range(7,len(testing_set)):
            X_test.append(testing_set[i-7:i,0])
            #Convert list to numpy arrays
        X_test=np.array(X_test)
        
        #Reshaping: Adding 3rd dimension
        X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        
        #Testing Prediction
        predicted_stock_price=regressor.predict(X_test)
        
        #Getting original prices back from scaled values
        predicted_stock_price=sc.inverse_transform(predicted_stock_price)
        fig = plt.figure(figsize=(10,7),dpi=65)
        plt.plot(real_stock_price,label='Actual Price')  
        plt.plot(predicted_stock_price,label='Predicted Price')
          
        plt.legend(loc=4)
        plt.savefig('static/LSTM.png')
        plt.close(fig)
        
        
        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        
        
        #Forecasting Prediction
        forecasted_stock_price=regressor.predict(X_forecast)
        
        #Getting original prices back from scaled values
        forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
        
        lstm_pred=forecasted_stock_price[0,0]
        print()
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by LSTM: ",lstm_pred)
        print("LSTM RMSE:",error_lstm)
        print("##############################################################################")
        return lstm_pred,error_lstm
    
    # Get the stock symbol from the form
    quote = nm

    # Try-except to check if valid stock symbol
    try:
        df = get_historical(quote)
    except:
        return render_template('index.html', not_found=True)
    else:
        arima_pred, error_arima = ARIMA_ALGO(df)
        lstm_pred, error_lstm = LSTM_ALGO(df)
        
        # Perform sentiment analysis on news articles related to the stock symbol
        news_articles = get_news_articles(quote)
        sentiments=[]
        titles=[]
        if news_articles:
            article_titles, article_sentiments = perform_sentiment_analysis(news_articles,model)
            for article in article_titles:
                titles.append(article)
            
            # Output individual article sentiments
            print("Individual article sentiments:")
            for sentiment in article_sentiments:
                sentiment_label = sentiment.lower().replace(' sentiment', '')
                sentiments.append(sentiment_label.lower())
            
            # Calculate combined sentiment for the stock
            combined_sentiment = Counter(article_sentiments).most_common(1)[0][0]
            print("\nCombined sentiment for", quote, "based on news articles:", combined_sentiment)
            # Plot bar graph for sentiments
            sentiment_counts = Counter(sentiments)
            labels, counts = zip(*sentiment_counts.items())
            colors = {'positive': '#AFE1AF', 'neutral': '#7CB9E8', 'negative': '#F33A6A'}
            plt.figure(figsize=(10,7),dpi=65)
            plt.bar(labels, counts, color=[colors[label] for label in labels])
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.title('Sentiment Analysis for ' + quote)
            plt.savefig('static/sentiment_analysis.png')  # Save the plot to static folder
            plt.close()
        else:
            print("No news articles found for", quote)
            combined_sentiment = "No news articles found"
        

        plt.figure(figsize=(10,7),dpi=65)
        plt.plot(df['Close'])
        plt.title('Recent Trends for ' + quote)
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.grid(True)
        plt.savefig('static/recent_trends.png')
        plt.close()
        
        return render_template('results.html', quote=quote, arima_pred=round(arima_pred,2), lstm_pred=round(lstm_pred,2), error_arima=round(error_arima,2), error_lstm=round(error_lstm,2), combined_sentiment=combined_sentiment,article_titles=titles,sentiments=sentiments)

if __name__ == '__main__':
    threading.Thread(target=app.run, daemon=True).start()