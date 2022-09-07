import streamlit as st
import pandas as pd
import pickle as pkl
import itertools
import snscrape.modules.twitter as sntwitter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = (stopwords.words('english'))
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta, date
from wordcloud import WordCloud
from PIL import Image
from zoneinfo import ZoneInfo
import pytz
import warnings
warnings.filterwarnings('ignore')






#st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("🐦 Sentiments Analyzer  😄😑😢")



keyword = st.sidebar.text_input('Enter your keyword 🔎 ', '')
st.write('Search term 🔎  :    ', keyword)

cities = st.sidebar.text_input('City name 📍', '')
st.write('The current selected city is  📍  :   ', cities)


radii = st.sidebar.slider('Radius (in Km) 🌍',0)
st.write('The selected radius is  🌍  :   ', radii)



nt = st.sidebar.slider('No. of tweets', 1, 1000, 0)
st.write('Number of tweets shown :   ', nt )


col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(" Enter start Date  📅", key='#start_date_range', value=pd.to_datetime((date.today() - timedelta(days=1)), format="%Y-%m-%d"), min_value=pd.to_datetime("2006-03-21", format="%Y-%m-%d"), max_value=pd.to_datetime("today", format="%Y-%m-%d"))
    #st.write(start_date)

with col2:
    end_date = st.date_input("Enter end Date  📅", key='#end_date_range', value=pd.to_datetime("today", format="%Y-%m-%d"), min_value=pd.to_datetime("2006-03-21", format="%Y-%m-%d"), max_value=pd.to_datetime("today", format="%Y-%m-%d"))
    



# function for fetching the tweets                                                                      
def fetch_tweets(keyword,city,radius,number,start_date, end_date):
  
    tweets = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper( f'{keyword} since:{start_date} until:{end_date} near:{city} within:{radius}km').get_items(),number))
    #st.write(tweets)
    tweets['date'] = pd.to_datetime(tweets['date'], format = "%Y-%m-%d").dt.date
    

    
    #tweets['years'] = pd.DatetimeIndex(tweets['date']).year
    #tweets['month'] = pd.DatetimeIndex(tweets['date']).month_name()
    #tweet_year = tweets.query(f'years=={yearss}')
    
    return(tweets)

# function for generating compound score

def compound_score(df_name):

    sid = SentimentIntensityAnalyzer()
    df_name['content'] = df_name['content'].str.lower()
    df_name['content'] = df_name['content'].str.replace("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", '', regex = True)
    df_name['content'] = df_name['content'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',regex = True)
    # adding stop words to the existing list
    new_stopwords = ["isnt", "dont", "maam", "according", "u", "climate change", "global warming", "know","climate", "change",          "warming","global","say", "go", "use", "said", "warmingchange", "warmingclimate","changeglobal","warming change", "thats","us",  "see","via","may", "im", "boobs", "let", "rt","cant", "done","doesnt","let",    "due","im","still","doesnt","even","etc","saying","want","please","heard","going"]
    stop.extend(new_stopwords)
    df_name['content']  = df_name['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df_name.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    df_name['negative'] = df_name['content'].astype(str).apply(lambda x:sid.polarity_scores(x)['neg'])
    df_name['neutral'] = df_name['content'].astype(str).apply(lambda x:sid.polarity_scores(x)['neu'])
    df_name['positive'] = df_name['content'].astype(str).apply(lambda x:sid.polarity_scores(x)['pos'])
    df_name['compound'] = df_name['content'].astype(str).apply(lambda x:sid.polarity_scores(x)['compound'])
    df_name['comp_score'] = df_name['compound'].apply(lambda c: 'positive' if c > 0.3 else ('negative' if c< -0.3 else 'neutral'))
    return(df_name)

def for_plot(df_name):
    df_name['date'] = pd.to_datetime(df_name['date']).dt.date
    df_name['date'] = pd.to_datetime(df_name['date'])
    df_name['date'] = pd.to_datetime(df_name['date'],format='%Y%m%d')
    df_name['year'] = pd.DatetimeIndex(df_name['date']).year
    df_name['month'] = pd.DatetimeIndex(df_name['date']).month_name()
    #years = df_name.["year"].value_counts().nunique()
    #for i in years: 
        #if i==1:
            #df_name = df_name.groupby(["month", "years"]).agg({'compound':'mean'}).sort_values('month', ascending = True).set_index("month")
        #else:
    df_name = df_name.groupby("year").agg({'compound':'mean'}).sort_values('year', ascending = True)
    return(df_name)

 

def cloud(max_word, max_font):
    
    wc = WordCloud(background_color="white", colormap='Dark2', max_words=max_word,
     max_font_size=max_font,height = 750, width = 950)
   
    # generate word cloud
    long_string = ' '.join(list(scores['content'].astype(str).values))
    wc.generate(long_string)
    # show the figure
    #plt.figure(figsize=(150,100))
    fig, axes = plt.subplots(1,2, gridspec_kw={'width_ratios': [10, 10]})
    axes[0].imshow(wc, interpolation="bilinear")
    # recolor wordcloud and show
    for ax in axes:
        ax.set_axis_off()
    st.image(wc.to_array())

    
#for streamlit app    
    
if((keyword != '') & (cities != '') & (radii != 0) & (nt != 0)):
    only_tweet = fetch_tweets(keyword, cities, radii, nt,start_date,end_date)[['date', 'content']]   
    
    st.write('Tweets  💬', only_tweet)
    
    scores = compound_score(only_tweet)
    plot = for_plot(scores)


    with st.container():
        st.subheader('Sentiments over time 📊 📈 : ')
        st.line_chart(data = plot)


    def main():
        with st.container():
            st.write("# Text Summarization with a WordCloud 📖 ")
            col1, col2 = st.columns(2)
            with col1:
                max_word = st.slider("Max words", 10, 50, 100)
                #st.write(max_word)
            with col2:
                max_font = st.slider("Max Font Size", 100, 350, 100)
                #st.write(max_word)

            cpad1, col, pad2 = st.columns((5,20,5))
            with col:
                st.write(cloud( max_word, max_font), use_column_width=True)

    if __name__=="__main__":
        main()






