import streamlit as st
import pandas as pd
import emoji
import regex
import re
import numpy as np
from collections import Counter
import plotly.express as px
import datetime
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import time
import base64
from matplotlib import style


# -*- coding: utf-8 -*-
"""
Created on : &nbsp;&nbsp; Sat 1-08-2020  &nbsp;&nbsp; &nbsp;&nbsp; Author:&nbsp;&nbsp;  @mjindal585
"""

st.set_option('deprecation.showfileUploaderEncoding', False)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def preprocessing(df):
    #st.write(df['messages'].head())
    df['messages'].to_json('new.json',index=True)
    df = pd.read_json('new.json')
    df = df.T
    #df = df.drop(['type','actor','action','inviter','actor_id','members','message_id'],axis=1)
    #df = df[df['file'] != '(File not included. Change data exporting settings to download.)']
    #df = df[df['photo'] != '(File not included. Change data exporting settings to download.)']
    #df = df.drop(['photo','height','width','file','thumbnail','mime_type','media_type','duration_seconds','edited','forwarded_from'],axis=1)
    df.to_csv('new2.csv',encoding="utf-8-sig",index=False)
    df=pd.read_csv('new2.csv')
    df['text'] = df['text'].astype(str)

    def message_clean(df):
        message = ''
        if df.loc['text'][0] == '[':
            text = df.loc['text'].strip().strip('[').strip(']')        
            escapes = ''.join([chr(char) for char in range(1, 32)])
            translator = str.maketrans('', '', escapes)
            t = text.translate(translator)
            results = re.findall(r"'(.*?)'|\"(.*?)\"|({.*?})", t) #check for text only and dictionary using regex
            results = [item for elem in results for item in elem if len(item)] # Clean empty records
            for e in results:
                if(e[0]=='{'):
                    d = eval(e)
                    message = message + " " + d['text'].strip()
                else:
                    message = message + " " + e.strip()
        else:
            message = df.loc['text']
        return message
    
    df['message'] = df.apply(message_clean,axis=1)
    df.dropna(subset=['text'],inplace=True)
    df.drop(df[df.text == 'nan'].index , inplace=True)
    df.drop(df[df.text == 'NaN'].index , inplace=True)
    df = df[['id', 'date', 'text', 'from', 'from_id', 'reply_to_message_id','message']]
    df['date'] = df.apply(lambda x: ' '.join(x['date'].split('T')),axis=1).astype('datetime64[ns]')
    df['from'] = df['from'] + ' ' + '( ' + df['from_id'].astype(str) + ' )'
    final = df[['id','date','from','message','reply_to_message_id']]
    final.to_csv('finalchat-custom.csv',index=False)


def run_analysis(df):
    def split_count(text):
        emoji_list = []
        data = regex.findall(r'\X', text)
        for word in data:
            if any(char in emoji.UNICODE_EMOJI for char in word):
                emoji_list.append(word)
        return emoji_list

    df["emoji"] = df["message"].apply(split_count)
    df['Messagecount'] = 1
    emojis = sum(df['emoji'].str.len())
    #st.write(emojis)
    total_messages = df.shape[0]
    URLPATTERN = r'(https?://\S+)'
    df['urlcount'] = df.message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
    links = np.sum(df.urlcount)

    df['Letter_Count'] = df['message'].apply(lambda s : len(s))
    df['Word_Count'] = df['message'].apply(lambda s : len(s.split(' ')))
    df["emoji_count"]= df['emoji'].str.len()
    #st.write(df['from'].unique())
    df = df[df['from'].notna()]
    #st.write(df['from'].unique())

    



    #generate stats dataset for each user
    
    l = df['from'].unique()
    stats = list()

    for i in range(len(l)):
        # Filtering out messages of particular user
        req_df= df[df["from"] == l[i]]
        # req_df will contain messages of only one particular user
        # shape will give the number of rows which indirectly means the number of messages
        message_sent = req_df.shape[0]
        #Word_Count contains of total words in one message. Sum of all words/ Total Messages will yield words per message
        words_per_message = (np.sum(req_df['Word_Count']))/req_df.shape[0]
        # emojis conists of total emojis
        emojis_stat = sum(req_df['emoji'].str.len())
        #links consist of total links
        links_stat = sum(req_df["urlcount"])   
        #append teh data to stats list
        stats.append([l[i],message_sent,words_per_message,emojis_stat,links_stat])
    
    stats_df = pd.DataFrame(stats,columns=['From','Messages_sent','Words_per_message','Emojis_sent','Links_sent'])
    stats_df.to_csv('stats.csv',index=False)  
    
    total_emojis_list = list([a for b in df.emoji for a in b])
    emoji_dict = dict(Counter(total_emojis_list))
    emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
    emoji_df = pd.DataFrame(emoji_dict, columns=['emoji', 'count'])

    #most replied messages
    reply_list = list([a for a in df.reply_to_message_id ])
    reply_dict = dict(Counter(reply_list))
    reply_dict = sorted(reply_dict.items(), key=lambda x: x[1], reverse=True)

    def f(i):
        x = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return x[i]

    df['date'] = df['date'].astype('datetime64[ns]')

    #st.write(df.dtypes)

    day_df=pd.DataFrame()
    day_df['message'] = df["message"]
    day_df['day_of_date'] = df['date'].dt.weekday
    day_df['day_of_date'] = day_df["day_of_date"].apply(f)
    day_df["messagecount"] = 1
    day = day_df.groupby("day_of_date").sum()
    day.reset_index(inplace=True)

    text = " ".join(review for review in df.message)

    stopwords = set(STOPWORDS)
    stopwords.update(["ra", "ga", "na", "ani", "em", "ki", "ah","ha","la","eh","ne","le","ni","lo","Ma","Haa","ni"])
    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

    #st.write(df['from'].value_counts().head(10))
    #print(df['from'].value_counts().head(10))
    
    
    options=["Select","Total number of words typed, messages, emojis and links sent","Generate a dataframe for count of each emoji used and plot a pie chart", \
            "Details of the most replied messages","Day and Date wise statistics of messages sent","Most Active People, Time , Days","Details of the Longest Message",\
            "Generate and Download Stats for each User","Word Cloud of the chat"]
    
    ch = st.selectbox("Select Option",options)
    
    if ch == "Select":
        st.warning("No option selected")

    elif ch == "Total number of words typed, messages, emojis and links sent":
        st.write("Stats")
        st.write("Words Typed : " + str(len(text)))
        st.write("Messages : "+ str(total_messages))
        st.write("Emojis : " + str(emojis))
        st.write("Links : " + str(links))
    
    elif ch== "Generate a dataframe for count of each emoji used and plot a pie chart":
        st.write(emoji_df)
        fig = px.pie(emoji_df, values='count', names='emoji')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)

    elif ch == "Details of the most replied messages":
        for i,j in reply_dict[:10]:
            for k in range(len(df)):
                if df.iloc[k]['id'] == i:
                    st.write(df.iloc[k][['from','message']])

    elif ch == "Day and Date wise statistics of messages sent":
        fig = px.line_polar(day, r='messagecount', theta='day_of_date', line_close=True)
        fig.update_traces(fill='toself')
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                visible=True,
            )),
            showlegend=False
        )
        st.plotly_chart(fig)

        df['Date'] = [datetime.datetime.date(d) for d in df['date']] 
        data = df['Date'].value_counts()
        fig = px.bar(data, orientation='v')
        
        st.plotly_chart(fig)

    elif ch == "Details of the Longest Message":
        st.write(df.iloc[df['Word_Count'].argmax()][['id','date','from','message']])

    elif ch == "Word Cloud of the chat":
        plt.figure(figsize=(15,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

    elif ch == "Generate and Download Stats for each User":
        
        st.write(stats_df)  

        def download_link(object_to_download, download_filename, download_link_text):
        
            if isinstance(object_to_download,pd.DataFrame):
                object_to_download = object_to_download.to_csv(index=False)

                # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(object_to_download.encode()).decode()

            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

        if st.button('Click here to Generate download Link '):
            tmp_download_link = download_link(stats_df, 'stats.csv', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

    elif ch == "Most Active People, Time , Days":
        st.write("Most Active People")
        st.write(df['from'].value_counts().head(10))

        st.write("\n")
        style.use('seaborn-darkgrid')

        df['Time'] = [datetime.datetime.time(d).strftime('%H:%M') for d in df['date']] 
        df['Time'].value_counts().head(10).plot.barh() # Top 10 Times of the day at which the most number of messages were sent
        plt.title("Most Active Time")
        plt.xlabel('Number of messages')
        plt.ylabel('Time')

        st.pyplot(plt)


        
        df['Date'] = [datetime.datetime.date(d) for d in df['date']] 
        df['Date'].value_counts().head(10).plot.barh()
        plt.title("Most Active Days")
        plt.xlabel('Number of Messages')
        plt.ylabel('Date')
        st.pyplot(plt)


def app():
    st.title("Telegram Group Chat Analysis 💬 🗣️ 💭 ")
    option=["About","Analyze Chat"]
    choice = st.sidebar.selectbox("Menu",option)

    if choice == "About":

        st.subheader("This app aims to perform various analysis on telegram group chats:")
        st.write("1. Get the total number of words typed, messages, emojis and links sent")
        st.write("2. Generate stats for each user and store it in a csv file")
        st.write("3. Generate a dataframe for count of each emoji used and plot a pie chart")
        st.write("4. Details of the most replied messages")
        st.write("5. Day wise statistics of messages sent")
        st.write("6. Most Active People, Time , Days")
        st.write("7. Details of the Longest Message")
        st.write("8. Word Cloud of the chat")

        #st.subheader(' ------------------------ Created By : MOHIT JINDAL ----------------------')

    else:

        st.subheader("Upload the exported chat (in .json format) to begin")

        uploaded_chat = st.file_uploader("Choose a file....",type="json")
        if uploaded_chat:
            st.success("Uploaded Successfully!")
            df = pd.read_json(uploaded_chat)
            #st.write(df)
            #st.write(type(uploaded_chat))
            time.sleep(2)
            with st.spinner(text="Preprocessing Data"):
                df = preprocessing(df)
            st.balloons()
            df = pd.read_csv('finalchat.csv')
            st.success("Processed!")
            #st.write(df.iloc[67]['message'])

            run_analysis(df)




if __name__ == "__main__":
    app()