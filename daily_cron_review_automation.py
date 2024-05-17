import os
import datetime

import pandas as pd
from google.cloud import bigquery
import requests

os.environ['TZ'] = 'Asia/Calcutta'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser("/home/mounareddy/OneDrive/Documents/mouna_reddy.json")

client=bigquery.Client()

vendor = ""

review_query = """ """.format(vendor,vendor)
review_details = client.query(review_query).to_dataframe()


review_details["createdat"] = review_details["createdat"].apply(lambda x: x.strftime('%Y-%m-%d'))

print(len(review_details))



review_details.groupby('productTag')["reviewcontent"].count().reset_index().sort_values(by="reviewcontent",ascending=False).head(20)

import pandas as pd
#import profanity_check

# Assuming you have a DataFrame called 'df' with a 'Text' column containing text data
# Create a new column 'Offensive' to store the offensive flag
#review_details['Offensive'] = review_details['reviewcontent'].apply(lambda x: profanity_check.predict([x])[0])

review_details['Offensive'] = 0

review_details

#1. how offencive
#2. meaningfull
#3. good or bad review
#3. if good how is the rating (relate to rating and review)

import nltk
#nltk.download('vader_lexicon')

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the pre-trained VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()
sia

# Assuming you have a DataFrame called 'df' with a 'Review' column containing comments
# Create a new column 'Positivity Score' to store the sentiment scores
review_details['Positivity Score'] = review_details['reviewcontent'].apply(lambda x: sia.polarity_scores(x)['pos'])

# Alternatively, you can calculate the compound sentiment score, which combines positive, negative, and neutral scores
review_details['compound sentiment score'] = review_details['reviewcontent'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Sort the DataFrame by the positivity score in descending order
review_details = review_details.sort_values('compound sentiment score', ascending=False).reset_index(drop=True)

# mark below 3 as disapp
# negat - dis
# off - pending
# pos and 4/5 rating  appr

review_score = review_details[['reviewId','createdat', 'reviewcontent', 'rating', 'Offensive',
       'Positivity Score', 'compound sentiment score']].drop_duplicates()

review_score["sentiment_lable"] = ''

review_score.loc[(review_score["compound sentiment score"]<0),"sentiment_lable"] = -1

review_score.loc[(review_score["compound sentiment score"]>0),"sentiment_lable"] = 1

review_score.loc[(review_score["compound sentiment score"]==0),"sentiment_lable"] = 0

review_score = review_score.fillna(0)

negative_words = ['Terrible', 'Awful', 'Poor', 'Horrible', 'Bad', 'Disappointing', 'Unreliable', 'Defective', 'Useless', 'Faulty',
                 'Shoddy', 'Flimsy', 'Ineffective', 'Mediocre', 'Slow', 'Difficult', 'Frustrating', 'Overpriced', 'Cheap',
                 'Not recommended', 'Junk', 'Waste', 'Trash', 'Unusable', 'Disastrous', 'Unsatisfactory', 'Inferior', 'Glitchy',
                 'Scam', 'Broken', 'Displeasing', 'Dismal', 'Unimpressive', 'Horrendous', 'Pathetic', 'Lousy', 'Disgusting',
                 'Dreadful', 'Irritating', 'Hated', 'Unpleasant', 'Buggy', 'Crappy', 'Clumsy', 'Annoying', 'Flawed', 'Deceptive',
                 'Fails', 'Uncomfortable', 'Limited', 'Unsatisfying', 'Cracked', 'Unsuitable', 'Rude', 'Untrustworthy',
                 'Unresponsive', 'Regrettable', 'Annoyed', 'Disheartening', 'Boring', 'Unimpressed', 'Lacks', 'Inaccurate',
                 'Dreadful', 'Insufficient', 'Lackluster', 'Unappealing', 'Subpar', 'Tacky', 'Difficulties', 'Displeased',
                 'Unfriendly', 'Poorly', 'Awkward', 'Unattractive', 'Unpromising', 'Unconvincing', 'Unreliable', 'Uninspired',
                 'Distasteful', 'Unusable','fool','Fake',"don’t",'dont',"don't","waste", 'Disturbing', 'Unwanted', 'Stale', 
                  'Unpleasant', 'Unoriginal', 'Unhelpful', 'Unorganized',"Not Recommended", "shipping charges", "shipping charges","used by someone","not buy",
                 'Unimpressive', 'Unflattering', 'Unhealthy', 'Unimpressive', 'Uninspiring', 'Unsightly', 'Unsympathetic', 'Unproven',
                 'Unacceptable', 'Unattractive', 'Unbalanced', 'Unnerving', 'Unsteady', 'Unremarkable', 'Unproductive', 'Unsettling',
                 'Unwanted', 'Unskilled', 'Uninformed', 'Unstable', 'Unexciting', 'Unsuccessful', 'Unkempt', 'Unfriendly', 'Unruly',
                 'Unmanageable', 'Unskilled', 'Unconvincing', 'Unfinished', 'Unfit', 'Unoriginal', 'Unworthy', 'Unimpressed',
                 'Unrealistic', 'Unintelligible', 'Unsubstantial', 'Unappetizing', 'Unwelcome', 'Uninterested', 'Unrefined',
                 'Unsatisfying', 'Unpolished', 'Unsavory', 'Uninspired', 'Unpleasant', 'Unappreciated', 'Unprepared', 'Unproven',
                 'Unprofitable', 'Unsanitary', 'Unsuccessful', 'Unreliable', 'Unstable', 'Unattractive', 'Uncomfortable',
                 'Unsatisfactory', 'Unappealing', 'Unreliable', 'Unkempt', 'Unconvincing', 'Unfinished', 'Unfit', 'Unoriginal',
                 'Unworthy', 'Unimpressed', 'Unrealistic', 'Unintelligible', 'Unsubstantial', 'Unappetizing', 'Unwelcome',
                 'Uninterested', 'Unrefined', 'Unsatisfying', 'Unpolished', 'Unsavory', 'Uninspired', 'Unpleasant',
                 'Unappreciated', 'Unprepared', 'Unproven', 'Unprofitable', 'Unsanitary', 'Unsuccessful', 'Unreliable',
                 'Unstable', 'Unattractive', 'Uncomfortable', 'Unsatisfactory', 'Unappealing']

# Create a function to check if a review contains any negative words
def check_negative(review):
    for phrase in negative_words:
        words = phrase.lower().split()
        if all(word.lower() in review.lower() for word in words):
            return "Disapproved"
    return "Approved"
    
# Apply the function to the 'review_statement' column and create a new column 'status'
review_score['status'] = review_score['reviewcontent'].apply(check_negative)

# Display the DataFrame with the new 'status' column



import numpy as np
review_score.loc[(review_score["rating"]==0) & (review_score["compound sentiment score"]>0),"rating"] = 5

review_score["Action"] = ''

review_score.loc[(review_score["rating"]>=3)&(review_score["Offensive"]==0)&(review_score["sentiment_lable"]>=0)&(review_score["Positivity Score"]>0),"Action"] = "Approved"

review_score.loc[(review_score["rating"]<3) | (review_score["sentiment_lable"]==-1)|(review_score["Positivity Score"]==0),"Action"] = "Disapproved"

review_score.loc[(review_score["status"]=="Disapproved")&(review_score["Action"]=="Approved"),"Action"] = "Disapproved"

review_score.loc[(review_score["rating"]>3)&(review_score["Offensive"]==1),"Action"] = "For Agent Review"

review_score.loc[(review_score["Positivity Score"]==0)&(review_score["sentiment_lable"]==0)&(review_score["compound sentiment score"]==0)&(review_score["Action"]=="Approved"),"Action"] = "For Agent Review"

#review_score.loc[(review_score["Positivity Score"]<=0.3)& (review_score["Action"]=="Approved"), "Action"] = "Disapproved"

review_score.loc[(review_score["Positivity Score"]<=0.2)& (review_score["Action"]=="Approved"), "Action"] = "For Agent Review"

review_score[review_score["Action"]=="For Agent Review"][["rating","Offensive","sentiment_lable"]].drop_duplicates()

review_score[review_score["Action"]=="Approved"][["rating","Offensive","sentiment_lable"]].drop_duplicates()

len(review_score[review_score["Action"]=="For Agent Review"][["rating","Offensive","sentiment_lable"]])

len(review_score[review_score["Action"]=="Approved"][["rating","Offensive","sentiment_lable"]])

review_score[review_score["Action"]=="Disapproved"][["rating","Offensive","sentiment_lable"]].drop_duplicates()

len(review_score[review_score["Action"]=="Disapproved"][["rating","Offensive","sentiment_lable"]])

#review_score[(review_score["Positivity Score"]<=0.6)& (review_score["Action"]=="Approved")]

#    1: approved / active,
#    3: disapproved / rejected
#    4: For Agent Review

review_score["statusId"] = 0

review_score.loc[review_score["Action"]=="Approved","statusId"] = 1

review_score.loc[review_score["Action"]=="Disapproved","statusId"] = 3

review_score.loc[review_score["Action"]=="For Agent Review","statusId"] = 4

review_score.statusId.value_counts()

review_score.Action.value_counts()

review_score[["reviewId","reviewcontent","rating","Action"]].to_csv("reviews_data_DS_labeled.csv",index=False)



len(review_score[["reviewId","statusId"]].to_dict('records'))



import requests
import json

url = "https://acl.mgapis.com/customer-reviews-ms/bulkAutoApprovalReviews"

payload = json.dumps(review_score[["reviewId","statusId"]].to_dict('records'))

headers = {
  'authority': 'mag.myglamm.net',
  'accept': 'application/json, text/plain, */*',
  'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
  'cache-control': 'no-cache',
  'content-type': 'application/json',
  'origin': 'https://nucleus-alpha.myglamm.net',
  'pragma': 'no-cache',
  'referer': 'https://nucleus-alpha.myglamm.net/',
  'sec-ch-ua': '"Google Chrome";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
  'sec-ch-ua-mobile': '?0',
  'sec-ch-ua-platform': '"Linux"',
  'sec-fetch-dest': 'empty',
  'sec-fetch-mode': 'cors',
  'sec-fetch-site': 'same-site',
  'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
  'userId': '100251',
  'apikey': 'fac2c144a7da65d81c939597bf983613'
}

response = requests.request("PATCH", url, headers=headers, data=payload)

print(response.text)







