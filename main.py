import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import missingno as msno
# import emoji
import spacy
nlp = spacy.load('en_core_web_sm')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st



header = st.container()
# model_training = st.set_page_config()
# dataset = st.set_page_config()

header.title('Welcome to Starbucks Rating Sentiment Project')
header.text('Add any text you want !')

st.subheader('Ratings vs Customers')

#st.title('Here we will go throght basic EDA'
data = pd.read_csv('reviews_data.csv')
review_dist = pd.DataFrame(data["Rating"].value_counts())
header.bar_chart(review_dist)

st.title('Time to train the model')
st.text('Now we will train different models:')