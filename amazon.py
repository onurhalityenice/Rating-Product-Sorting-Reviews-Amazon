##############################################################
# Rating Product & Sorting Reviews in Amazon
##############################################################

# Variables:
# reviewerID - ID of the reviewer
# asin - ID of the product
# reviewerName - name of the reviewer
# helpful - helpfulness rating of the review
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)
# day_diff - Number of days since evaluation
# helpful_yes - The number of times the review was found helpful
# total_vote - Number of votes given to the review

import numpy as np
import pandas as pd
import math
import scipy.stats as st
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# PART 1: Calculating Average Rating Based on Current Comments and Comparing with Existing Average Rating
###################################################

# Reading the Data Set and Calculating the Average Score of the Product
df = pd.read_csv("Miuul/WEEK_4/datasets/amazon_review.csv")
df.head()
df.info()
df["overall"].mean()

# Converting date variables to date format
def convert_datetime(dataframe, variable):
  dataframe[variable] = pd.to_datetime(df[variable])

convert_datetime(df, "reviewTime")

# Calculating Weighted Average of Score by Date
df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)
current_date = pd.to_datetime(str(df['reviewTime'].max()))
df["day_diff"] = (current_date - df['reviewTime']).dt.days

# Determination of time-based average weights
def time_based_weighted_average(dataframe, w1=35, w2=25, w3=20, w4=15, w5=5):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.2), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.2)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.4)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.4)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.6)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.6)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w4 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w5 / 100

time_based_weighted_average(df)
time_based_weighted_average(df, 24, 22, 20, 18, 16)

###################################################
# Part 2: Specifying 20 Reviews to Display on the Product Detail Page for the Product
###################################################

# Generating the helpful_no Variable
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# Calculating and Adding the score_pos_neg_diff, score_average_rating and wilson_lower_bound Scores to the Data
def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


wilson_lower_bound(1952, 68, confidence=0.95)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Identifying 20 Interpretations and Interpreting Results
f.sort_values("wilson_lower_bound", ascending=False).head(20)

df[["overall","helpful_yes","helpful_no","total_vote","wilson_lower_bound"]].sort_values(by="wilson_lower_bound",ascending=False).head(20)


### THE END ###