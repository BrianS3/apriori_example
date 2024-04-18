
# Analysis of Tweet Emoji Itemsets Using Apriori Algorithm

This notebook illustrates how to analyze tweet emoji itemsets using the Apriori algorithm with visualizations in Altair. The following sections break down the code into understandable parts.

[Read the blog post!](https://easydatadoesit.org/the-apriori-algorithm-an-example-of-machine-learning-in-python/)

## Import Libraries
```python
import re
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import apriori
import pandas as pd
import altair as alt
import numpy as np
```

## Helper Functions
Defines helper functions for cleaning lists and creating matrices for Apriori analysis.
```python
def listToString(s): 
    str1 = " "
    return(str1.join(s))

def enter_the_matrix(df): 
    emoji_list = df.emojis.unique()
    emoji_set = set(emoji_list)
    df['emojis'] = df.text.apply(lambda text: np.unique([chr for chr in text if chr in emoji_set]))
    mlb = MultiLabelBinarizer()
    emoji_matrix = pd.DataFrame(data=mlb.fit_transform(df.emojis), index=df.index, columns=mlb.classes_)
    return emoji_matrix
```

## Data Preparation
Load and prepare tweet data for analysis.
```python
tweets = pd.read_csv('assets/tweets.csv')
tweets['emojis'] = tweets['text'].str.findall(r'[^\\w\\s.,\"@\\'?/#!$%\\^&\\*;:{}=\\-_`~()\\U0001F1E6-\\U0001F1FF]').str.len()
boxer_emojis = ['☘️','🇮🇪','🍀','💸','🤑','💰','💵','😴','😂','🤣','🥊','👊','👏','🇮🇪','💪','🔥','😭','💰']
for emoji in boxer_emojis:
    tweets[emoji] = tweets.text.str.count(emoji)
tweets['irish_pride'] = tweets['☘️'] + tweets['🇮🇪'] + tweets['🍀']
tweets['money_team'] = tweets['💸'] + tweets['🤑'] + tweets['💰'] + tweets['💵']
tweets['datetime'] = pd.to_datetime(tweets['created_at'])
tweets = tweets.set_index('datetime')
```

## Matrix and Itemsets Generation
Generate matrices and frequent itemsets using the Apriori algorithm.
```python
tweet_copy = tweets.copy()
tweet_copy2 = tweets.copy()
tweets_df = tweet_copy
boxer_df = tweet_copy2

tweet_copy['emojis'] = tweets['text'].str.findall(r'[^\\w\\s.,\"@\\'?/#!$%\\^&\\*;:{}=\\-_`~()\\U0001F1E6-\\U0001F1FF]')
tweet_copy['emojis'] = tweet_copy['emojis'].apply(lambda x: listToString(x))

tweet_copy2['emojis'] = tweets['text'].str.findall(str(boxer_emojis))
tweet_copy2['emojis'] = tweet_copy2['emojis'].apply(lambda x: listToString(x))

tweet_all = enter_the_matrix(tweet_copy)
boxer = enter_the_matrix(tweet_copy2)

tweet_all.reset_index(inplace=True)
tweet_all.drop('datetime', axis=1, inplace=True)

boxer.reset_index(inplace=True)
boxer.drop('datetime', axis=1, inplace=True)

tweet_all_frequent_3itemsets = emoji_frequent_itemsets(tweet_all, min_support=0.0005, k=3)
boxer_frequent_3itemsets = emoji_frequent_itemsets(boxer, min_support=0.00001, k=3)
boxer_frequent_3itemsets = boxer_frequent_3itemsets.loc[60:]

tweet_all_frequent_2itemsets = emoji_frequent_itemsets(tweet_all, min_support=0.0025, k=2)
boxer_frequent_2itemsets = emoji_frequent_itemsets(boxer, min_support=0.0005, k=2)
boxer_frequent_2itemsets = boxer_frequent_2itemsets.loc[19:]
```

## Visualization
Visualize the data using Altair for interactive charts.
```python
Title = alt.Chart(
    {"values": [{"text": ['The most common tweet itemsets for all vs "boxer" emojis']}]}
).mark_text(size=24, color='black', lineBreak='/n', align='left', dx=-50, fontStyle='bold').encode(
    text="text:N"
)

# Additional chart configuration code here...

# Combine and configure charts
charts = (chart1|chart2)
alt.vconcat

(Title, (subtitle|subtitle2|subtitle3), subtitle4, line, charts, background = '#F0F0F0'
           ).configure_axis(
    grid=False,
).configure_view(
    strokeWidth=0, strokeOpacity=0
)
```

The notebook includes a complex sequence of data manipulation and visualization steps to explore the relationships between different emoji combinations in tweets, applying machine learning and data visualization techniques.
