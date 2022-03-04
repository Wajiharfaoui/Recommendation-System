# Collaborative Filtering Models Findings Report 

## Data Preprocessing for `Surprise`

In terms of data wrangling, there was a minimal amount of manipulation required in order to produce a dataframe which would be compatible with classes from the Surprise library. Surprise requires data to follow a three-column structure of user ID, item id (in this case `artistID`), and a rating. As the `user_artist.dat` file already complied with this requirement no columns were dropped. 

However, instead of a discrete rating column, the `user_artist` data contains a continuous feature `weight` which indicates the number of songs played per artist for each unique user. A little visualization demonstrated a large amount of skew in this data: 

![image](/PlayCountVsFrequency.png)

We can see in the figure above how left-skewed the `weight` data is. However, due to the scale of the horizontal axis, which represents frequency, the right-tail distribution caused by the skew is obfuscated from the chart. In fact the maximum `weight` is actually `352698` plays, but only has a frequency of one, therefore does not appear in this plot. 

Having a continuous measure like `weight` is not compatible with `Surprise`, which requires some kind of discrete rating scale such as 1-5. Therefore, the next step was too discretize or categorize the `weight` column to make it conform to a scale. A 1-5 rating scale was used as the distribution of `weight` could be equally divided in five quantiles. The code chunk below demonstrates how this discretization was achieved using the `qcut` function from `Pandas`. 

``` Python
#discretize weights using qcuts 
user_artists_df['weight_quantiles'] = pd.qcut(user_artists_df['weight'],
                           q=[0,.2,.4,.6,.8,1],
                           labels=False,
                           precision=0)

```

Upon categorizing the `weight` and renaming it `weight_quantiles` the observations are almost equally distributed within these categories the exact values counts can be seen below: 

* 0    18770
* 3    18581
* 4    18548
* 2    18469
* 1    18466

## `Train`/`Test` Split + Initializing Surprise Objects

Next we perform a traditional train/test split of the `user_artists` data with a test-set size of `0.3`. Next is a crucial step when working with the `Surprise` library. The `Reader` class is used to parse our brand new ratings (`weight_quantiles`) based on given scale (1-5 of course). Secondly, a full dataset object is created for subsequent cross-validation as well as `Surprise`-specific trainset and testset objects. The code for this can be seen below: 

``` Python
#create reader object 
reader = surprise.Reader(rating_scale=(1,5)) #1:5 scale 

#create surprise train and test set objects
data = surprise.Dataset.load_from_df(UA_df_cf[["userID","artistID","weight_quantiles"]], reader)
UA_train = surprise.Dataset.load_from_df(UA_train, reader).build_full_trainset()
UA_test = list(UA_test.itertuples(index=False, name=None))

```

