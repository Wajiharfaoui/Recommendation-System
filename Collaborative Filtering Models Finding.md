# Introduction 
In this project we will be improvinge Recommendation Systems of "LastFM", a music streaming platform. Currently the company only recommends top 10 most popular artists to all users. We aim to increase user engagement with the platform by applying item, user and content based recommendation systems. 

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


# Content Based Recommendation System 
The matrix for Content based recommendation systems is created using two data frames: user_taggedartists.dat and tags.dat.
We further discuss steps executed on these datasets in order to create the content matrix.
## Data preprocessing and content matrix for content-based recommendation system
To being, we create a full-date variable column on user_taggedartists.dat. We can see that dates when users tagged artists are mostly frm 2000 and on. 

![image](/TagsDistribution0.png)
![image](/TagsDistribution1.png)

We then proceed to create qualitative variables for the same data frame, and a recency variable column later, by categorizing dates artists were tagged by a user as "Very Old" if tagged before January 1970, "Old" if tagged before Jaunary 1984, "New" if tagged before January 2010 and "Very New" from January 2010 and further.

We then merge the two data frames, one containing a tag value of each tag ID, and the other one containg UserID, Artist ID and Recency value. 

To create content matrix, we pivot the merged table twice in order to turn categorical variables in dummy variables. Then, two pivoted tables merged together are combined into a content matrix, with ArtistID as an index and Genre and Recency as variables. The resulting matrix is: 

``` Python
cb.head()
```
## Applying the model

We initiate the model at NN = 10, filtering the matrix for 10 nearest neighbors with non-negative similarity.
Then, we fit on content using content matrix, and on ratings using train dataset. 

``` Python
# init content-based
cb_mod = ContentBased(NN=10)

# fit on content
cb_mod.fit(cb)

# fit on train_ratings
cb_mod.fit_ratings(UA_train)

cb_pred = cb_mod.test(UA_test)
```
We get the following evaluation results: 

``` Python
# compute metrics for CB RS
cb_res = eval.evaluate(cb_pred, topn=5, rating_cutoff=3.5).rename(columns={'value':'Content_based_10'})
cb_res
```

* RMSE	0.891378
* MAE	0.669223
* Recall	0.386871
* Precision	0.760339
* F1	0.512814
* NDCG@5	0.872620

