# Collaborative Filtering Models Findings Report 

## Data Preprocessing for 'Surprise'

In terms of data wrangling, there was a minimal amount of manipulation required in order to produce a dataframe which would be compatible with classes from the Surprise library. Surprise requires data to follow a three-column structure of user ID, item id (in this case `artistID`), and a rating. As the `user_artist.dat` file already complied with this requirement no columns were dropped. 

However, instead of a discrete rating column, the `user_artist` data contains a continuous feature `weight` which indicates the number of songs played per artist for each unique user. A little visualization demonstrated a large amount of skew in this data: 

![image](/Users/dimitrikestenbaum/Desktop/RecSys/PlayCountVsFrequency.png)

