"""
This script preprocess the film dataset to make it ready for class interaction.

The result dataset is meant to fulfill the following criteria:
- 2 numeric colums with different ranges
- 1 binary column
- 1 categorical column (more than 2 categories)
- 1 class that can be represented as numeric, ratio or binary
- No more than 20 rows

The dataset used is 250IMDB (250 most rated movies in IMDb):
https://www.kaggle.com/datasets/rajugc/imdb-top-250-movies-dataset

The dataset result is upload in the repository as imdb.csv

@author: jparisu

@note: this script assumes you have the dataset downloaded locally as original_imdb.csv
This is to avoid the extra space required in the repository
"""

import pandas as pd
import numpy as np
import seaborn as sns

###############################################################################

def convert_to_numeric(value):
    # Remove dollar sign and commas
    value = value.replace('$', '').replace(',', '')
    # If the value is not a digit, return NaN
    if not value.isdigit():
        return np.nan
    return int(value)

# Equivalent genres
genre_transformation = {
    'Adventure': 'Action',
    'Western': 'Action',
    'Animation': 'Family',
    'Biography': 'History',
    'Crime': 'Action',
    'Documentary': 'History',
    'War': 'History',
    'Western': 'Action',
    'Sci-Fi': 'Fantasy',
    'Thriller': 'Mystery',
    'Horror': 'Mystery',
    'Film-Noir': 'Mystery',
    'Musical': 'Family',
    'Music': 'Family',
    'Sport': 'Family',
    'Romance': 'Family',
}

genre_priority = [
    'Fantasy',
    'History',
    'Family',
    'Mystery',
    'Comedy',
    'Action',
    'Drama'
]

def translation_genre(genre):
    """Transform a genre to its equivalent"""
    if genre in genre_transformation.keys():
        return genre_transformation[genre]
    return genre

def prior_genre(genres):
    """Get all genres, transform them and take the one with more priority"""
    gl = set([translation_genre(x) for x in genres.split(',')])
    for g in genre_priority:
        if g in gl:
            return g
    return ""

###############################################################################

# Load the IMDb dataset
db_original = pd.read_csv('original_imdb.csv')

# Create a copy of the original DataFrame
df = db_original.copy()

# Transform genre list
df['genre'] = df['genre'].apply(lambda x: prior_genre(x))

# Create new column 'adult' based on certificate
children_cert = ['G', 'PG', 'PG-13']
df['adult'] = df['certificate'].apply(lambda x: 'no' if x in children_cert else 'yes')

# Select required columns
df = df[['name', 'year', 'genre', 'adult', 'budget', 'box_office']]

###############################################################################
# Save the whole dataset in raw
df.to_csv('../imdb_raw.csv', index=False)
###############################################################################

# Ranges to consider a film great
budget_range = 5

# Convert budget and box_office to numeric
df['budget'] = df['budget'].apply(convert_to_numeric)
df['box_office'] = df['box_office'].apply(convert_to_numeric)

# Remove rows with NaN values
df = df.dropna()

# Divide budget and box_office by 1M
df['budget'] = df['budget'] / 1e6
df['box_office'] = df['box_office'] / 1e6

# Create a new column 'profit'
df['c_profit'] = (df['box_office']) / df['budget']

# Change name of box_office to c_box_office
df = df.rename(columns={'box_office': 'c_box_office'})

# Create new column 'c_performance' based on profit
df['c_performance'] = df['c_profit'].apply(lambda x: 'great' if x > budget_range else 'expected')

# Round every float column to 2 decimals
df = df.round(2)

###############################################################################
# Save the whole dataset
df.to_csv('../imdb_all.csv', index=False)
###############################################################################

# Remove abnormal data
abnormal = ['Princess Mononoke', 'The Lion King', '3 Idiots']

# Filter the films
df = df[~df['name'].isin(abnormal)]

###############################################################################
# Save the result filtered
df.to_csv('../imdb_filter.csv', index=False)
###############################################################################

# Define the filters
genres = ['Mystery', 'Family', 'Action']
years = [1990, 1999]

# Filter films by genre
df = df[df['genre'].isin(genres)]

# Filter films between two years
df = df[(df['year'] >= years[0]) & (df['year'] <= years[1])]


###############################################################################
# Save the result filtered
df.to_csv('../imdb.csv', index=False)
###############################################################################

# Select specific values
films = ["The Iron Giant", "American History X", "Casino", "Reservoir Dogs", "Heat", "Forrest Gump",]

# Filter the films
df = df[df['name'].isin(films)]

###############################################################################
# Save the result reduced
df.to_csv('../imdb_reduced.csv', index=False)
###############################################################################
