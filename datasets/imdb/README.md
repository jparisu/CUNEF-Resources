# IMDB Dataset

This preprocessed dataset has been created from [IMDB Top 250 Movies Dataset
](https://www.kaggle.com/datasets/rajugc/imdb-top-250-movies-dataset) and it has been adapted for its use in class.

## Columns

name,year,theme,adult,budget,box_office

- `name`: Title of the movie.
- `year`: Year of the movie.
- `theme`: Genre of the movie.
- `adult`: Indicates if the movie is for adults.
- `budget`: Budget of the movie.
- `box_office`: Box office of the movie.

## Preprocess

The original dataset has been preprocessed to make it easier to work with this dataset manually:

- Only films from 2008
- Only films of genre History, Sci-Fi or Comedy (only)
  - The column genre has been refactored so it only contains one of the three genres if the original one was one of them.
  - Some genres have been transformed:
    - Biography -> History
    - Documentary -> History
- `adult` column has been calculated for those values that are not `G`, `PG` or `PG-13`:
- Remove film *3 iditios* because it is an outlier.
