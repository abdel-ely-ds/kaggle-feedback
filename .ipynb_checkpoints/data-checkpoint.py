import pandas as pd
from datasets import Dataset, load_metric

MAIN_PATH = "/media/abdelelyds/Elements/Feedback/"
TRAIN = "train/"
CSV_FILENAME = "train.csv.zip"
TRANSFORMED_CSV_FILENAME = "one_single_train.csv"

def get_raw_text(text_id: str):
    with open(MAIN_PATH + TRAIN + f'{text_id}.txt') as f:
        return f.read()

def read_csv_file() -> pd.DataFrame:
    return pd.read_csv(MAIN_PATH + CSV_FILENAME)

def get_transformed_raw_dataset(train: pd.DataFrame,
                                filename: str = TRANSFORMED_CSV_FILENAME,
                                save=False
                                ) -> pd.DataFrame:

    df1 = train.groupby('id')['discourse_type'].apply(list).reset_index(name='classlist')
    df2 = train.groupby('id')['discourse_start'].apply(list).reset_index(name='starts')
    df3 = train.groupby('id')['discourse_end'].apply(list).reset_index(name='ends')
    df4 = train.groupby('id')['predictionstring'].apply(list).reset_index(name='predictionstrings')

    df = pd.merge(df1, df2, how='inner', on='id')
    df = pd.merge(df, df3, how='inner', on='id')
    df = pd.merge(df, df4, how='inner', on='id')
    df['text'] = df['id'].apply(get_raw_text)

    if save:
        df.to_csv(MAIN_PATH + filename, index=False)

    return df


def to_huggingface_dataset(df: pd.DataFrame,
                           test_size=0.1,
                           shuffle=True,
                           seed=42
                           ):
    ds = Dataset.from_pandas(df)
    return ds.train_test_split(test_size=test_size, shuffle=shuffle, seed=seed)
