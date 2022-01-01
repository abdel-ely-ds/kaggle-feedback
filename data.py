import pandas as pd
from datasets import Dataset, load_metric
import ast

MAIN_PATH = "/media/abdelelyds/Elements/Feedback/"
TRAIN = "train/"
CSV_FILENAME = "train.csv.zip"
TRANSFORMED_CSV_FILENAME = "one_single_train.csv"


def get_raw_text(text_id: str,
                 main_path: str = MAIN_PATH,
                 folder: str = TRAIN
                 ):
    with open(main_path + folder + f'{text_id}.txt') as f:
        return f.read()


def read_csv_file(main_path: str = MAIN_PATH,
                  csv_filename: str = CSV_FILENAME
                  ) -> pd.DataFrame:
    return pd.read_csv(main_path + csv_filename)


def read_transformed_csv_file(main_path: str = MAIN_PATH,
                              transformed_csv_filename: str = TRANSFORMED_CSV_FILENAME
                              ) -> pd.DataFrame:
    df = pd.read_csv(main_path + transformed_csv_filename)
    df['starts'] = df.starts.apply(ast.literal_eval)
    df['ends'] = df.ends.apply(ast.literal_eval)
    df['classlist'] = df.classlist.apply(ast.literal_eval)
    return df


def get_transformed_raw_dataset(train: pd.DataFrame,
                                main_path: str = MAIN_PATH,
                                filename: str = TRANSFORMED_CSV_FILENAME,
                                class_type: str = 'discourse_type',
                                start: str = 'discourse_start',
                                end: str = 'discourse_end',
                                predstring: str = 'predictionstring',
                                save=False
                                ) -> pd.DataFrame:
    df1 = train.groupby('id')[class_type].apply(list).reset_index(name='classlist')
    df2 = train.groupby('id')[start].apply(list).reset_index(name='starts')
    df3 = train.groupby('id')[end].apply(list).reset_index(name='ends')
    df4 = train.groupby('id')[predstring].apply(list).reset_index(name='predictionstrings')

    df = pd.merge(df1, df2, how='inner', on='id')
    df = pd.merge(df, df3, how='inner', on='id')
    df = pd.merge(df, df4, how='inner', on='id')
    df['text'] = df['id'].apply(get_raw_text)

    if save:
        df.to_csv(main_path + filename, index=False)

    return df


def to_hg_dataset(df: pd.DataFrame,
                  test_size=0.1,
                  shuffle=True,
                  seed=42
                  ):
    """to hugging face dataset"""
    ds = Dataset.from_pandas(df)
    return ds.train_test_split(test_size=test_size, shuffle=shuffle, seed=seed)
