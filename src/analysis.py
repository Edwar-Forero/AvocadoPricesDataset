def basic_info(df):
    print(df.info())
    print(df.describe())


def unique_values(df):

    result = {}

    for col in df.columns:
        result[col] = df[col].nunique()

    return result


def missing_values(df):

    return df.isnull().sum()


def frequency_distribution(df, column):

    if column in df.columns:
        return df[column].value_counts()

    else:
        print(f"La columna {column} no existe")
