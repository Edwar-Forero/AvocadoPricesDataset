import pandas as pd


def clean_columns(df):

    df.columns = df.columns.str.lower()

    return df


def convert_date(df):

    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    return df


def add_total_sale(df):

    df["total_amount"] = df["quantity"] * df["price"]

    return df


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)]

def count_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask = (df[column] < lower) | (df[column] > upper)
    
    total_outliers = mask.sum()
    total_rows = len(df)

    print(f"Columna: {column}")
    print(f"Outliers: {total_outliers}")
    print(f"Total filas: {total_rows}")
    print(f"Porcentaje: {(total_outliers / total_rows) * 100:.2f}%")

    return mask
