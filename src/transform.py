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
