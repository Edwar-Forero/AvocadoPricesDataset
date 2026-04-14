def create_customer_dimension(df):

    table_customer = df[["customer_id", "gender", "age"]].drop_duplicates()

    def age_group(age):

        if age <= 28:
            return "18-28"
        elif age <= 38:
            return "29-38"
        elif age <= 48:
            return "39-48"
        elif age <= 58:
            return "49-58"
        else:
            return "59+"

    table_customer["age_group"] = table_customer["age"].apply(age_group)

    return table_customer


def create_category_dimension(df):

    table_category = df[["category"]].drop_duplicates().reset_index(drop=True)

    table_category["category_id"] = table_category.index + 1

    table_category = table_category[["category_id", "category"]]

    return table_category


def create_payment_dimension(df):

    table_payment = df[["payment_method"]
                       ].drop_duplicates().reset_index(drop=True)

    table_payment["payment_id"] = table_payment.index + 1

    table_payment = table_payment[["payment_id", "payment_method"]]

    return table_payment


def create_mall_dimension(df):

    table_mall = df[["shopping_mall"]].drop_duplicates().reset_index(drop=True)

    table_mall["mall_id"] = table_mall.index + 1

    table_mall = table_mall[["mall_id", "shopping_mall"]]

    return table_mall


def create_date_dimension(df):

    table_date = df[["invoice_date"]].drop_duplicates().reset_index(drop=True)

    table_date["date_id"] = table_date["invoice_date"].dt.strftime(
        "%Y%m%d").astype(int)

    table_date["day"] = table_date["invoice_date"].dt.day
    table_date["month"] = table_date["invoice_date"].dt.month
    table_date["year"] = table_date["invoice_date"].dt.year

    return table_date


def create_sales_fact(df, category, payment, mall):

    df = df.merge(category, on="category")
    df = df.merge(payment, on="payment_method")
    df = df.merge(mall, on="shopping_mall")

    df["date_id"] = df["invoice_date"].dt.strftime("%Y%m%d").astype(int)

    table_sales = df[
        [
            "invoice_no",
            "customer_id",
            "category_id",
            "payment_id",
            "mall_id",
            "date_id",
            "quantity",
            "price",
            "total_amount",
        ]
    ]

    return table_sales
