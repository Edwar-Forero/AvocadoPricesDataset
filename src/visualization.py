import matplotlib.pyplot as plt
import seaborn as sns


# Funciones de visualización para el análisis exploratorio de datos

# ventas por categoría
def sales_by_category(df):

    sales = df.groupby("category")["total_amount"].sum()

    sales.plot(kind="bar")

    plt.title("Ventas por categoría")
    plt.xlabel("Categoría")
    plt.ylabel("Total ventas")

    plt.show()

# ventas por método de pago


def sales_by_payment_method(df):

    sales = df.groupby("payment_method")["total_amount"].sum()

    sales.plot(kind="bar")

    plt.title("Ventas por método de pago")
    plt.xlabel("Método de pago")
    plt.ylabel("Total ventas")

    plt.show()

# ventas por mes


def sales_by_month(df):
    df["month"] = df["invoice_date"].dt.to_period("M")

    sales_month = df.groupby("month")["total_amount"].sum()

    plt.figure()

    sales_month.plot()

    plt.title("Ventas por mes")
    plt.xlabel("Mes")
    plt.ylabel("Total ventas")

    plt.show()

# distribución de edad de los clientes


def distribution_age(df):

    plt.figure()

    sns.histplot(df["age"], bins=20, kde=True)

    plt.title("Distribución de edad de los clientes")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")

    plt.show()


# clientes por mayor cantidad de compras
def top_customers(df):

    top_customers = (
        df.groupby("customer_id")["total_amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10))

    plt.figure()

    top_customers.plot(kind="bar")

    plt.title("Top 10 clientes con mayor volumen de compras")
    plt.xlabel("Cliente")
    plt.ylabel("Total compras")

    plt.show()

def sale_category(df):
    sale = df.pivot_table(
        values="total_amount",
        index="category",
        columns="payment_method",
        aggfunc="sum"
    )

    plt.figure()

    sns.heatmap(sale, annot=True, fmt=".0f", cmap="coolwarm")

    plt.title("Ventas por Categoría y Método de Pago")
    plt.xlabel("Método de Pago")
    plt.ylabel("Categoría")

    plt.show()


def category_month(df):
    df["month"] = df["invoice_date"].dt.to_period("M")

    sales_cat_month = df.groupby(["month","category"])["total_amount"].sum().unstack()

    plt.figure()

    sales_cat_month.plot()

    plt.title("Tendencia de ventas por categoría")
    plt.xlabel("Mes")
    plt.ylabel("Total ventas")

    plt.show()

def top_malls_by_year(df):
    df["year"] = df["invoice_date"].dt.year

    sales = df.groupby(["year", "shopping_mall"])["total_amount"].sum().unstack()

    plt.figure(figsize=(12, 6))

    sales.plot(kind="bar", ax=plt.gca())

    plt.title("Ventas por centro comercial por año")
    plt.xlabel("Año")
    plt.ylabel("Total de ventas")
    plt.xticks(rotation=0)
    plt.legend(title="Centro comercial", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.show()


def sales_by_mall(df):
    sales_mall = df.groupby("shopping_mall")["total_amount"].sum().sort_values(ascending=False)

    plt.figure()

    sales_mall.plot(kind="bar")

    plt.title("Ventas totales por centro comercial")
    plt.xlabel("Centro comercial")
    plt.ylabel("Total de ventas")

    plt.xticks(rotation=45)

    plt.show()