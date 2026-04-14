from sqlalchemy import create_engine


def connect_db():

    user = "postgres.hffhjjbknmuklllzdtfa"
    password = "7ONoBYZDZYh50sAV"
    host = "aws-1-us-east-2.pooler.supabase.com"
    port = "6543"
    database = "postgres"
    pool_mode = "transaction"

    engine = create_engine(
        f'postgresql://{user}:{password}@{host}:{port}/{database}')

    return engine


def verify_connection(engine):

    try:
        with engine.connect() as connection:
            print("Conexión exitosa a la base de datos")
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")


def load_table(df, table_name, engine):

    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False
    )

    print(f"Tabla {table_name} cargada correctamente")
