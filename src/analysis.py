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

def encontrar_duplicados(df: pd.DataFrame, subset=None, n=10):
    print("===== ANÁLISIS DE DUPLICADOS =====")
    
    total = len(df)
    
    # Detectar duplicados (incluye todas las columnas si subset=None)
    duplicados_mask = df.duplicated(subset=subset, keep=False)
    
    total_duplicados = duplicados_mask.sum()
    duplicados_unicos = df.duplicated(subset=subset).sum()
    
    print(f"Total de registros: {total}")
    print(f"Filas duplicadas (incluyendo repetidas): {total_duplicados}")
    print(f"Duplicados únicos a eliminar: {duplicados_unicos}")
    
    return df[duplicados_mask]

def eliminar_duplicados(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    print("===== LIMPIEZA DE DUPLICADOS =====")
    
    num_antes = len(df)
    
    duplicados = df.duplicated().sum()
    
    print(f"Duplicados detectados: {duplicados}")
    
    if duplicados == 0:
        print("No se encontraron duplicados.")
        return df
    
    # Eliminación
    df_limpio = df.drop_duplicates(subset=subset)
    
    num_despues = len(df_limpio)
    eliminados = num_antes - num_despues
    
    print(f"Registros eliminados: {eliminados}")
    print(f"Total final: {num_despues}")
    
    return df_limpio
