import numpy as np
import pandas as pd
import re


def _object_a_str_python(serie: pd.Series) -> pd.Series:
    """Convierte object a strings de Python (dtype object); evita dtype numpy str en pandas recientes."""
    out = []
    for v in serie:
        if pd.isna(v):
            out.append(np.nan)
        elif isinstance(v, str):
            out.append(v)
        else:
            out.append(str(v))
    return pd.Series(out, index=serie.index, dtype=object)

def estandarizar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza los nombres de las columnas: minúsculas, cambia espacios por '_'
    y quita caracteres especiales para que sean más estables al procesar.
    """
    # Usar .rename() pero con una lambda compleja o reconstruyendo .columns
    nuevas_cols = []
    for col in df.columns:
        # Convertir a minúsculas y quitar espacios en extremos
        col_new = str(col).strip().lower()
        # Cambiar espacios intermedios por guion bajo
        col_new = re.sub(r'\s+', '_', col_new)
        # Quitar acentos
        col_new = re.sub(r'[áäâà]', 'a', col_new)
        col_new = re.sub(r'[éëêè]', 'e', col_new)
        col_new = re.sub(r'[íïîì]', 'i', col_new)
        col_new = re.sub(r'[óöôò]', 'o', col_new)
        col_new = re.sub(r'[úüûù]', 'u', col_new)
        # Quitar cualquier carácter que no sea alfanumérico o guion bajo
        col_new = re.sub(r'[^a-z0-9_]', '', col_new)
        nuevas_cols.append(col_new)
        
    df.columns = nuevas_cols
    return df

def ajustar_tipos_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte tipos de datos para optimizar en memoria DataFrames grandes.
    'object' -> 'category' (si hay poca variabilidad relativa)
    'float64' -> 'float32'
    'int64' -> 'int32'
    """
    n_filas = len(df)
    
    for col in df.columns:
        dtype = str(df[col].dtype)

        # Columnas object (incl. variantes): homogeneizar a str Python para Parquet.
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = _object_a_str_python(df[col])
                
        # Downcast de numéricos
        elif dtype.startswith('float'):
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        elif dtype.startswith('int'):
             # Ignorar booleanos o similares
             df[col] = pd.to_numeric(df[col], downcast='integer')
             
    return df