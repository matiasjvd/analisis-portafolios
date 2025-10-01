import json
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st
import altair as alt
import unicodedata

# ==============================================
# Utilidades de transformación (basado en json_transformer.py)
# ==============================================

def normalize_percent(value: Optional[float]) -> float:
    """Convierte pesos en [0,1] a %; si ya vienen en %, los deja; maneja None como 0."""
    if value is None:
        return 0.0
    try:
        v = float(value)
    except Exception:
        return 0.0
    return v * 100.0 if abs(v) <= 1.0 else v


def json_to_dataframe(data: dict) -> pd.DataFrame:
    """
    Convierte el JSON de clientes/portafolios a un DataFrame tabular con filas por holding
    y columnas para métricas a nivel portafolio, asset class y activo.
    Incluye campos opcionales: Geography, Subclass, Market si vienen en el JSON.
    """
    rows = []
    clients = data.get('clients', [])

    for cliente in clients:
        for p in cliente.get('portfolios', []):
            # Valor de portafolio (si falta, suma asset classes/holdings)
            p_val = p.get('value')
            if p_val is None:
                tmp = 0.0
                for ac in p.get('asset_types', []):
                    ac_val = ac.get('value')
                    if ac_val is None:
                        ac_val = sum((h.get('value') or 0) for h in ac.get('holdings', []))
                    tmp += ac_val or 0
                p_val = tmp

            # Peso portafolio normalizado a %
            p_weight = normalize_percent(p.get('weight'))

            for ac in p.get('asset_types', []):
                # Valor asset class (si falta, suma holdings)
                ac_val = ac.get('value')
                if ac_val is None:
                    ac_val = sum((h.get('value') or 0) for h in ac.get('holdings', []))

                # Peso asset class normalizado a %
                ac_weight = normalize_percent(ac.get('weight'))

                for h in ac.get('holdings', []):
                    rows.append({
                        "Cliente":                cliente.get('name'),
                        "Portafolio":             p.get('name'),
                        "Asset_Class":            ac.get('type_id'),
                        # ID estable del activo para merges con diccionario
                        "Asset_ID":               h.get('asset_id'),
                        # Por defecto usamos el ID como nombre; puede mapearse luego
                        "Asset":                  h.get('asset_id'),

                        # Dimensiones opcionales en holdings (si están en el JSON)
                        "Geography":             h.get('geography'),
                        "Subclass":              h.get('subclass'),
                        "Market":                h.get('market'),

                        # Portafolio
                        "Portfolio_Value":        p_val,
                        "Portfolio_Weight(%)":    p_weight,
                        "Portfolio_YTD":          p.get('ytd'),
                        "Portfolio_MTD":          p.get('mtd'),
                        "Portfolio_D1":           p.get('d1'),

                        # Asset class
                        "AC_Value":               ac_val,
                        "AC_Weight(%)":           ac_weight,
                        "AC_YTD":                 ac.get('ytd'),
                        "AC_MTD":                 ac.get('mtd'),
                        "AC_D1":                  ac.get('d1'),

                        # Activo
                        "Value":                  h.get('value'),
                        "Weight_Asset(%)":        normalize_percent(h.get('weight')),
                        "YTD":                    h.get('ytd'),
                        "MTD":                    h.get('mtd'),
                        "D1":                     h.get('d1'),
                    })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Numéricos + NaN -> 0
    num_cols = [
        "Portfolio_Value","Portfolio_Weight(%)","Portfolio_YTD","Portfolio_MTD","Portfolio_D1",
        "AC_Value","AC_Weight(%)","AC_YTD","AC_MTD","AC_D1",
        "Value","Weight_Asset(%)","YTD","MTD","D1"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Orden de columnas
    ordered_cols = [
        "Cliente","Portafolio","Asset_Class","Asset_ID","Asset","Geography","Subclass","Market",
        "Portfolio_Value","Portfolio_Weight(%)","Portfolio_YTD","Portfolio_MTD","Portfolio_D1",
        "AC_Value","AC_Weight(%)","AC_YTD","AC_MTD","AC_D1",
        "Value","Weight_Asset(%)","YTD","MTD","D1"
    ]
    df = df[[c for c in ordered_cols if c in df.columns]]

    return df


@st.cache_data(show_spinner=False)
def load_json(uploaded_file) -> dict:
    """Lee un archivo JSON subido por Streamlit y lo devuelve como dict."""
    try:
        uploaded_file.seek(0)
        return json.load(uploaded_file)
    except Exception as e:
        st.error(f"No se pudo leer el JSON: {e}")
        return {}


@st.cache_data(show_spinner=False)
def load_dictionary_xlsx(uploaded_file) -> pd.DataFrame:
    """Lee diccionario desde Excel. No se aplica por defecto (IDs se mantienen)."""
    try:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.warning(f"No se pudo leer el diccionario Excel: {e}")
        return pd.DataFrame()


def compute_comparisons(df: pd.DataFrame, targets: List[Tuple[str, str]]):
    """Calcula pivotes por Asset_Class (valor y %) y métricas a nivel portafolio para los targets."""
    if not targets:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    targets_df = pd.DataFrame(targets, columns=["Cliente", "Portafolio"])
    df_cmp = df.merge(targets_df, on=["Cliente", "Portafolio"], how="inner")

    if df_cmp.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Composición por Asset_Class usando métricas de categoría del JSON (sin sumar holdings)
    comp = (
        df_cmp.sort_values(["Cliente","Portafolio","Asset_Class"]) \
              .groupby(["Cliente","Portafolio","Asset_Class"], as_index=False) \
              .agg(
                  AC_Value=("AC_Value", "first"),
                  AC_Weight_pct=("AC_Weight(%)", "first"),
                  AC_YTD=("AC_YTD", "first"),
                  AC_MTD=("AC_MTD", "first"),
                  AC_D1=("AC_D1", "first"),
              )
    )

    # Pivotes: valores y % provistos por categoría directamente
    pivot_value = comp.pivot(index="Asset_Class", columns=["Cliente","Portafolio"], values="AC_Value").fillna(0)
    pivot_pct   = comp.pivot(index="Asset_Class", columns=["Cliente","Portafolio"], values="AC_Weight_pct").fillna(0)

    # Métricas a nivel Portafolio (tomar primeras por grupo)
    metrics = (
        df_cmp.sort_values(["Cliente","Portafolio"])  # asegurar orden estable
              .groupby(["Cliente","Portafolio"], as_index=False)
              .agg(
                  Portfolio_Value=("Portfolio_Value","first"),
                  Portfolio_Weight_pct=("Portfolio_Weight(%)","first"),
                  YTD=("Portfolio_YTD","first"),
                  MTD=("Portfolio_MTD","first"),
                  D1 =("Portfolio_D1","first"),
              )
    )

    # Redondeos para lectura (porcentajes a 1 decimal; retornos se dejan con mayor precisión y se formatean al mostrar)
    pivot_pct  = pivot_pct.round(1)
    metrics[["Portfolio_Weight_pct","YTD","MTD","D1"]] = metrics[["Portfolio_Weight_pct","YTD","MTD","D1"]].round(6)

    return pivot_value, pivot_pct, metrics


def compute_comp_long(df: pd.DataFrame, targets: List[Tuple[str, str]]) -> pd.DataFrame:
    """Devuelve composición larga por Asset_Class con % por portafolio (para gráficos 100%)."""
    if not targets:
        return pd.DataFrame()
    targets_df = pd.DataFrame(targets, columns=["Cliente", "Portafolio"])
    df_cmp = df.merge(targets_df, on=["Cliente", "Portafolio"], how="inner")
    if df_cmp.empty:
        return pd.DataFrame()

    # Usar métricas de categoría del JSON para % por Asset_Class
    comp = (
        df_cmp.sort_values(["Cliente","Portafolio","Asset_Class"]) \
              .groupby(["Cliente","Portafolio","Asset_Class"], as_index=False) \
              .agg(
                  AC_Value=("AC_Value", "first"),
                  AC_Weight_pct=("AC_Weight(%)", "first"),
              )
    )
    comp["Pct"] = pd.to_numeric(comp["AC_Weight_pct"], errors="coerce").fillna(0) / 100.0
    comp["PortfolioKey"] = comp.apply(lambda r: f"{r['Cliente']} | {r['Portafolio']}", axis=1)
    return comp[["PortfolioKey","Asset_Class","Pct"]]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana columnas MultiIndex a 'Cliente | Portafolio'."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" | ".join([str(x) for x in c]) for c in df.columns.to_list()]
    return df


def pivot_to_long_for_chart(pivot_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convierte pivot ancho (index=Asset_Class, cols=Cliente|Portafolio) a formato largo para chart."""
    flat = flatten_columns(pivot_df)
    flat = flat.reset_index()  # Asset_Class como columna
    long_df = flat.melt(id_vars=["Asset_Class"], var_name="PortfolioKey", value_name=value_name)
    return long_df


def combine_weight_and_metric_flat(weight_df: pd.DataFrame, metric_df: pd.DataFrame, metric_label: str) -> pd.DataFrame:
    """Combina pivotes: deja 'Peso (%)' fijo y agrega la métrica seleccionada como segunda subcolumna por portafolio.
    - Entrada: weight_df y metric_df con mismo índice (p. ej., Asset_Class o Asset) y mismas columnas (portafolios MultiIndex).
    - Salida: DataFrame con columnas planas: "Cliente | Portafolio | Peso (%)" y "Cliente | Portafolio | <metric_label>".
    """
    # Alinear índice y columnas
    all_index = weight_df.index.union(metric_df.index)
    weight = weight_df.reindex(all_index).fillna(0)
    metric = metric_df.reindex(all_index).fillna(0)

    out = pd.DataFrame(index=all_index)
    if isinstance(weight.columns, pd.MultiIndex):
        cols = list(weight.columns)
        for col in cols:
            port_str = " | ".join([str(x) for x in col])
            out[f"{port_str} | Peso (%)"] = weight[col]
            out[f"{port_str} | {metric_label}"] = metric[col] if col in metric.columns else 0
    else:
        cols = list(weight.columns)
        for col in cols:
            out[f"{col} | Peso (%)"] = weight[col]
            out[f"{col} | {metric_label}"] = metric[col] if col in metric.columns else 0
    return out


def chart_values(long_df: pd.DataFrame) -> alt.Chart:
    """Barras lado a lado por Valor."""
    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X('Asset_Class:N', title='Asset Class'),
            y=alt.Y('Value:Q', title='Valor'),
            color=alt.Color('PortfolioKey:N', title='Portafolio'),
            xOffset='PortfolioKey:N',
            tooltip=['Asset_Class','PortfolioKey','Value']
        )
        .properties(height=400)
    )
    return chart


def chart_percent_by_assetclass(long_df: pd.DataFrame) -> alt.Chart:
    """Barras apiladas por Asset_Class (% del portafolio por Asset_Class)."""
    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X('Asset_Class:N', title='Asset Class'),
            y=alt.Y('Pct:Q', title='% del Portafolio', axis=alt.Axis(format='.1%')),
            color=alt.Color('PortfolioKey:N', title='Portafolio'),
            tooltip=['Asset_Class','PortfolioKey',alt.Tooltip('Pct:Q', format='.1%')]
        )
        .properties(height=400)
    )
    return chart


def chart_percent_by_portfolio(comp_long: pd.DataFrame) -> alt.Chart:
    """Barras horizontales agrupadas por Asset Class (no apiladas), comparando portafolios."""
    chart = (
        alt.Chart(comp_long)
        .mark_bar()
        .encode(
            y=alt.Y('Asset_Class:N', title='Asset Class'),
            x=alt.X('Pct:Q', title='% del Portafolio', axis=alt.Axis(format='.1%')),
            color=alt.Color('PortfolioKey:N', title='Portafolio'),
            yOffset='PortfolioKey:N',
            tooltip=['Asset_Class','PortfolioKey',alt.Tooltip('Pct:Q', format='.1%')]
        )
        .properties(height=max(240, 26 * comp_long['Asset_Class'].nunique()))
    )
    return chart


def compute_asset_details(df: pd.DataFrame, targets: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Agrega por (Cliente, Portafolio, Asset_Class, Asset):
    - Asset_Value: suma de valores
    - YTD_w, MTD_w, D1_w: retornos ponderados por valor
    - Pct_in_AC(%): porcentaje del activo dentro de su Asset_Class
    - Pct_in_Portfolio(%): porcentaje del activo sobre el total del portafolio
    - Incluye columnas de dimensión representativas por activo: Geography, Subclass, Market (si existen)
      usando el ítem con mayor peso en Valor dentro del grupo.
    """
    if not targets:
        return pd.DataFrame()

    targets_df = pd.DataFrame(targets, columns=["Cliente", "Portafolio"])
    df_cmp = df.merge(targets_df, on=["Cliente", "Portafolio"], how="inner")
    if df_cmp.empty:
        return pd.DataFrame()

    # Agregación al nivel de activo
    keys = ["Cliente","Portafolio","Asset_Class","Asset"]

    # Dimensiones potenciales presentes en el DataFrame
    dim_cols = [c for c in ["Geography","Market"] if c in df_cmp.columns]

    def agg_group(g: pd.DataFrame) -> pd.Series:
        total_val = g['Value'].sum()
        if total_val == 0:
            ytd = mtd = d1 = 0.0
        else:
            ytd = float((g['Value'] * g['YTD']).sum() / total_val)
            mtd = float((g['Value'] * g['MTD']).sum() / total_val)
            d1  = float((g['Value'] * g['D1']).sum()  / total_val)
        # Selección representativa de dimensiones por mayor peso en Valor
        dims = {}
        for col in dim_cols:
            v = g[[col, 'Value']].copy()
            # Normalizar valores de dimensión: quitar espacios y marcar vacíos como NA
            v[col] = v[col].astype(str).str.strip()
            v[col] = v[col].replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, 'NONE': pd.NA, 'NaN': pd.NA})
            v = v.dropna(subset=[col])
            if v.empty:
                rep = '(Sin dato)'
            else:
                by_val = v.groupby(col, as_index=False)['Value'].sum().sort_values('Value', ascending=False)
                rep = by_val.iloc[0][col]
            dims[col] = rep
        return pd.Series({
            'Asset_Value': total_val,
            'YTD_w': ytd,
            'MTD_w': mtd,
            'D1_w': d1,
            **dims,
        })

    asset_df = df_cmp.groupby(keys, as_index=False).apply(agg_group).reset_index(drop=True)

    # Totales por Asset_Class para % dentro del AC
    ac_totals = asset_df.groupby(["Cliente","Portafolio","Asset_Class"], as_index=False).agg(AC_Total_Value=("Asset_Value","sum"))
    asset_df = asset_df.merge(ac_totals, on=["Cliente","Portafolio","Asset_Class"], how="left")
    asset_df['Pct_in_AC(%)'] = (asset_df['Asset_Value'] / asset_df['AC_Total_Value']).fillna(0) * 100

    # Totales por Portafolio para % relativo al portafolio completo
    port_totals = asset_df.groupby(["Cliente","Portafolio"], as_index=False).agg(Port_Total_Value=("Asset_Value","sum"))
    asset_df = asset_df.merge(port_totals, on=["Cliente","Portafolio"], how="left")
    asset_df['Pct_in_Portfolio(%)'] = (asset_df['Asset_Value'] / asset_df['Port_Total_Value']).fillna(0) * 100

    # Normalizar dimensiones para visualización
    for c in dim_cols:
        asset_df[c] = asset_df[c].fillna('(Sin dato)').astype(str).str.strip()

    asset_df["PortfolioKey"] = asset_df.apply(lambda r: f"{r['Cliente']} | {r['Portafolio']}", axis=1)
    return asset_df


def compute_group_breakdown(df: pd.DataFrame, targets: List[Tuple[str, str]], group_field: str) -> pd.DataFrame:
    """
    Agrega por (Cliente, Portafolio, Asset_Class, group_field):
    - Group_Value: suma de valores
    - YTD_w, MTD_w, D1_w: retornos ponderados por valor
    - Pct_in_Portfolio(%): porcentaje del grupo sobre el total del Asset Class
    """
    if not targets or group_field not in df.columns:
        return pd.DataFrame()

    targets_df = pd.DataFrame(targets, columns=["Cliente", "Portafolio"])
    df_cmp = df.merge(targets_df, on=["Cliente", "Portafolio"], how="inner")
    if df_cmp.empty:
        return pd.DataFrame()

    # Normalizar nombre del item del grupo
    df_cmp = df_cmp.copy()
    # Normalizamos valores de la dimensión para evitar diferencias por espacios/casing
    df_cmp[group_field] = df_cmp[group_field].fillna('(Sin dato)').astype(str).str.strip()

    keys = ["Cliente","Portafolio","Asset_Class", group_field]

    def agg_group(g: pd.DataFrame) -> pd.Series:
        total_val = g['Value'].sum()
        if total_val == 0:
            ytd = mtd = d1 = 0.0
        else:
            ytd = float((g['Value'] * g['YTD']).sum() / total_val)
            mtd = float((g['Value'] * g['MTD']).sum() / total_val)
            d1  = float((g['Value'] * g['D1']).sum()  / total_val)
        return pd.Series({
            'Group_Value': total_val,
            'YTD_w': ytd,
            'MTD_w': mtd,
            'D1_w': d1,
        })

    gdf = df_cmp.groupby(keys, as_index=False).apply(agg_group).reset_index(drop=True)

    # Totales por Asset_Class dentro del portafolio (para % relativo al AC)
    ac_totals = gdf.groupby(["Cliente","Portafolio","Asset_Class"], as_index=False).agg(AC_Total_Value=("Group_Value","sum"))
    gdf = gdf.merge(ac_totals, on=["Cliente","Portafolio","Asset_Class"], how="left")
    gdf['Pct_in_Portfolio(%)'] = (gdf['Group_Value'] / gdf['AC_Total_Value']).fillna(0) * 100

    gdf["PortfolioKey"] = gdf.apply(lambda r: f"{r['Cliente']} | {r['Portafolio']}", axis=1)
    gdf = gdf.rename(columns={group_field: 'GroupItem'})
    gdf['GroupField'] = group_field
    return gdf


def compute_ac_summary(df: pd.DataFrame, targets: List[Tuple[str, str]]) -> pd.DataFrame:
    """Resumen por (Cliente, Portafolio, Asset_Class): valor total, peso de categoría y retornos provistos por el JSON."""
    if not targets:
        return pd.DataFrame()
    targets_df = pd.DataFrame(targets, columns=["Cliente", "Portafolio"])
    df_cmp = df.merge(targets_df, on=["Cliente", "Portafolio"], how="inner")
    if df_cmp.empty:
        return pd.DataFrame()

    def agg_ac(g: pd.DataFrame) -> pd.Series:
        total_val = g['Value'].sum()
        def _first_or_zero(col: str) -> float:
            if col in g.columns and len(g[col]) > 0:
                try:
                    return float(g[col].iloc[0])
                except Exception:
                    return 0.0
            return 0.0
        ytd = _first_or_zero('AC_YTD')
        mtd = _first_or_zero('AC_MTD')
        d1  = _first_or_zero('AC_D1')
        w   = _first_or_zero('AC_Weight(%)')  # peso de la categoría provisto por el JSON (ya en %)
        return pd.Series({
            'AC_Total_Value': total_val,
            'AC_Weight_pct': w,
            'AC_YTD': ytd,
            'AC_MTD': mtd,
            'AC_D1': d1,
        })

    ac_summary = (
        df_cmp.groupby(["Cliente","Portafolio","Asset_Class"], as_index=False)
              .apply(agg_ac)
              .reset_index(drop=True)
    )
    ac_summary["PortfolioKey"] = ac_summary.apply(lambda r: f"{r['Cliente']} | {r['Portafolio']}", axis=1)
    return ac_summary


# ==============================================
# UI Streamlit
# ==============================================
st.set_page_config(page_title='Comparador de Portafolios', layout='wide')

st.title('Comparador de Portafolios')
st.markdown(
    "Carga un archivo JSON con la estructura de clientes/portafolios y un diccionario Excel (opcional). "
    "Selecciona múltiples portafolios para comparar y visualiza composición por Asset_Class en valor y en porcentaje."
)

col1, col2 = st.columns(2)
with col1:
    json_file = st.file_uploader("Subir JSON de clientes/portafolios", type=["json"], accept_multiple_files=False)
with col2:
    dict_file = st.file_uploader("Subir diccionario (Excel .xlsx) — opcional", type=["xlsx"], accept_multiple_files=False)

apply_dict = st.checkbox("Aplicar diccionario para nombres (IDs → nombres)", value=True, help="Por defecto se aplicará el diccionario si está disponible.")

# Cargar datos
if json_file is None:
    st.info("Sube el archivo JSON para comenzar.")
    st.stop()

raw_data = load_json(json_file)
if not raw_data:
    st.stop()

df = json_to_dataframe(raw_data)
if df.empty:
    st.warning("No se obtuvieron filas a partir del JSON (¿faltan holdings?).")
    st.stop()

# Diccionario (opcional). No aplicar por defecto, solo previsualizar.
dict_df = pd.DataFrame()
if dict_file is not None:
    dict_df = load_dictionary_xlsx(dict_file)
    with st.expander("Vista previa diccionario (primeras filas)"):
        st.dataframe(dict_df.head(20))

    if apply_dict and not dict_df.empty:
        cols = [c.lower() for c in dict_df.columns]

        # 1) Mapeo de assets (asset_id/ticker -> nombre) con normalización y sin duplicar filas
        #    Soporta diccionarios con clave 'Asset_ID' o 'Ticker'.
        if ('nombre' in cols) and 'Asset' in df.columns and (('asset_id' in cols) or ('ticker' in cols)):
            # Preferir 'asset_id' si existe, de lo contrario usar 'ticker'
            if 'asset_id' in cols:
                col_key = dict_df.columns[[c.lower()=="asset_id" for c in dict_df.columns]][0]
            else:
                col_key = dict_df.columns[[c.lower()=="ticker" for c in dict_df.columns]][0]
            col_nombre = dict_df.columns[[c.lower()=="nombre" for c in dict_df.columns]][0]

            name_df = dict_df[[col_key, col_nombre]].copy()
            name_df[col_key] = name_df[col_key].astype(str).str.strip()
            # preferir primer nombre no vacío por clave
            name_df[col_nombre] = name_df[col_nombre].astype(str).str.strip()
            name_df = name_df[name_df[col_nombre] != ""]
            name_df = name_df.drop_duplicates(subset=[col_key], keep='first')
            asset_map = dict(zip(name_df[col_key], name_df[col_nombre]))

            # Normalizar clave en df (suponemos que Asset_ID del JSON coincide con Ticker/Asset_ID del diccionario)
            if 'Asset_ID' in df.columns:
                df['Asset_ID'] = df['Asset_ID'].astype(str).str.strip()
                mapped = df['Asset_ID'].map(asset_map)
                df['Asset'] = mapped.where(mapped.notna(), df['Asset'].astype(str))
            else:
                df['Asset'] = df['Asset'].astype(str)

        # 2) Mapeo de asset class (type_id -> nombre) — deshabilitado salvo columna dedicada
        # Evitamos pisar Asset_Class con 'Nombre' de activos. Si hubiera una columna específica,
        # p. ej. 'Nombre Asset Class', se podría activar.
        if {'type_id','nombre asset class'}.issubset(set(cols)) and 'Asset_Class' in df.columns:
            col_type_id = dict_df.columns[[c.lower()=="type_id" for c in dict_df.columns]][0]
            col_ac_name = dict_df.columns[[c.lower()=="nombre asset class" for c in dict_df.columns]][0]
            ac_df = dict_df[[col_type_id, col_ac_name]].copy()
            ac_df[col_type_id] = ac_df[col_type_id].astype(str).str.strip()
            ac_df = ac_df.drop_duplicates(subset=[col_type_id], keep='first')
            ac_map = dict(zip(ac_df[col_type_id], ac_df[col_ac_name]))
            df['Asset_Class'] = df['Asset_Class'].astype(str).str.strip().map(lambda x: ac_map.get(x, x))

        # 3) Completar Geography/Subclass/Market desde diccionario si existen columnas
        #    El diccionario puede traer variantes en nombres (p. ej., 'Geography/Subclass', 'Geography / Subclass').
        def _norm_col(s: str) -> str:
            s = str(s).lower()
            # quitar acentos
            s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
            # dejar solo alfanumérico para comparar robusto: geography/subclass -> geographysubclass
            return "".join(ch for ch in s if ch.isalnum())

        norm_cols = {_norm_col(c): c for c in dict_df.columns}
        # Usar como clave 'Asset_ID' o, si no existe, 'Ticker'
        key_col = norm_cols.get('assetid') or norm_cols.get('ticker')
        if key_col and 'Asset_ID' in df.columns:
            # Normalizar claves para evitar fallos de merge por tipo/espacios
            df['Asset_ID'] = df['Asset_ID'].astype(str).str.strip()

            # Mapeo de fuentes posibles en el diccionario -> columnas objetivo en el DataFrame
            # - 'Geography / Subclass' ahora alimenta Geography por defecto (prioridad geográfica)
            dim_sources_norm = {
                'Geography': ['geography', 'geographysubclass'],
                'Subclass':  ['subclass'],
                'Market':    ['market']
            }
            for target_col, candidates_norm in dim_sources_norm.items():
                # Buscar la primera columna presente en el diccionario que calce con los candidatos normalizados
                source_key_norm = next((k for k in candidates_norm if k in norm_cols), None)
                if not source_key_norm:
                    continue
                src_col = norm_cols[source_key_norm]
                tmp_col = f"{target_col}_dict"

                # Construir sub_map SOLO por clave de activo para evitar no-matches por 'Portafolio'
                sub_map = dict_df[[key_col, src_col]].copy()
                sub_map[key_col] = sub_map[key_col].astype(str).str.strip()
                sub_map[src_col] = sub_map[src_col].astype(str).str.strip()

                # Eliminar valores vacíos y renombrar columnas a las de df
                sub_map = sub_map[sub_map[src_col] != ""]
                sub_map = sub_map.rename(columns={key_col: 'Asset_ID', src_col: tmp_col})

                # Evitar multiplicación de filas: garantizar unicidad por clave
                sub_map = sub_map.drop_duplicates(subset=['Asset_ID'], keep='first')

                df = df.merge(sub_map, on=['Asset_ID'], how='left')
                if target_col in df.columns:
                    df[target_col] = df[target_col].fillna(df[tmp_col])
                else:
                    df[target_col] = df[tmp_col]
                if tmp_col in df.columns:
                    df.drop(columns=[tmp_col], inplace=True)

# Selección de portafolios (múltiples)
all_pairs_labels = (
    df[['Cliente','Portafolio']]
    .drop_duplicates()
    .sort_values(['Cliente','Portafolio'])
    .apply(lambda r: f"{r['Cliente']} | {r['Portafolio']}", axis=1)
    .tolist()
)

selected_labels = st.multiselect(
    "Selecciona portafolios (2 a N)", options=all_pairs_labels, default=[]
)

if len(selected_labels) < 2:
    st.info("Selecciona al menos 2 portafolios para comparar.")
    st.stop()

# Convertir etiquetas a tuplas (Cliente, Portafolio)
selected_pairs: List[Tuple[str, str]] = [tuple(lbl.split(" | ", 1)) for lbl in selected_labels]

pivot_value, pivot_pct, metrics = compute_comparisons(df, selected_pairs)

if pivot_value.empty or pivot_pct.empty:
    st.warning("No hay datos para los portafolios seleccionados.")
    st.stop()

# Mostrar tablas generales
st.subheader("Tablas de Composición y Métricas")

# Helpers de formateo en %:
# - _fmt_pct_abs: ya está en escala porcentaje (0–100) → muestra 1 decimal
# - _fmt_pct_ratio: está en proporción (−1..1) → multiplica por 100 y muestra 1 decimal
def _fmt_pct_abs(x):
    try:
        v = float(x)
    except Exception:
        return x
    if pd.isna(v):
        return ""
    return f"{v:,.1f}%"

def _fmt_pct_ratio(x):
    try:
        v = float(x)
    except Exception:
        return x
    if pd.isna(v):
        return ""
    return f"{v*100.0:,.1f}%"

# Helper para formateo monetario
def _fmt_money(x):
    try:
        v = float(x)
    except Exception:
        return x
    if pd.isna(v):
        return ""
    return f"${v:,.0f}"

# Tablas por métrica: menos abrumadoras mediante pestañas
st.markdown("**Composición por Asset_Class**")

# Preparar pivotes de % y valor
pct_df = pivot_pct.copy()
val_df = pivot_value.copy()

# Retornos provistos por categoría (YTD/MTD/D1)
ac_summary_top = compute_ac_summary(df, selected_pairs)
if not ac_summary_top.empty:
    piv_ytd = ac_summary_top.pivot(index="Asset_Class", columns=["Cliente","Portafolio"], values="AC_YTD").fillna(0)
    piv_mtd = ac_summary_top.pivot(index="Asset_Class", columns=["Cliente","Portafolio"], values="AC_MTD").fillna(0)
    piv_d1  = ac_summary_top.pivot(index="Asset_Class", columns=["Cliente","Portafolio"], values="AC_D1" ).fillna(0)
else:
    piv_ytd = pd.DataFrame(index=pct_df.index)
    piv_mtd = pd.DataFrame(index=pct_df.index)
    piv_d1  = pd.DataFrame(index=pct_df.index)

# Alinear índices entre todas las tablas existentes
all_idx = set(pct_df.index)
if not val_df.empty: all_idx = all_idx.union(val_df.index)
if not piv_ytd.empty: all_idx = all_idx.union(piv_ytd.index)
if not piv_mtd.empty: all_idx = all_idx.union(piv_mtd.index)
if not piv_d1.empty:  all_idx = all_idx.union(piv_d1.index)
all_idx = sorted(all_idx)

if not pct_df.empty: pct_df = pct_df.reindex(all_idx).fillna(0)
if not val_df.empty: val_df = val_df.reindex(all_idx).fillna(0)
if not piv_ytd.empty: piv_ytd = piv_ytd.reindex(all_idx).fillna(0)
if not piv_mtd.empty: piv_mtd = piv_mtd.reindex(all_idx).fillna(0)
if not piv_d1.empty:  piv_d1  = piv_d1.reindex(all_idx).fillna(0)

# Selector de retorno para mostrar junto con Peso fijo
metric_option = st.selectbox(
    "Retorno a mostrar junto con Peso (%):",
    options=["YTD", "MTD", "D1"],
    index=0,
)

metric_map = {
    "YTD": piv_ytd,
    "MTD": piv_mtd,
    "D1":  piv_d1,
}
sel_metric_df = metric_map.get(metric_option, pd.DataFrame()).copy()

# Combinación Peso fijo + Métrica seleccionada, con subcolumnas por portafolio
comb_df = combine_weight_and_metric_flat(pct_df, sel_metric_df, metric_option)
show = comb_df.reset_index().rename(columns={comb_df.index.name or 'index': 'Asset_Class'})
num_cols = show.select_dtypes(include=['number']).columns
fmt_map = {}
for c in num_cols:
    if 'Valor' in c:
        fmt_map[c] = _fmt_money
    elif 'Peso (%)' in c or 'Weight' in c or 'Pct' in c:
        fmt_map[c] = _fmt_pct_abs
    elif any(k in c for k in ['YTD','MTD','D1']):
        fmt_map[c] = _fmt_pct_ratio
    else:
        fmt_map[c] = _fmt_pct_abs
st.dataframe(show.style.format(fmt_map))

st.markdown("**Métricas de Portafolio**")
# Formateo: Portfolio_Value en moneda; pesos (0–100) con _fmt_pct_abs; retornos (−1..1) con _fmt_pct_ratio
if not metrics.empty:
    met_df = metrics.copy()
    fmt = {}
    if 'Portfolio_Value' in met_df.columns:
        fmt['Portfolio_Value'] = _fmt_money
    if 'Portfolio_Weight_pct' in met_df.columns:
        fmt['Portfolio_Weight_pct'] = _fmt_pct_abs
    for col in ['YTD','MTD','D1']:
        if col in met_df.columns:
            fmt[col] = _fmt_pct_ratio
    st.dataframe(met_df.style.format(fmt))
else:
    st.dataframe(metrics)

# Botones de descarga
csv_value = flatten_columns(pivot_value).to_csv(index=False).encode('utf-8')
csv_pct   = flatten_columns(pivot_pct).to_csv(index=False).encode('utf-8')
csv_met   = metrics.to_csv(index=False).encode('utf-8')

colD1, colD2, colD3 = st.columns(3)
with colD1:
    st.download_button("Descargar VALOR (CSV)", csv_value, file_name="composicion_valor.csv", mime="text/csv")
with colD2:
    st.download_button("Descargar % (CSV)", csv_pct, file_name="composicion_pct.csv", mime="text/csv")
with colD3:
    st.download_button("Descargar Métricas (CSV)", csv_met, file_name="metricas_portafolio.csv", mime="text/csv")

# Gráficos deshabilitados temporalmente a pedido del usuario
# st.subheader("Composición 100% por Portafolio")
# comp_long = compute_comp_long(df, selected_pairs)
# if comp_long.empty:
#     st.info("No hay datos para los portafolios seleccionados.")
# else:
#     # Barras verticales 100% por portafolio, comparando categoría (Asset_Class) dentro de cada barra
#     st.altair_chart(chart_percent_by_portfolio(comp_long), use_container_width=True)

# ===========================
# Desglose por Asset Class (activos: peso en portafolio y retornos)
# y por Geography / Subclass / Market
# ===========================
st.subheader("Desglose por Asset Class (activos y dimensiones: Geography, Subclass, Market)")

asset_details = compute_asset_details(df, selected_pairs)
geo_break = compute_group_breakdown(df, selected_pairs, 'Geography') if 'Geography' in df.columns else pd.DataFrame()
sub_break = compute_group_breakdown(df, selected_pairs, 'Subclass')  if 'Subclass'  in df.columns else pd.DataFrame()
mar_break = compute_group_breakdown(df, selected_pairs, 'Market')    if 'Market'    in df.columns else pd.DataFrame()

# Resumen por AC
ac_summary = compute_ac_summary(df, selected_pairs)

# Ordenar Asset Class por tamaño total agregado (suma sobre portafolios)
if not ac_summary.empty:
    ac_order = (
        ac_summary.groupby('Asset_Class', as_index=False)
        .agg(_sum=('AC_Total_Value', 'sum'))
        .sort_values('_sum', ascending=False)['Asset_Class']
        .tolist()
    )
else:
    ac_order = sorted(df['Asset_Class'].dropna().unique().tolist())

for ac in ac_order:
    with st.expander(f"Asset Class: {ac}", expanded=False):
        # Resumen por portafolio para el AC
        ac_res = ac_summary[ac_summary['Asset_Class'] == ac].copy()
        if not ac_res.empty:
            res_cols = ['PortfolioKey','AC_Weight_pct','AC_Total_Value','AC_YTD','AC_MTD','AC_D1']
            st.markdown("**Resumen por Portafolio**")
            # Formatear: peso (0–100) con 1 decimal; valor en moneda; retornos (−1..1) con 1 decimal
            fmt_cols = {
                'AC_Weight_pct': _fmt_pct_abs,
                'AC_Total_Value': _fmt_money,
                'AC_YTD': _fmt_pct_ratio,
                'AC_MTD': _fmt_pct_ratio,
                'AC_D1': _fmt_pct_ratio,
            }
            st.dataframe(ac_res[res_cols].sort_values('AC_Total_Value', ascending=False).reset_index(drop=True).style.format(fmt_cols))

        # Tabs principales por dimensión
        top_tabs = st.tabs(["Activos", "Geography", "Market"])

        # --- Activos ---
        with top_tabs[0]:
            det_ac = asset_details[asset_details['Asset_Class'] == ac] if not asset_details.empty else pd.DataFrame()
            if det_ac.empty:
                st.info("Sin datos de activos para este Asset Class.")
            else:
                # Tabla base con dimensiones (Geography / Market) por activo y portafolio. Se quita Subclass.
                # Añadimos el peso de la categoría (AC) reportado por el JSON para cada portafolio.
                if not ac_res.empty and 'AC_Weight_pct' in ac_res.columns:
                    det_ac = det_ac.merge(ac_res[['PortfolioKey','AC_Weight_pct']].drop_duplicates(), on='PortfolioKey', how='left')
                    det_ac = det_ac.rename(columns={'AC_Weight_pct': 'AC_Weight(%)'})
                # Mostrar explícitamente el peso dentro de la categoría
                dim_cols_present = [c for c in ['Geography','Market'] if c in det_ac.columns]
                show_cols = ['Asset'] + dim_cols_present + ['PortfolioKey','AC_Weight(%)','Pct_in_AC(%)','Pct_in_Portfolio(%)','YTD_w','MTD_w','D1_w']
                show_cols = [c for c in show_cols if c in det_ac.columns]
                tbl = det_ac[show_cols].copy()
                # Ordenar por valor del activo (suma) descendente
                order_assets = (
                    det_ac.groupby('Asset', as_index=False)
                    .agg(_val=('Asset_Value','sum'))
                    .sort_values('_val', ascending=False)['Asset']
                    .tolist()
                )
                tbl['Asset'] = pd.Categorical(tbl['Asset'], categories=order_assets, ordered=True)
                tbl = tbl.sort_values(['Asset','PortfolioKey']).reset_index(drop=True)

                st.markdown("**Activos con dimensiones**")
                # Formateo de porcentajes con 1 decimal para pesos y retornos
                fmt_cols_tbl = {}
                for c in ['AC_Weight(%)','Pct_in_AC(%)','Pct_in_Portfolio(%)','YTD_w','MTD_w','D1_w']:
                    if c in tbl.columns:
                        if c in ['AC_Weight(%)','Pct_in_AC(%)','Pct_in_Portfolio(%)']:
                            fmt_cols_tbl[c] = _fmt_pct_abs
                        else:
                            fmt_cols_tbl[c] = _fmt_pct_ratio
                st.dataframe(tbl.style.format(fmt_cols_tbl))

                # Pivotes por activo: Peso dentro del AC (%) y Retornos ponderados
                piv_weight = det_ac.pivot_table(index='Asset', columns='PortfolioKey', values='Pct_in_AC(%)', aggfunc='first').fillna(0)
                piv_ytd    = det_ac.pivot_table(index='Asset', columns='PortfolioKey', values='YTD_w', aggfunc='first').fillna(0)
                piv_mtd    = det_ac.pivot_table(index='Asset', columns='PortfolioKey', values='MTD_w', aggfunc='first').fillna(0)
                piv_d1     = det_ac.pivot_table(index='Asset', columns='PortfolioKey', values='D1_w',  aggfunc='first').fillna(0)

                # Orden por importancia (suma de valor del activo)
                piv_weight = piv_weight.reindex(order_assets)
                piv_ytd    = piv_ytd.reindex(order_assets)
                piv_mtd    = piv_mtd.reindex(order_assets)
                piv_d1     = piv_d1.reindex(order_assets)

                # Tabs de retornos: la tabla siempre muestra Peso fijo + Retorno seleccionado
                t_ytd, t_mtd, t_d1 = st.tabs(["YTD", "MTD", "D1"])
                with t_ytd:
                    comb = combine_weight_and_metric_flat(piv_weight, piv_ytd, "YTD").reset_index().rename(columns={piv_weight.index.name or 'index': 'Asset'})
                    fmt = {}
                    for c in comb.select_dtypes(include=['number']).columns:
                        if 'Peso (%)' in c or 'Pct' in c:
                            fmt[c] = _fmt_pct_abs
                        else:
                            fmt[c] = _fmt_pct_ratio
                    st.dataframe(comb.style.format(fmt))
                with t_mtd:
                    comb = combine_weight_and_metric_flat(piv_weight, piv_mtd, "MTD").reset_index().rename(columns={piv_weight.index.name or 'index': 'Asset'})
                    fmt = {}
                    for c in comb.select_dtypes(include=['number']).columns:
                        if 'Peso (%)' in c or 'Pct' in c:
                            fmt[c] = _fmt_pct_abs
                        else:
                            fmt[c] = _fmt_pct_ratio
                    st.dataframe(comb.style.format(fmt))
                with t_d1:
                    comb = combine_weight_and_metric_flat(piv_weight, piv_d1,  "D1").reset_index().rename(columns={piv_weight.index.name or 'index': 'Asset'})
                    fmt = {}
                    for c in comb.select_dtypes(include=['number']).columns:
                        if 'Peso (%)' in c or 'Pct' in c:
                            fmt[c] = _fmt_pct_abs
                        else:
                            fmt[c] = _fmt_pct_ratio
                    st.dataframe(comb.style.format(fmt))

        # Helper para pintar cada dimensión con sub-bloques expandibles por categoría
        def render_dim(dim_name: str, dim_df: pd.DataFrame):
            det = dim_df[(dim_df['Asset_Class'] == ac)] if not dim_df.empty else pd.DataFrame()
            if det.empty:
                st.info(f"Sin datos de {dim_name} para este Asset Class.")
                return

            # Inferimos el nombre del campo de la dimensión desde el propio dataframe
            try:
                group_field = det['GroupField'].iloc[0]
            except Exception:
                group_field = dim_name  # fallback solo para mostrar

            # Normalizar categorías vacías/NaN a '(Sin dato)'
            det = det.copy()
            if 'GroupItem' in det.columns:
                det['GroupItem'] = det['GroupItem'].astype(str).str.strip()
                # Reemplazar 'nan', 'NaN', 'None' o vacío por '(Sin dato)'
                det['GroupItem'] = det['GroupItem'].replace(to_replace=r'^(?:nan|NaN|None)?$', value='(Sin dato)', regex=True)

            # Orden de categorías por importancia (valor agregado)
            order_items = (
                det.groupby('GroupItem', as_index=False)
                .agg(_val=('Group_Value','sum'))
                .sort_values('_val', ascending=False)['GroupItem']
                .tolist()
            )

            # 1) Pivotes por categoría: Peso dentro del AC (%) y Retornos ponderados
            piv_weight = det.pivot_table(index='GroupItem', columns='PortfolioKey', values='Pct_in_Portfolio(%)', aggfunc='first').fillna(0)
            piv_ytd    = det.pivot_table(index='GroupItem', columns='PortfolioKey', values='YTD_w', aggfunc='first').fillna(0)
            piv_mtd    = det.pivot_table(index='GroupItem', columns='PortfolioKey', values='MTD_w', aggfunc='first').fillna(0)
            piv_d1     = det.pivot_table(index='GroupItem', columns='PortfolioKey', values='D1_w',  aggfunc='first').fillna(0)

            piv_weight = piv_weight.reindex(order_items)
            piv_ytd    = piv_ytd.reindex(order_items)
            piv_mtd    = piv_mtd.reindex(order_items)
            piv_d1     = piv_d1.reindex(order_items)

            # Tabs de retornos: la tabla siempre muestra Peso fijo + Retorno seleccionado
            t_ytd, t_mtd, t_d1 = st.tabs(["YTD", "MTD", "D1"])
            with t_ytd:
                comb = combine_weight_and_metric_flat(piv_weight, piv_ytd, "YTD").reset_index().rename(columns={piv_weight.index.name or 'index': group_field})
                fmt = {}
                for c in comb.select_dtypes(include=['number']).columns:
                    if 'Peso (%)' in c or 'Pct' in c:
                        fmt[c] = _fmt_pct_abs
                    else:
                        fmt[c] = _fmt_pct_ratio
                st.dataframe(comb.style.format(fmt))
            with t_mtd:
                comb = combine_weight_and_metric_flat(piv_weight, piv_mtd, "MTD").reset_index().rename(columns={piv_weight.index.name or 'index': group_field})
                fmt = {}
                for c in comb.select_dtypes(include=['number']).columns:
                    if 'Peso (%)' in c or 'Pct' in c:
                        fmt[c] = _fmt_pct_abs
                    else:
                        fmt[c] = _fmt_pct_ratio
                st.dataframe(comb.style.format(fmt))
            with t_d1:
                comb = combine_weight_and_metric_flat(piv_weight, piv_d1,  "D1").reset_index().rename(columns={piv_weight.index.name or 'index': group_field})
                fmt = {}
                for c in comb.select_dtypes(include=['number']).columns:
                    if 'Peso (%)' in c or 'Pct' in c:
                        fmt[c] = _fmt_pct_abs
                    else:
                        fmt[c] = _fmt_pct_ratio
                st.dataframe(comb.style.format(fmt))

            # 2) Detalle de activos por categoría (después de los pivotes)
            try:
                targets_df = pd.DataFrame(selected_pairs, columns=["Cliente", "Portafolio"]) if selected_pairs else pd.DataFrame()
                if not targets_df.empty:
                    df_cmp = df.merge(targets_df, on=["Cliente", "Portafolio"], how="inner")
                else:
                    df_cmp = df.copy()
                df_cmp = df_cmp[df_cmp['Asset_Class'] == ac].copy()
                if group_field in df_cmp.columns:
                    df_cmp[group_field] = df_cmp[group_field].fillna('(Sin dato)').astype(str).str.strip()
                else:
                    df_cmp[group_field] = '(Sin dato)'

                # Agregación por activo dentro de la categoría
                keys = ["Cliente","Portafolio","Asset_Class", group_field, "Asset"]
                def agg_asset(g: pd.DataFrame) -> pd.Series:
                    total_val = g['Value'].sum()
                    if total_val == 0:
                        ytd = mtd = d1 = 0.0
                    else:
                        ytd = float((g['Value'] * g['YTD']).sum() / total_val)
                        mtd = float((g['Value'] * g['MTD']).sum() / total_val)
                        d1  = float((g['Value'] * g['D1']).sum()  / total_val)
                    return pd.Series({
                        'Asset_Value': total_val,
                        'YTD_w': ytd,
                        'MTD_w': mtd,
                        'D1_w': d1,
                    })
                assets_g = df_cmp.groupby(keys, as_index=False).apply(agg_asset).reset_index(drop=True)
                grp_tot = assets_g.groupby(["Cliente","Portafolio","Asset_Class", group_field], as_index=False).agg(Group_Total_Value=("Asset_Value","sum"))
                assets_g = assets_g.merge(grp_tot, on=["Cliente","Portafolio","Asset_Class", group_field], how="left")
                assets_g['Pct_in_Group(%)'] = (assets_g['Asset_Value'] / assets_g['Group_Total_Value']).fillna(0) * 100

                # También calcular peso dentro del Asset Class (AC)
                ac_tot = assets_g.groupby(["Cliente","Portafolio","Asset_Class"], as_index=False).agg(AC_Total_Value=("Asset_Value","sum"))
                assets_g = assets_g.merge(ac_tot, on=["Cliente","Portafolio","Asset_Class"], how="left")
                assets_g['Pct_in_AC(%)'] = (assets_g['Asset_Value'] / assets_g['AC_Total_Value']).fillna(0) * 100

                assets_g["PortfolioKey"] = assets_g.apply(lambda r: f"{r['Cliente']} | {r['Portafolio']}", axis=1)
                assets_g = assets_g.rename(columns={group_field: 'GroupItem'})

                # Detalle por categoría (sin expanders para evitar anidación)
                for gi in order_items:
                    st.markdown(f"**{gi} — detalle de activos por portafolio**")
                    sub = assets_g[assets_g['GroupItem'] == gi].copy()
                    if sub.empty:
                        st.info("Sin activos para esta categoría.")
                    else:
                        # Chequeo de consistencia: suma por portfolio dentro de categoría = 100%
                        chk = (
                            sub[['PortfolioKey','Pct_in_Group(%)']]
                            .groupby('PortfolioKey', as_index=False)['Pct_in_Group(%)']
                            .sum()
                        )
                        # Tolerancia de 0.1% para evitar falsas alarmas por redondeo
                        if (chk['Pct_in_Group(%)'].sub(100.0).abs() > 0.1).any():
                            st.warning("Advertencia: las sumas de Peso(%) por portafolio no dan 100% dentro de la categoría.")

                        # Pivotes base
                        piv_weight_ac = sub.pivot_table(index='Asset', columns='PortfolioKey', values='Pct_in_AC(%)', aggfunc='first').fillna(0)
                        piv_y         = sub.pivot_table(index='Asset', columns='PortfolioKey', values='YTD_w',     aggfunc='first').fillna(0)
                        piv_m         = sub.pivot_table(index='Asset', columns='PortfolioKey', values='MTD_w',     aggfunc='first').fillna(0)
                        piv_d         = sub.pivot_table(index='Asset', columns='PortfolioKey', values='D1_w',      aggfunc='first').fillna(0)

                        # Orden por importancia usando el peso dentro del AC
                        order_items_assets = (piv_weight_ac.sum(axis=1).sort_values(ascending=False)).index.tolist()
                        piv_weight_ac = piv_weight_ac.reindex(order_items_assets)
                        piv_y         = piv_y.reindex(order_items_assets)
                        piv_m         = piv_m.reindex(order_items_assets)
                        piv_d         = piv_d.reindex(order_items_assets)

                        # Tabs de retornos: la tabla siempre muestra Peso fijo + Retorno seleccionado
                        t_ytd, t_mtd, t_d1 = st.tabs(["YTD", "MTD", "D1"])
                        with t_ytd:
                            comb = combine_weight_and_metric_flat(piv_weight_ac, piv_y, "YTD").reset_index().rename(columns={piv_weight_ac.index.name or 'index': 'Asset'})
                            fmt = {}
                            for c in comb.select_dtypes(include=['number']).columns:
                                if 'Peso (%)' in c or 'Pct' in c:
                                    fmt[c] = _fmt_pct_abs
                                else:
                                    fmt[c] = _fmt_pct_ratio
                            st.dataframe(comb.style.format(fmt))
                        with t_mtd:
                            comb = combine_weight_and_metric_flat(piv_weight_ac, piv_m, "MTD").reset_index().rename(columns={piv_weight_ac.index.name or 'index': 'Asset'})
                            fmt = {}
                            for c in comb.select_dtypes(include=['number']).columns:
                                if 'Peso (%)' in c or 'Pct' in c:
                                    fmt[c] = _fmt_pct_abs
                                else:
                                    fmt[c] = _fmt_pct_ratio
                            st.dataframe(comb.style.format(fmt))
                        with t_d1:
                            comb = combine_weight_and_metric_flat(piv_weight_ac, piv_d, "D1").reset_index().rename(columns={piv_weight_ac.index.name or 'index': 'Asset'})
                            fmt = {}
                            for c in comb.select_dtypes(include=['number']).columns:
                                if 'Peso (%)' in c or 'Pct' in c:
                                    fmt[c] = _fmt_pct_abs
                                else:
                                    fmt[c] = _fmt_pct_ratio
                            st.dataframe(comb.style.format(fmt))
            except Exception as e:
                st.warning(f"No fue posible construir el detalle de activos para {dim_name}: {e}")

        # --- Geography ---
        with top_tabs[1]:
            render_dim('Geography', geo_break)
        # --- Market ---
        with top_tabs[2]:
            render_dim('Market', mar_break)

st.caption("Basado en la lógica de json_transformer.py.")