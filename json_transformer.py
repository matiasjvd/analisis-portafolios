import json

# Abrir y leer un archivo JSON
with open("clientes_sistema.json", "r", encoding="utf-8") as f:
    data = json.load(f)   # Convierte el JSON en un diccionario de Python

import pandas as pd

rows = []

for cliente in data['clients']:
    for p in cliente['portfolios']:
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
        pw = p.get('weight')
        p_weight = (pw*100 if (pw is not None and abs(pw) <= 1) else (pw or 0))

        for ac in p.get('asset_types', []):
            # Valor asset class (si falta, suma holdings)
            ac_val = ac.get('value')
            if ac_val is None:
                ac_val = sum((h.get('value') or 0) for h in ac.get('holdings', []))

            # Peso asset class normalizado a %
            aw = ac.get('weight')
            ac_weight = (aw*100 if (aw is not None and abs(aw) <= 1) else (aw or 0))

            for h in ac.get('holdings', []):
                hw = h.get('weight')
                rows.append({
                    "Cliente":    cliente.get('name'),
                    "Portafolio": p.get('name'),
                    "Asset_Class": ac.get('type_id'),
                    "Asset":      h.get('asset_id'),

                    # Portafolio
                    "Portfolio_Value":     p_val,
                    "Portfolio_Weight(%)": p_weight,
                    "Portfolio_YTD":       p.get('ytd'),
                    "Portfolio_MTD":       p.get('mtd'),
                    "Portfolio_D1":        p.get('d1'),

                    # Asset class
                    "AC_Value":            ac_val,
                    "AC_Weight(%)":        ac_weight,
                    "AC_YTD":              ac.get('ytd'),
                    "AC_MTD":              ac.get('mtd'),
                    "AC_D1":               ac.get('d1'),

                    # Activo
                    "Value":               h.get('value'),
                    "Weight_Asset(%)":     (hw*100 if (hw is not None and abs(hw) <= 1) else (hw or 0)),
                    "YTD":                 h.get('ytd'),
                    "MTD":                 h.get('mtd'),
                    "D1":                  h.get('d1'),
                })

df = pd.DataFrame(rows)

# Numéricos + NaN -> 0
num_cols = [
    "Portfolio_Value","Portfolio_Weight(%)","Portfolio_YTD","Portfolio_MTD","Portfolio_D1",
    "AC_Value","AC_Weight(%)","AC_YTD","AC_MTD","AC_D1",
    "Value","Weight_Asset(%)","YTD","MTD","D1"
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Orden final (opcional)
df = df[[
    "Cliente","Portafolio","Asset_Class","Asset",
    "Portfolio_Value","Portfolio_Weight(%)","Portfolio_YTD","Portfolio_MTD","Portfolio_D1",
    "AC_Value","AC_Weight(%)","AC_YTD","AC_MTD","AC_D1",
    "Value","Weight_Asset(%)","YTD","MTD","D1"
]]
import pandas as pd

# === 1) Selección de portafolios a comparar ===
# Usa tuplas (Cliente, Portafolio). Ejemplos:
targets = [
    ("Inmobiliaria Truro Ltda", "Internacional"),
    ("Modelo2024", "Internacional"),
    # ("Otro Cliente", "Nacional"),
]

# Armo un DataFrame de targets para filtrar rápido
targets_df = pd.DataFrame(targets, columns=["Cliente", "Portafolio"])
df_cmp = df.merge(targets_df, on=["Cliente", "Portafolio"], how="inner")

# === 2) Composición por Asset_Class usando métricas de categoría del JSON ===
# Evitamos sumar holdings: tomamos directamente los campos AC_* provistos en el JSON
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

# Tablas “lado a lado” (pivot) por Asset_Class usando directamente los valores de categoría
pivot_value = comp.pivot(index="Asset_Class", columns=["Cliente","Portafolio"], values="AC_Value").fillna(0)
pivot_pct   = comp.pivot(index="Asset_Class", columns=["Cliente","Portafolio"], values="AC_Weight_pct").fillna(0)

# === 3) Métricas a nivel Portafolio (peso y retornos) ===
# Tomamos la primera fila por portafolio porque esas columnas son constantes por portafolio en cada fila
metrics = (
    df_cmp.sort_values(["Cliente","Portafolio"])
          .groupby(["Cliente","Portafolio"], as_index=False)
          .agg(
              Portfolio_Value=("Portfolio_Value","first"),
              Portfolio_Weight_pct=("Portfolio_Weight(%)","first"),
              YTD=("Portfolio_YTD","first"),
              MTD=("Portfolio_MTD","first"),
              D1 =("Portfolio_D1","first"),
          )
)

# === 4) (Opcional) Redondeo para lectura rápida ===
pivot_pct  = pivot_pct.round(2)
metrics[["Portfolio_Weight_pct","YTD","MTD","D1"]] = metrics[["Portfolio_Weight_pct","YTD","MTD","D1"]].round(4)

# === 5) Mostrar resultados ===
print("\n--- Composición por Asset_Class (VALOR) ---")
print(pivot_value)

print("\n--- Composición por Asset_Class (% del portafolio) ---")
print(pivot_pct)

print("\n--- Métricas de Portafolio ---")
print(metrics)
