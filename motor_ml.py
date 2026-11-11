import os, ssl, json, urllib.request, pandas as pd, numpy as np, joblib
from datetime import datetime, timedelta
from scipy.stats import poisson

def cargar_datos():
    df = pd.read_csv('laliga_espanol_limpio.csv')
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.sort_values('Fecha').reset_index(drop=True)
    df['Temporada'] = df['Fecha'].apply(lambda x: f"{x.year}/{str(x.year+1)[-2:]}" if pd.notnull(x) and x.month >= 8 else (f"{x.year-1}/{str(x.year)[-2:]}" if pd.notnull(x) else "N/A"))
    return df

def cargar_modelos():
    return joblib.load('modelo_local.pkl'), joblib.load('modelo_visitante.pkl')

def normalizar_nombre(nombre):
    mapa = {"FC Barcelona": "Barcelona", "Real Madrid CF": "Real Madrid", "Atlético Madrid": "Ath Madrid", "Atlético de Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao", "Real Betis": "Betis", "Real Sociedad": "Sociedad", "Villarreal CF": "Villarreal", "Valencia CF": "Valencia", "Sevilla FC": "Sevilla", "RC Celta": "Celta", "Celta Vigo": "Celta", "CA Osasuna": "Osasuna", "Getafe CF": "Getafe", "RCD Mallorca": "Mallorca", "Girona FC": "Girona", "Rayo Vallecano": "Vallecano", "Deportivo Alavés": "Alaves", "Alavés": "Alaves", "RCD Espanyol": "Espanyol", "UD Las Palmas": "Las Palmas", "CD Leganés": "Leganes", "Leganés": "Leganes", "Real Valladolid": "Valladolid", "Málaga CF": "Malaga", "Málaga": "Malaga", "Levante UD": "Levante", "Elche CF": "Elche", "Granada CF": "Granada", "Cádiz CF": "Cadiz", "UD Almería": "Almeria", "Real Racing Club": "Santander", "Racing Santander": "Santander", "RC Deportivo": "La Coruna", "Deportivo La Coruña": "La Coruna"}
    return mapa.get(str(nombre).strip(), str(nombre).replace(" CF", "").replace(" FC", "").replace(" UD", "").replace(" CD", "").replace(" RC", "").replace(" RCD", "").replace("Real ", "").strip())

def obtener_proximas_jornadas(equipos_validos):
    hoy = datetime.now()
    hoy_str = hoy.strftime('%Y-%m-%d')
    jornadas = {}

    if os.path.exists('pasado_2026_27.csv'):
        df_p = pd.read_csv('pasado_2026_27.csv').fillna('')
        for _, r in df_p.iterrows():
            j_nom = f"{str(r['Jornada']).strip()} [Jugada]"
            jornadas.setdefault(j_nom, []).append({"Fecha": str(r['Fecha']).strip()[:10], "Local": normalizar_nombre(r['Local']), "Visitante": normalizar_nombre(r['Visitante']), "Res": str(r['Resultado']).strip()})

    idx_prox = len(jornadas)
    conectado_vivo = False

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(urllib.request.getproxies()), urllib.request.HTTPSHandler(context=ctx))
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        url = f"https://espn.com{hoy.strftime('%Y%m%d')}-{(hoy + timedelta(days=30)).strftime('%Y%m%d')}&limit=100"
        with opener.open(url, timeout=4) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        for ev in data.get('events', []):
            comps = ev.get('competitions', [])
            if not comps: continue
            comp = comps[0].get('competitors', [])
            if len(comp) < 2: continue
            hc = next((c for c in comp if c.get('homeAway') == 'home'), {})
            ac = next((c for c in comp if c.get('homeAway') == 'away'), {})
            sem = ev.get('week', {}).get('number')
            ronda = f"Jornada {sem}" if sem else "Próxima Jornada"
            jornadas.setdefault(ronda, []).append({"Fecha": ev.get('date', '')[:10], "Local": normalizar_nombre(hc.get('team', {}).get('displayName', '')), "Visitante": normalizar_nombre(ac.get('team', {}).get('displayName', '')), "Res": None})
        if len(jornadas) > idx_prox:
            conectado_vivo = True
    except Exception:
        pass

    if not conectado_vivo:
        futuro_crudo = "Jornada 13 (20-23 Nov)|2026-11-20|Athletic Club|RCD Espanyol;Jornada 13 (20-23 Nov)|2026-11-21|Deportivo Alavés|RC Deportivo;Jornada 13 (20-23 Nov)|2026-11-21|Rayo Vallecano|Valencia CF;Jornada 13 (20-23 Nov)|2026-11-21|Sevilla FC|Real Betis;Jornada 13 (20-23 Nov)|2026-11-21|Real Madrid|RC Celta;Jornada 13 (20-23 Nov)|2026-11-22|FC Barcelona|Villarreal CF;Jornada 13 (20-23 Nov)|2026-11-22|Real Racing Club|Real Sociedad;Jornada 13 (20-23 Nov)|2026-11-22|CA Osasuna|Málaga CF;Jornada 13 (20-23 Nov)|2026-11-22|Getafe CF|Atlético de Madrid;Jornada 13 (20-23 Nov)|2026-11-23|Levante UD|Elche CF;Jornada 14 (27-30 Nov)|2026-11-27|Levante UD|Real Racing Club;Jornada 14 (27-30 Nov)|2026-11-28|Real Betis|Rayo Vallecano;Jornada 14 (27-30 Nov)|2026-11-28|Málaga CF|Athletic Club;Jornada 14 (27-30 Nov)|2026-11-28|Elche CF|Atlético de Madrid;Jornada 14 (27-30 Nov)|2026-11-28|Real Madrid|Deportivo Alavés;Jornada 14 (27-30 Nov)|2026-11-29|RC Celta|Villarreal CF;Jornada 14 (27-30 Nov)|2026-11-29|RC Deportivo|FC Barcelona;Jornada 14 (27-30 Nov)|2026-11-29|RCD Espanyol|Getafe CF;Jornada 14 (27-30 Nov)|2026-11-29|Real Sociedad|Sevilla FC;Jornada 14 (27-30 Nov)|2026-11-30|Valencia CF|CA Osasuna"
        encontrada = False
        for linea in futuro_crudo.split(";"):
            j_nom, fecha, loc, vis = linea.split("|")
            if j_nom not in jornadas and not encontrada and fecha >= hoy_str:
                idx_prox, encontrada = len(jornadas), True
            jornadas.setdefault(j_nom, []).append({"Fecha": fecha, "Local": normalizar_nombre(loc), "Visitante": normalizar_nombre(vis), "Res": None})

    estado = "Híbrido: Pasado CSV + Futuro Conectado en Vivo" if conectado_vivo else "Híbrido: Pasado CSV + Futuro Auto-avance Semanal"
    return jornadas, min(idx_prox, len(jornadas) - 1), estado

def predecir_partido(df, m_loc, m_vis, local, visitante):
    duelos = df[(df['Equipo_Local'] == local) & (df['Equipo_Visitante'] == visitante)].tail(8)
    if duelos.empty:
        g_loc, g_vis = df[df['Equipo_Local'] == local]['Goles_Local_Final'].tail(10).mean(), df[df['Equipo_Visitante'] == visitante]['Goles_Visitante_Final'].tail(10).mean()
        g_loc, g_vis = (1.3 if pd.isna(g_loc) else float(g_loc)), (1.0 if pd.isna(g_vis) else float(g_vis))
    else:
        datos = pd.DataFrame.from_records([{'H2H_Goles_Local': float(duelos['Goles_Local_Final'].mean()), 'H2H_Goles_Visitante': float(duelos['Goles_Visitante_Final'].mean()), 'H2H_Tiros_Local': float(duelos['Tiros_A_Puerta_Local'].mean()), 'H2H_Tiros_Visitante': float(duelos['Tiros_A_Puerta_Visitante'].mean())}]).fillna(1.2)
        g_loc, g_vis = float(m_loc.predict(datos).item()), float(m_vis.predict(datos).item())
    g_loc, g_vis = max(0.1, g_loc), max(0.1, g_vis)
    matriz = np.outer([poisson.pmf(i, g_loc) for i in range(7)], [poisson.pmf(i, g_vis) for i in range(7)])
    return g_loc, g_vis, round(float(np.sum(np.tril(matriz, -1))) * 100, 1), round(float(np.sum(np.diag(matriz))) * 100, 1), round(float(np.sum(np.triu(matriz, 1))) * 100, 1)

def stats_equipo(df, eq):
    loc, vis = df[df['Equipo_Local'] == eq], df[df['Equipo_Visitante'] == eq]
    tot = len(loc) + len(vis)
    if tot == 0: return {"win": 0.0, "draw": 0.0, "loss": 0.0, "ppg": 0.0, "gf": 0.0, "gc": 0.0, "tiros": 0.0, "corners": 0.0}
    vic = (loc['Goles_Local_Final'] > loc['Goles_Visitante_Final']).sum() + (vis['Goles_Visitante_Final'] > vis['Goles_Local_Final']).sum()
    emp = (loc['Goles_Local_Final'] == loc['Goles_Visitante_Final']).sum() + (vis['Goles_Visitante_Final'] == vis['Goles_Local_Final']).sum()
    return {"win": round((vic / tot) * 100, 1), "draw": round((emp / tot) * 100, 1), "loss": round(((tot - vic - emp) / tot) * 100, 1), "ppg": round(((vic * 3) + emp) / tot, 2), "gf": round((loc['Goles_Local_Final'].sum() + vis['Goles_Visitante_Final'].sum()) / tot, 2), "gc": round((loc['Goles_Visitante_Final'].sum() + vis['Goles_Local_Final'].sum()) / tot, 2), "tiros": round((loc['Tiros_A_Puerta_Local'].sum() + vis['Tiros_A_Puerta_Visitante'].sum()) / tot, 2), "corners": round((loc['Corners_Local'].sum() + vis['Corners_Visitante'].sum()) / tot, 2)}

def evo_equipo(df, eq):
    p = df[(df['Equipo_Local'] == eq) | (df['Equipo_Visitante'] == eq)].copy()
    if p.empty: return pd.DataFrame(columns=['Temporada', 'Goles_Promedio', 'Puntos_Promedio', 'Equipo'])
    p['Goles'] = np.where(p['Equipo_Local'] == eq, p['Goles_Local_Final'], p['Goles_Visitante_Final'])
    p['Rival'] = np.where(p['Equipo_Local'] == eq, p['Goles_Visitante_Final'], p['Goles_Local_Final'])
    p['Puntos'] = np.where(p['Goles'] > p['Rival'], 3, np.where(p['Goles'] == p['Rival'], 1, 0))
    e = p.groupby('Temporada').agg(Goles_Promedio=('Goles', 'mean'), Puntos_Promedio=('Puntos', 'mean')).reset_index()
    e['Equipo'] = eq
    return e
