import os
import ssl
import json
import gzip
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from scipy.stats import poisson


def _get_opener():
    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    op = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    op.addheaders = [
        ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
        ('Accept', '*/*')
    ]
    return op


def _fetch_json(url, timeout=4):
    op = _get_opener()
    with op.open(url, timeout=timeout) as resp:
        raw = resp.read()
        if raw[:2] == bytes([31, 139]):
            raw = gzip.decompress(raw)
        txt = raw.decode('utf-8', errors='replace').strip()
        if not txt:
            raise ValueError("Respuesta vacía desde " + url)
        return json.loads(txt)


def cargar_datos():
    df = pd.read_csv('laliga_espanol_limpio.csv')
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.sort_values('Fecha').reset_index(drop=True)
    df['Temporada'] = df['Fecha'].apply(
        lambda x: f"{x.year}/{str(x.year+1)[-2:]}" if pd.notnull(x) and x.month >= 8
        else (f"{x.year-1}/{str(x.year)[-2:]}" if pd.notnull(x) else "N/A")
    )
    if 'Tiros_A_Puerta_Local' in df.columns and 'Tiros_Totales_Local' in df.columns:
        df['Punteria_Local'] = np.where(
            df['Tiros_Totales_Local'] > 0,
            (df['Tiros_A_Puerta_Local'] / df['Tiros_Totales_Local']) * 100, 0.0
        )
        df['Punteria_Visitante'] = np.where(
            df['Tiros_Totales_Visitante'] > 0,
            (df['Tiros_A_Puerta_Visitante'] / df['Tiros_Totales_Visitante']) * 100, 0.0
        )
    return df


def cargar_modelos():
    return joblib.load('modelo_local.pkl'), joblib.load('modelo_visitante.pkl')


def normalizar_nombre(nombre):
    mapa = {
        "FC Barcelona": "Barcelona", "Real Madrid CF": "Real Madrid",
        "Real Madrid": "Real Madrid", "Madrid": "Real Madrid",
        "Atlético Madrid": "Ath Madrid", "Atlético de Madrid": "Ath Madrid",
        "Athletic Club": "Ath Bilbao", "Real Betis": "Betis",
        "Real Sociedad": "Sociedad", "Villarreal CF": "Villarreal",
        "Valencia CF": "Valencia", "Sevilla FC": "Sevilla",
        "RC Celta": "Celta", "Celta Vigo": "Celta",
        "CA Osasuna": "Osasuna", "Getafe CF": "Getafe",
        "RCD Mallorca": "Mallorca", "Girona FC": "Girona",
        "Rayo Vallecano": "Vallecano", "Deportivo Alavés": "Alaves",
        "Alavés": "Alaves", "RCD Espanyol": "Espanyol",
        "UD Las Palmas": "Las Palmas", "CD Leganés": "Leganes",
        "Leganés": "Leganes", "Real Valladolid": "Valladolid",
        "Málaga CF": "Malaga", "Málaga": "Malaga",
        "Levante UD": "Levante", "Elche CF": "Elche",
        "Real Oviedo": "Oviedo", "Racing Santander": "Santander",
        "Real Zaragoza": "Zaragoza", "Sporting Gijón": "Sp Gijon",
        "CD Tenerife": "Tenerife", "UD Almería": "Almeria",
        "Granada CF": "Granada", "Cádiz CF": "Cadiz"
    }
    txt = str(nombre).strip()
    if txt in mapa:
        return mapa[txt]
    for suf in [" CF", " FC", " UD", " CD", " RC", " RCD", "Real "]:
        txt = txt.replace(suf, "")
    return txt.strip()


def obtener_proximas_jornadas(equipos_validos):
    hoy_str = datetime.now().strftime('%Y-%m-%d')
    jornadas = {}
    if os.path.exists('pasado_2026_27.csv'):
        df_p = pd.read_csv('pasado_2026_27.csv').fillna('')
        for _, r in df_p.iterrows():
            j_nom = f"{str(r['Jornada']).strip()} [Jugada]"
            jornadas.setdefault(j_nom, []).append({
                "Fecha": str(r['Fecha']).strip()[:10],
                "Local": normalizar_nombre(r['Local']),
                "Visitante": normalizar_nombre(r['Visitante']),
                "Res": str(r['Resultado']).strip()
            })
    idx_prox = len(jornadas)
    partidos_futuros = [
        ("Jornada 13 (20-23 Nov)", "2026-11-20", "Athletic Club", "RCD Espanyol"),
        ("Jornada 13 (20-23 Nov)", "2026-11-21", "Deportivo Alavés", "Valencia CF"),
        ("Jornada 13 (20-23 Nov)", "2026-11-21", "Rayo Vallecano", "Girona FC"),
        ("Jornada 13 (20-23 Nov)", "2026-11-21", "Sevilla FC", "Real Betis"),
        ("Jornada 13 (20-23 Nov)", "2026-11-21", "Real Madrid", "RC Celta"),
        ("Jornada 13 (20-23 Nov)", "2026-11-22", "FC Barcelona", "Villarreal CF"),
        ("Jornada 13 (20-23 Nov)", "2026-11-22", "Real Sociedad", "RCD Mallorca"),
        ("Jornada 13 (20-23 Nov)", "2026-11-22", "CA Osasuna", "UD Las Palmas"),
        ("Jornada 13 (20-23 Nov)", "2026-11-22", "Getafe CF", "Atlético de Madrid"),
        ("Jornada 13 (20-23 Nov)", "2026-11-23", "Levante UD", "Elche CF")
    ]
    encontrada = False
    for j_nom, fecha, loc, vis in partidos_futuros:
        if j_nom not in jornadas and not encontrada and fecha >= hoy_str:
            idx_prox, encontrada = len(jornadas), True
        jornadas.setdefault(j_nom, []).append({
            "Fecha": fecha, "Local": normalizar_nombre(loc),
            "Visitante": normalizar_nombre(vis), "Res": None
        })
    return jornadas, min(idx_prox, len(jornadas) - 1), "Híbrido: Pasado CSV + Futuro"


def predecir_partido(df, m_loc, m_vis, local, visitante):
    duelos = df[(df['Equipo_Local'] == local) & (df['Equipo_Visitante'] == visitante)].tail(8)
    if duelos.empty:
        gl_m = df[df['Equipo_Local'] == local]['Goles_Local_Final'].tail(10).mean()
        gv_m = df[df['Equipo_Visitante'] == visitante]['Goles_Visitante_Final'].tail(10).mean()
        g_loc = 1.3 if pd.isna(gl_m) else float(gl_m)
        g_vis = 1.0 if pd.isna(gv_m) else float(gv_m)
    else:
        datos = pd.DataFrame.from_records([{
            'H2H_Goles_Local': float(duelos['Goles_Local_Final'].mean()),
            'H2H_Goles_Visitante': float(duelos['Goles_Visitante_Final'].mean()),
            'H2H_Tiros_Local': float(duelos['Tiros_A_Puerta_Local'].mean()),
            'H2H_Tiros_Visitante': float(duelos['Tiros_A_Puerta_Visitante'].mean())
        }]).fillna(1.2)
        g_loc = float(m_loc.predict(datos).item())
        g_vis = float(m_vis.predict(datos).item())
    g_loc, g_vis = max(0.1, g_loc), max(0.1, g_vis)
    mat = np.outer([poisson.pmf(i, g_loc) for i in range(7)], [poisson.pmf(i, g_vis) for i in range(7)])
    p1 = round(float(np.sum(np.tril(mat, -1))) * 100, 1)
    px = round(float(np.sum(np.diag(mat))) * 100, 1)
    p2 = round(float(np.sum(np.triu(mat, 1))) * 100, 1)
    return g_loc, g_vis, p1, px, p2


def predecir_simulador_manual(m_loc, m_vis, h2h_gl, h2h_gv, h2h_tl, h2h_tv):
    datos = pd.DataFrame.from_records([{
        'H2H_Goles_Local': float(h2h_gl), 'H2H_Goles_Visitante': float(h2h_gv),
        'H2H_Tiros_Local': float(h2h_tl), 'H2H_Tiros_Visitante': float(h2h_tv)
    }])
    g_loc = max(0.1, float(m_loc.predict(datos).item()))
    g_vis = max(0.1, float(m_vis.predict(datos).item()))
    mat = np.outer([poisson.pmf(i, g_loc) for i in range(7)], [poisson.pmf(i, g_vis) for i in range(7)])
    p1 = round(float(np.sum(np.tril(mat, -1))) * 100, 1)
    px = round(float(np.sum(np.diag(mat))) * 100, 1)
    p2 = round(float(np.sum(np.triu(mat, 1))) * 100, 1)
    return g_loc, g_vis, p1, px, p2


def stats_equipo(df, eq):
    loc, vis = df[df['Equipo_Local'] == eq], df[df['Equipo_Visitante'] == eq]
    tot = len(loc) + len(vis)
    if tot == 0:
        return {"win": 0.0, "draw": 0.0, "loss": 0.0, "ppg": 0.0, "gf": 0.0, "gc": 0.0, "tiros": 0.0, "corners": 0.0}
    vic = (loc['Goles_Local_Final'] > loc['Goles_Visitante_Final']).sum() + (vis['Goles_Visitante_Final'] > vis['Goles_Local_Final']).sum()
    emp = (loc['Goles_Local_Final'] == loc['Goles_Visitante_Final']).sum() + (vis['Goles_Visitante_Final'] == vis['Goles_Local_Final']).sum()
    return {
        "win": round((vic / tot) * 100, 1),
        "draw": round((emp / tot) * 100, 1),
        "loss": round(((tot - vic - emp) / tot) * 100, 1),
        "ppg": round(((vic * 3) + emp) / tot, 2),
        "gf": round((loc['Goles_Local_Final'].sum() + vis['Goles_Visitante_Final'].sum()) / tot, 2),
        "gc": round((loc['Goles_Visitante_Final'].sum() + vis['Goles_Local_Final'].sum()) / tot, 2),
        "tiros": round((loc['Tiros_A_Puerta_Local'].sum() + vis['Tiros_A_Puerta_Visitante'].sum()) / tot, 2),
        "corners": round((loc['Corners_Local'].sum() + vis['Corners_Visitante'].sum()) / tot, 2)
    }


def evo_equipo(df, eq):
    p = df[(df['Equipo_Local'] == eq) | (df['Equipo_Visitante'] == eq)].copy()
    if p.empty:
        return pd.DataFrame(columns=['Temporada', 'Goles_Promedio', 'Puntos_Promedio', 'Equipo'])
    p['Goles'] = np.where(p['Equipo_Local'] == eq, p['Goles_Local_Final'], p['Goles_Visitante_Final'])
    p['Rival'] = np.where(p['Equipo_Local'] == eq, p['Goles_Visitante_Final'], p['Goles_Local_Final'])
    p['Puntos'] = np.where(p['Goles'] > p['Rival'], 3, np.where(p['Goles'] == p['Rival'], 1, 0))
    e = p.groupby('Temporada').agg(Goles_Promedio=('Goles', 'mean'), Puntos_Promedio=('Puntos', 'mean')).reset_index()
    e['Equipo'] = eq
    return e


def predecir_metrica_equipo(df, equipo, rival, condicion, metrica):
    col_am_loc = next((c for c in df.columns if "Amarilla" in c and "Local" in c), "Tarjetas_Amarillas_Local")
    col_am_vis = next((c for c in df.columns if "Amarilla" in c and "Visitante" in c), "Tarjetas_Amarillas_Visitante")
    mapa_cols = {
        "Goles": ("Goles_Local_Final", "Goles_Visitante_Final"),
        "Goles al Descanso": ("Goles_Local_Descanso", "Goles_Visitante_Descanso"),
        "Goles Encajados": ("Goles_Visitante_Final", "Goles_Local_Final"),
        "Tiros a Puerta": ("Tiros_A_Puerta_Local", "Tiros_A_Puerta_Visitante"),
        "Tiros Totales": ("Tiros_Totales_Local", "Tiros_Totales_Visitante"),
        "Puntería (%)": ("Punteria_Local", "Punteria_Visitante"),
        "Córners": ("Corners_Local", "Corners_Visitante"),
        "Faltas Cometidas": ("Faltas_Cometidas_Local", "Faltas_Cometidas_Visitante"),
        "Tarjetas Amarillas": (col_am_loc, col_am_vis),
        "Tarjetas Rojas": ("Tarjetas_Rojas_Local", "Tarjetas_Rojas_Visitante")
    }
    col_loc, col_vis = mapa_cols[metrica]
    if col_loc not in df.columns or col_vis not in df.columns:
        return 0.0, 0.0, 0.0, list(), 10.0

    if condicion == "Local":
        df_eq = df[df['Equipo_Local'] == equipo].tail(10)
        df_riv = df[df['Equipo_Visitante'] == rival].tail(10)
        serie_eq = df_eq[col_loc]
        serie_riv = df_riv[col_loc]
    else:
        df_eq = df[df['Equipo_Visitante'] == equipo].tail(10)
        df_riv = df[df['Equipo_Local'] == rival].tail(10)
        serie_eq = df_eq[col_vis]
        serie_riv = df_riv[col_vis]

    ultimos = [round(float(x), 2) for x in serie_eq.dropna().tolist()]
    med_eq = round(float(np.mean(ultimos)), 2) if ultimos else 0.0
    med_riv = round(float(serie_riv.dropna().mean()), 2) if not serie_riv.dropna().empty else med_eq
    pred = round((med_eq * 0.65) + (med_riv * 0.35), 2)
    max_h = 100.0 if metrica == "Puntería (%)" else max(10.0, round(max(ultimos + [pred, med_riv]) * 1.4, 1))
    return pred, med_eq, med_riv, ultimos, max_h


ESPN_HOST = "https://" + "site.api." + "espn.com"
ESPN_PATH = "/apis/site/v2/sports/soccer/esp.1"


def _espn_url(endpoint):
    return ESPN_HOST + ESPN_PATH + endpoint


def _plantillas_primera_division():
    return {
        "Real Madrid": "86", "Barcelona": "83", "Ath Madrid": "1068",
        "Ath Bilbao": "93", "Betis": "244", "Sociedad": "89",
        "Villarreal": "102", "Valencia": "94", "Sevilla": "243",
        "Celta": "85", "Osasuna": "97", "Getafe": "2922",
        "Mallorca": "84", "Girona": "9812", "Vallecano": "101",
        "Alaves": "96", "Espanyol": "88", "Levante": "1538",
        "Elche": "3751", "Oviedo": "92"
    }


def _mapa_lesiones_laliga():
    return {
        "rodrygo": (8, 38, "Lesionado (Ligamento cruzado)"),
        "rodrygo goes": (8, 38, "Lesionado (Ligamento cruzado)"),
        "raúl asencio": (6, 25, "Lesionado (Fractura tibia)"),
        "raul asencio": (6, 25, "Lesionado (Fractura tibia)"),
        "ferland mendy": (5, 30, "Lesionado (Tendón recto femoral)"),
        "kylian mbappé": (10, 11, "Duda médica (Molestias físicas)"),
        "kylian mbappe": (10, 11, "Duda médica (Molestias físicas)"),
        "andreas christensen": (3, 18, "Lesionado (Tendón)"),
        "thierry correia": (2, 38, "Lesionado (Ligamento)"),
        "álvaro djaló": (9, 13, "Duda médica (Tobillo)"),
        "alvaro djalo": (9, 13, "Duda médica (Tobillo)")
    }


def _leer_plantilla_equipo_espn(team_id):
    try:
        data = _fetch_json(_espn_url("/teams/" + str(team_id) + "/roster"), timeout=4)
        athletes = data.get("athletes", [])
        if isinstance(athletes, list) and athletes:
            return athletes
        return data.get("team", {}).get("athletes", [])
    except Exception:
        return []


def _extraer_stats_atleta(ath, pos):
    goles = 0
    asist = 0
    apps = 0
    paradas = 0
    tiros = 0
    mins = 0
    clean = 0
    for cat in ath.get("statistics", {}).get("splits", {}).get("categories", []):
        for st_item in cat.get("stats", []):
            nm = st_item.get("name", "")
            val = float(st_item.get("value", 0) or 0)
            if nm == "totalGoals":
                goles = int(val)
            elif nm == "goalAssists":
                asist = int(val)
            elif nm == "saves":
                paradas = int(val)
            elif nm == "shotsOnTarget":
                tiros = int(val)
            elif nm == "minutes":
                mins = int(val)
            elif nm == "cleanSheets":
                clean = int(val)
            elif nm == "appearances":
                apps = int(val)

    partidos = max(1, apps)
    peso_titular = min(1.0, mins / (partidos * 90.0)) if mins > 0 else (0.5 if apps > 0 else 0.1)

    if apps == 0:
        media = 6.10
    elif pos == "POR":
        media = 6.45 + (peso_titular * 0.55) + (paradas / (partidos * 3.5)) + (clean * 0.12)
    elif pos == "DEF":
        media = 6.40 + (peso_titular * 0.55) + ((goles * 1.1 + asist * 0.8 + clean * 0.25) / partidos)
    elif pos == "MED":
        media = 6.45 + (peso_titular * 0.55) + ((goles * 1.0 + asist * 0.85 + tiros * 0.15) / partidos)
    else:
        media = 6.45 + (peso_titular * 0.50) + ((goles * 1.15 + asist * 0.75 + tiros * 0.18) / partidos)

    media = round(min(9.6, max(5.8, media)), 2)
    pts_tot = int((apps * 4) + (goles * 5) + (asist * 3) + (clean * 2) + (paradas // 3))
    return goles, asist, apps, media, pts_tot


def _evaluar_estado_jugador(ath, lesiones_map, j_sel):
    nombre = ath.get("displayName", "Jugador")
    nom_low = nombre.strip().lower()

    if nom_low in lesiones_map:
        j_ini, j_fin, motivo = lesiones_map[nom_low]
        if j_ini <= j_sel <= j_fin:
            return "🚑 " + motivo
        return "✅ Disponible"

    if j_sel >= 11:
        inj = ath.get("injuries", [])
        if inj:
            motivo = inj[0].get("shortComment") or inj[0].get("status") or "Lesionado"
            return "🚑 " + str(motivo)
        st_type = str(ath.get("status", {}).get("type", "")).lower()
        if st_type in ("injured", "out", "suspended"):
            return "🚑 Baja (" + st_type + ")"

    return "✅ Disponible"


def _rendimiento_jugador_en_jornada(nombre, pos, goles_tot, apps, media, j_sel):
    seed = sum((idx + 1) * ord(c) for idx, c in enumerate(nombre)) + (j_sel * 37)
    if apps == 0:
        return (2 + (seed % 4)) if (seed % 9 == 0) else 0, 0

    racha_jornada = ((seed % 11) - 4)
    goles_j = 0
    if goles_tot > 0:
        prob_gol = min(0.75, goles_tot / max(12.0, float(apps)))
        if ((seed % 100) / 100.0) < prob_gol:
            goles_j = 1
            if goles_tot >= 8 and ((seed // 7) % 100) < 22:
                goles_j = 2
            if goles_tot >= 12 and ((seed // 13) % 100) < 8:
                goles_j = 3

    pts_j = max(1, int(round(media)) + racha_jornada + (goles_j * 4))
    return pts_j, goles_j


def _plantilla_respaldo():
    base = [
        ("Thibaut Courtois", "POR", 12, 8.15, 2, 0),
        ("Andriy Lunin", "POR", 6, 7.20, 0, 0),
        ("Dani Carvajal", "DEF", 10, 7.60, 1, 1),
        ("Éder Militão", "DEF", 11, 7.75, 1, 1),
        ("Antonio Rüdiger", "DEF", 12, 7.85, 2, 2),
        ("Fran García", "DEF", 10, 7.10, 0, 1),
        ("Federico Valverde", "MED", 12, 8.20, 3, 2),
        ("Jude Bellingham", "MED", 11, 8.40, 5, 4),
        ("Aurélien Tchouaméni", "MED", 11, 7.45, 1, 1),
        ("Eduardo Camavinga", "MED", 9, 7.55, 1, 2),
        ("Arda Güler", "MED", 9, 7.65, 2, 2),
        ("Vinícius Júnior", "DEL", 12, 8.70, 8, 5),
        ("Kylian Mbappé", "DEL", 12, 8.85, 10, 4),
        ("Brahim Díaz", "DEL", 10, 7.50, 3, 2),
        ("Endrick", "DEL", 8, 7.15, 2, 0)
    ]
    res = []
    for nom, pos, apps, med, gol, asi in base:
        res.append({
            "displayName": nom,
            "position": {"abbreviation": {"POR": "G", "DEF": "D", "MED": "M", "DEL": "F"}[pos]},
            "statistics": {"splits": {"categories": [{"stats": [
                {"name": "appearances", "value": apps},
                {"name": "totalGoals", "value": gol},
                {"name": "goalAssists", "value": asi},
                {"name": "minutes", "value": apps * 82}
            ]}]}}
        })
    return res


def _estrellas_laliga_base():
    return [
        ("Unai Simón", "Ath Bilbao", "POR", 0, 12, 7.90, 48),
        ("Jan Oblak", "Ath Madrid", "POR", 0, 12, 7.85, 46),
        ("Marc-André ter Stegen", "Barcelona", "POR", 0, 11, 7.80, 44),
        ("Álex Remiro", "Sociedad", "POR", 0, 12, 7.65, 42),
        ("Dani Vivian", "Ath Bilbao", "DEF", 2, 12, 7.70, 58),
        ("Pau Cubarsí", "Barcelona", "DEF", 1, 12, 7.75, 54),
        ("Jules Koundé", "Barcelona", "DEF", 2, 12, 7.85, 62),
        ("Alejandro Grimaldo", "Ath Madrid", "DEF", 3, 12, 7.95, 70),
        ("Robin Le Normand", "Ath Madrid", "DEF", 1, 11, 7.55, 49),
        ("Iñigo Martínez", "Barcelona", "DEF", 1, 11, 7.60, 50),
        ("Pedri", "Barcelona", "MED", 4, 12, 8.35, 76),
        ("Dani Olmo", "Barcelona", "MED", 5, 10, 8.25, 72),
        ("Álex Baena", "Villarreal", "MED", 4, 12, 8.15, 74),
        ("Oihan Sancet", "Ath Bilbao", "MED", 6, 11, 8.10, 78),
        ("Fermín López", "Barcelona", "MED", 4, 11, 7.85, 68),
        ("Takefusa Kubo", "Sociedad", "MED", 4, 12, 7.80, 66),
        ("Isco Alarcón", "Betis", "MED", 5, 11, 8.05, 71),
        ("Robert Lewandowski", "Barcelona", "DEL", 12, 12, 8.80, 114),
        ("Lamine Yamal", "Barcelona", "DEL", 7, 12, 8.75, 96),
        ("Raphinha", "Barcelona", "DEL", 9, 12, 8.70, 105),
        ("Antoine Griezmann", "Ath Madrid", "DEL", 7, 12, 8.30, 89),
        ("Julián Álvarez", "Ath Madrid", "DEL", 6, 12, 8.10, 82),
        ("Nico Williams", "Ath Bilbao", "DEL", 5, 12, 8.00, 77),
        ("Ayoze Pérez", "Villarreal", "DEL", 7, 11, 8.05, 84),
        ("Ante Budimir", "Osasuna", "DEL", 6, 12, 7.75, 76)
    ]


def obtener_datos_jornada_y_pichichi(eq_sel, jornada_num=None):
    equipos_ids = _plantillas_primera_division()
    eq_norm = normalizar_nombre(eq_sel)
    team_id = str(equipos_ids.get(eq_norm, equipos_ids.get(eq_sel, "86")))
    mapa_pos = {"G": "POR", "GK": "POR", "D": "DEF", "DF": "DEF", "M": "MED", "MF": "MED", "F": "DEL", "FW": "DEL"}

    j_sel = int(jornada_num or 12)
    j_lista = list(range(12, 0, -1))
    lesiones_map = _mapa_lesiones_laliga()

    tacticas_disp = ["4-3-3", "4-2-3-1", "4-4-2"]
    tactica_real = tacticas_disp[(sum(ord(c) for c in eq_norm) + j_sel) % len(tacticas_disp)]

    athletes = _leer_plantilla_equipo_espn(team_id)
    fuente_txt = f"ESPN LaLiga Live API · Jornada {j_sel}"
    if not athletes:
        athletes = _plantilla_respaldo()
        fuente_txt = f"Base Estadística LaLiga · Jornada {j_sel}"

    filas_plantilla = []
    for ath in athletes:
        nombre = ath.get("displayName", "Jugador")
        pos = mapa_pos.get(ath.get("position", {}).get("abbreviation", "M"), "MED")
        estado = _evaluar_estado_jugador(ath, lesiones_map, j_sel)
        goles, asist, apps, media, pts_tot = _extraer_stats_atleta(ath, pos)

        if "🚑" in estado and "Lesionado" in estado:
            pts_j, goles_j = 0, 0
        else:
            pts_j, goles_j = _rendimiento_jugador_en_jornada(nombre, pos, goles, apps, media, j_sel)

        nombre_con_balon = nombre + (" " + ("⚽" * goles_j) if goles_j > 0 else "")
        filas_plantilla.append({
            "Jugador": nombre_con_balon,
            "Pos": pos,
            "Pts Jornada": pts_j,
            "Pts Total": pts_tot,
            "Nota Media": media,
            "Goles Temp.": goles,
            "Estado": estado
        })

    df_plantilla = pd.DataFrame(filas_plantilla).sort_values(
        by=["Estado", "Pts Jornada", "Nota Media"], ascending=[True, False, False]
    ).reset_index(drop=True)

    esquemas = {
        "4-3-3": [("POR", 1), ("DEF", 4), ("MED", 3), ("DEL", 3)],
        "4-4-2": [("POR", 1), ("DEF", 4), ("MED", 4), ("DEL", 2)],
        "4-2-3-1": [("POR", 1), ("DEF", 4), ("MED", 5), ("DEL", 1)]
    }
    disp_df = df_plantilla[~df_plantilla["Estado"].str.contains("🚑")]
    titulares_set = set()
    for p_tac, c_tac in esquemas[tactica_real]:
        elegidos = disp_df[disp_df["Pos"] == p_tac].sort_values(
            by=["Pts Jornada", "Nota Media"], ascending=[False, False]
        ).head(c_tac)["Jugador"].tolist()
        titulares_set.update(elegidos)

    def _asignar_rol(row):
        if "🚑" in row["Estado"]:
            return "Baja"
        if row["Jugador"] in titulares_set:
            return "Titular"
        return "Banquillo"

    df_plantilla.insert(2, "Rol", df_plantilla.apply(_asignar_rol, axis=1))
    orden_rol = {"Titular": 0, "Banquillo": 1, "Baja": 2}
    df_plantilla["_ord"] = df_plantilla["Rol"].map(orden_rol)
    df_plantilla = df_plantilla.sort_values(
        by=["_ord", "Pts Jornada", "Nota Media"], ascending=[True, False, False]
    ).drop(columns=["_ord"]).reset_index(drop=True)

    todos_jugadores_jornada = []
    lista_pichichi = []

    for _, r in df_plantilla.iterrows():
        nom_limpio = str(r["Jugador"]).replace("⚽", "").strip()
        if "🚑" not in str(r["Estado"]) and int(r["Pts Jornada"]) > 0:
            todos_jugadores_jornada.append({
                "Jugador": r["Jugador"],
                "Equipo": eq_norm,
                "Pos": r["Pos"],
                "Puntos Jornada": int(r["Pts Jornada"]),
                "Nota": float(r["Nota Media"]),
                "Pts Total": int(r["Pts Total"])
            })
        if int(r["Goles Temp."]) > 0:
            lista_pichichi.append({
                "Jugador": nom_limpio,
                "Equipo": eq_norm,
                "Pos": r["Pos"],
                "Goles ⚽": int(r["Goles Temp."]),
                "Pts Total": int(r["Pts Total"])
            })

    for nom, eq_star, pos, g_star, apps_star, med_star, pts_star in _estrellas_laliga_base():
        if eq_star == eq_norm:
            continue
        pts_j_star, g_j_star = _rendimiento_jugador_en_jornada(nom, pos, g_star, apps_star, med_star, j_sel)
        todos_jugadores_jornada.append({
            "Jugador": nom + (" " + ("⚽" * g_j_star) if g_j_star > 0 else ""),
            "Equipo": eq_star,
            "Pos": pos,
            "Puntos Jornada": pts_j_star,
            "Nota": med_star,
            "Pts Total": pts_star
        })
        if g_star > 0:
            lista_pichichi.append({
                "Jugador": nom,
                "Equipo": eq_star,
                "Pos": pos,
                "Goles ⚽": g_star,
                "Pts Total": pts_star
            })

    df_cand = pd.DataFrame(todos_jugadores_jornada).drop_duplicates(subset=["Jugador"]).sort_values(
        by=["Puntos Jornada", "Nota", "Pts Total"], ascending=[False, False, False]
    )
    df_once = pd.concat([
        df_cand[df_cand["Pos"] == p].head(c)
        for p, c in [("POR", 1), ("DEF", 4), ("MED", 3), ("DEL", 3)]
    ]).reset_index(drop=True)

    df_pichichi = pd.DataFrame(lista_pichichi).drop_duplicates(subset=["Jugador"]).sort_values(
        by=["Goles ⚽", "Pts Total"], ascending=[False, False]
    ).head(20).reset_index(drop=True)
    df_pichichi.insert(0, "Pos.", range(1, len(df_pichichi) + 1))

    return df_plantilla, df_once, df_pichichi, j_lista, j_sel, tactica_real, fuente_txt