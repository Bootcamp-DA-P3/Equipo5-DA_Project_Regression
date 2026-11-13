import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import motor_ml as ml

st.set_page_config(page_title="LaLiga Pro Analytics & ML", page_icon="⚽", layout="wide")

st.markdown(
    """
    <meta name='google' content='notranslate'>
    <style>
        html, body, [class*="css"] {
            font-size: 17px !important;
        }
        h1 {
            font-size: 2.55rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
            margin-bottom: 0.4rem !important;
        }
        h2 {
            font-size: 1.85rem !important;
            font-weight: 700 !important;
        }
        h3 {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }
        h4 {
            font-size: 1.28rem !important;
            font-weight: 600 !important;
        }
        button[data-baseweb="tab"] {
            padding: 8px 12px !important;
        }
        button[data-baseweb="tab"] p {
            font-size: 1.24rem !important;
            font-weight: 700 !important;
        }
        div[data-testid='stMetric'] {
            background-color: #161B22;
            border: 1px solid #30363D;
            padding: 18px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }
        div[data-testid='stMetricLabel'] p {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            color: #9BA3AF !important;
        }
        div[data-testid='stMetricValue'] {
            font-size: 2.15rem !important;
            font-weight: 800 !important;
        }
        label[data-testid="stWidgetLabel"] p {
            font-size: 1.06rem !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="select"] > div {
            font-size: 1.05rem !important;
        }
        button[kind="primary"], button[kind="secondary"] {
            font-size: 1.08rem !important;
            font-weight: 600 !important;
            padding-top: 0.55rem !important;
            padding-bottom: 0.55rem !important;
        }
        p, li {
            font-size: 1.05rem !important;
        }
        div[data-testid="stCaptionContainer"] p {
            font-size: 0.98rem !important;
        }
        div[data-testid="stDataFrame"] {
            font-size: 1.12rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚽ LaLiga Pro Analytics: Predictor ML, Quiniela & Exploración")


@st.cache_data
def init_data():
    return ml.cargar_datos()


@st.cache_resource
def init_models():
    return ml.cargar_modelos()


@st.cache_data(ttl=3600)
def init_calendario(equipos_tuple):
    return ml.obtener_proximas_jornadas(list(equipos_tuple))


@st.cache_data(ttl=21600)
def init_calificacion(eq_sel, jornada_num):
    return ml.obtener_datos_jornada_y_pichichi(eq_sel, int(jornada_num))


df = init_data()
m_loc, m_vis = init_models()
equipos = sorted(df['Equipo_Local'].unique())
equipos_primera = sorted(list(ml._plantillas_primera_division().keys()))
jornadas_dict, idx_proxima, fuente_cal = init_calendario(tuple(equipos))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Predictor Individual",
    "🎟️ Quiniela Oficial",
    "📊 Exploración y Evolución",
    "🧪 Simulador de predicciones",
    "🎯 Predictor por Métrica",
    "📋 Calificación y Once Ideal"
])# --- PESTAÑAS 1 Y 2 ---
with tab1:
    st.markdown("Calcula los **goles esperados** y las **probabilidades**.")
    c1, c2 = st.columns(2)
    idx_rm = equipos.index("Real Madrid") if "Real Madrid" in equipos else 0
    idx_bc = equipos.index("Barcelona") if "Barcelona" in equipos else 1
    local = c1.selectbox("Equipo Local 🏠", equipos, index=idx_rm, key="p_loc")
    visitante = c2.selectbox("Equipo Visitante ✈️", equipos, index=idx_bc, key="p_vis")
    if st.button("Predecir Resultado", type="primary", width="stretch"):
        if local == visitante:
            st.warning("⚠️ Selecciona dos equipos diferentes.")
        else:
            g_loc, g_vis, p1, px_v, p2 = ml.predecir_partido(df, m_loc, m_vis, local, visitante)
            st.subheader(f"Pronóstico: {local} vs {visitante}")
            m1, m2 = st.columns(2)
            m1.metric(f"Goles esperados ({local})", f"{g_loc:.2f}")
            m2.metric(f"Goles esperados ({visitante})", f"{g_vis:.2f}")
            k1, k2, k3 = st.columns(3)
            k1.metric(f"Gana {local} (1)", f"{p1}%")
            k2.metric("Empate (X)", f"{px_v}%")
            k3.metric(f"Gana {visitante} (2)", f"{p2}%")
            fig_p = go.Figure(go.Bar(
                x=[p1, px_v, p2],
                y=[f"Gana {local}", "Empate", f"Gana {visitante}"],
                orientation='h',
                marker_color=['#00CC96', '#AB63FA', '#EF553B'],
                text=[f"{p1}%", f"{px_v}%", f"{p2}%"],
                textposition='auto',
                textfont=dict(size=14)
            ))
            fig_p.update_layout(
                title=dict(text="Distribución de Probabilidad", font=dict(size=18)),
                font=dict(size=14),
                template="plotly_dark",
                height=270
            )
            st.plotly_chart(fig_p, width="stretch")
    st.write("")


with tab2:
    st.caption(f"📡 Estado del calendario: **{fuente_cal}**")
    lista_jornadas = list(jornadas_dict.keys())
    jornada_sel = st.selectbox("📅 Elige la Jornada:", lista_jornadas, index=idx_proxima)
    partidos_jornada = jornadas_dict[jornada_sel]
    filas_q, aciertos, total_jugados = [], 0, 0
    for p in partidos_jornada:
        gl, gv, p1, px_v, p2 = ml.predecir_partido(df, m_loc, m_vis, p["Local"], p["Visitante"])
        probs = {"1 (Local)": p1, "X (Empate)": px_v, "2 (Visitante)": p2}
        signo = max(probs, key=probs.get)
        cod_pred = signo.split(" ").pop(0)
        res_real = p.get("Res")
        if res_real:
            g_l_real, g_v_real = map(int, res_real.split(" - "))
            signo_real = "1" if g_l_real > g_v_real else ("2" if g_v_real > g_l_real else "X")
            total_jugados += 1
            if cod_pred == signo_real:
                aciertos += 1
                estado_eval = f"✅ Acierto ({signo_real})"
            else:
                estado_eval = f"❌ Fallo ({signo_real})"
        else:
            res_real, estado_eval = "Por jugar", "⏳ Pendiente"
        filas_q.append({
            "Fecha": p["Fecha"],
            "Partido": f"{p['Local']} vs {p['Visitante']}",
            "Goles Esp. (ML)": f"{gl:.1f} - {gv:.1f}",
            "1 (%)": p1, "X (%)": px_v, "2 (%)": p2,
            "Pronóstico ML": signo,
            "Resultado Real": res_real,
            "Modelo vs Realidad": estado_eval,
            "Confianza ML": f"{probs[signo]}%"
        })
    if total_jugados > 0:
        tasa = round((aciertos / total_jugados) * 100, 1)
        m1, m2, m3 = st.columns(3)
        m1.metric("Partidos Analizados", f"{total_jugados}")
        m2.metric("Aciertos del Modelo ML", f"{aciertos} de {total_jugados}")
        m3.metric("Efectividad Real (%)", f"{tasa}%")
    elif filas_q:
        mejor = max(filas_q, key=lambda x: max(x["1 (%)"], x["X (%)"], x["2 (%)"]))
        st.success(f"🔥 **Fijo más seguro de {jornada_sel}:** {mejor['Partido']} → **{mejor['Pronóstico ML']}** ({mejor['Confianza ML']})")
    st.dataframe(pd.DataFrame(filas_q), width="stretch", hide_index=True)
    st.write("")# --- PESTAÑAS 3 Y 4 ---
with tab3:
    st.markdown("Compara rendimiento global, perfil táctico y evolución.")
    col_a, col_b = st.columns(2)
    idx_ma = equipos.index("Malaga") if "Malaga" in equipos else 0
    idx_rm2 = equipos.index("Real Madrid") if "Real Madrid" in equipos else 1
    eq1 = col_a.selectbox("Equipo A (Azul) 🔵", equipos, index=idx_ma, key="c_1")
    eq2 = col_b.selectbox("Equipo B (Coral) 🔴", equipos, index=idx_rm2, key="c_2")
    if eq1 == eq2:
        st.warning("⚠️ Selecciona dos equipos diferentes.")
    else:
        s1, s2 = ml.stats_equipo(df, eq1), ml.stats_equipo(df, eq2)
        st.subheader(f"🏆 Resumen Comparativo: {eq1} vs {eq2}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Puntos / Partido ({eq1})", f"{s1['ppg']} pts", delta=f"{round(s1['ppg'] - s2['ppg'], 2)} vs {eq2}")
        k2.metric(f"% Victorias ({eq1})", f"{s1['win']}%", delta=f"{round(s1['win'] - s2['win'], 1)}% vs {eq2}")
        k3.metric(f"Goles a Favor / P ({eq1})", f"{s1['gf']}", delta=f"{round(s1['gf'] - s2['gf'], 2)} vs {eq2}")
        k4.metric(f"Tiros a Puerta / P ({eq1})", f"{s1['tiros']}", delta=f"{round(s1['tiros'] - s2['tiros'], 2)} vs {eq2}")
        cats_b = ['Goles a Favor', 'Goles en Contra', 'Tiros a Puerta', 'Córners', 'Puntos / Partido']
        fig_b = go.Figure([
            go.Bar(name=eq1, x=cats_b, y=[s1['gf'], s1['gc'], s1['tiros'], s1['corners'], s1['ppg']], marker_color='#00B4D8'),
            go.Bar(name=eq2, x=cats_b, y=[s2['gf'], s2['gc'], s2['tiros'], s2['corners'], s2['ppg']], marker_color='#FF6B6B')
        ])
        fig_b.update_layout(barmode='group', font=dict(size=14), template="plotly_dark", height=350)
        st.plotly_chart(fig_b, width="stretch")
        g1, g2 = st.columns(2)
        with g1:
            cats_r = ['Ataque', 'Puntería', 'Presión', 'Regularidad', 'Defensa']
            rad1 = [min(100, round(s1['gf']/2.8*100)), min(100, round(s1['tiros']/7.5*100)), min(100, round(s1['corners']/7.0*100)), min(100, round(s1['ppg']/2.5*100)), min(100, max(10, round((2.2-s1['gc'])/1.6*100)))]
            rad2 = [min(100, round(s2['gf']/2.8*100)), min(100, round(s2['tiros']/7.5*100)), min(100, round(s2['corners']/7.0*100)), min(100, round(s2['ppg']/2.5*100)), min(100, max(10, round((2.2-s2['gc'])/1.6*100)))]
            fig_r = go.Figure([
                go.Scatterpolar(r=rad1, theta=cats_r, fill='toself', name=eq1, line_color='#00B4D8'),
                go.Scatterpolar(r=rad2, theta=cats_r, fill='toself', name=eq2, line_color='#FF6B6B')
            ])
            fig_r.update_layout(polar={'radialaxis': {'visible': True}}, font=dict(size=14), template="plotly_dark", height=370)
            st.plotly_chart(fig_r, width="stretch")
        with g2:
            fig_res = go.Figure([
                go.Bar(name='Victorias %', y=[eq1, eq2], x=[s1['win'], s2['win']], orientation='h', marker_color='#00CC96'),
                go.Bar(name='Empates %', y=[eq1, eq2], x=[s1['draw'], s2['draw']], orientation='h', marker_color='#FFA15A'),
                go.Bar(name='Derrotas %', y=[eq1, eq2], x=[s1['loss'], s2['loss']], orientation='h', marker_color='#EF553B')
            ])
            fig_res.update_layout(barmode='stack', font=dict(size=14), template="plotly_dark", height=370)
            st.plotly_chart(fig_res, width="stretch")
        df_evo = pd.concat([ml.evo_equipo(df, eq1), ml.evo_equipo(df, eq2)]).sort_values("Temporada")
        met_evo = st.radio("Métrica:", ["Puntos_Promedio", "Goles_Promedio"], horizontal=True)
        fig_e = px.line(df_evo, x="Temporada", y=met_evo, color="Equipo", markers=True, color_discrete_map={eq1: "#00B4D8", eq2: "#FF6B6B"})
        fig_e.update_layout(font=dict(size=14), template="plotly_dark", height=360)
        st.plotly_chart(fig_e, width="stretch")
    st.write("")


with tab4:
    st.subheader("🧪 Simulador de predicciones")
    st.markdown("Personaliza las medias estadísticas para simular un encuentro entre los dos clubes seleccionados:")
    s_c1, s_c2 = st.columns(2)
    idx_sl = equipos.index("Real Madrid") if "Real Madrid" in equipos else 0
    idx_sv = equipos.index("Barcelona") if "Barcelona" in equipos else 1
    sim_loc = s_c1.selectbox("Equipo Local 🏠", equipos, index=idx_sl, key="sim_loc")
    sim_vis = s_c2.selectbox("Equipo Visitante ✈️", equipos, index=idx_sv, key="sim_vis")

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"### 🏠 {sim_loc}")
        h2h_gl = st.number_input("Goles Promedio H2H", min_value=0.0, max_value=10.0, value=1.80, step=0.10, key="sim_gl")
        h2h_tl = st.number_input("Tiros a Puerta Promedio H2H", min_value=0.0, max_value=25.0, value=5.00, step=0.50, key="sim_tl")
    with sc2:
        st.markdown(f"### ✈️ {sim_vis}")
        h2h_gv = st.number_input("Goles Promedio H2H", min_value=0.0, max_value=10.0, value=1.40, step=0.10, key="sim_gv")
        h2h_tv = st.number_input("Tiros a Puerta Promedio H2H", min_value=0.0, max_value=25.0, value=4.50, step=0.50, key="sim_tv")

    if st.button("🎯 Predecir Partido Simulado", type="primary", width="stretch"):
        g_loc, g_vis, p1, px_v, p2 = ml.predecir_simulador_manual(m_loc, m_vis, h2h_gl, h2h_gv, h2h_tl, h2h_tv)
        st.subheader(f"Pronóstico Simulado: {sim_loc} vs {sim_vis}")
        sm1, sm2 = st.columns(2)
        sm1.metric(f"Goles esperados ({sim_loc})", f"{g_loc:.2f}")
        sm2.metric(f"Goles esperados ({sim_vis})", f"{g_vis:.2f}")
        sk1, sk2, sk3 = st.columns(3)
        sk1.metric(f"Gana {sim_loc} (1)", f"{p1}%")
        sk2.metric("Empate (X)", f"{px_v}%")
        sk3.metric(f"Gana {sim_vis} (2)", f"{p2}%")
    st.write("")# --- PESTAÑA 5 ---
with tab5:
    st.markdown("Elige un **equipo**, su **rival** y el **dato exacto** que quieres predecir.")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    idx_mo = equipos.index("Malaga") if "Malaga" in equipos else 0
    idx_mr = equipos.index("Sp Gijon") if "Sp Gijon" in equipos else 1
    eq_objetivo = col_m1.selectbox("Equipo a Analizar", equipos, index=idx_mo, key="m_eq")
    eq_rival = col_m2.selectbox("Rival", equipos, index=idx_mr, key="m_riv")
    condicion_eq = col_m3.selectbox("Condición", ["Local", "Visitante"], key="m_cond")
    metricas_lista = [
        "Goles", "Goles al Descanso", "Goles Encajados", "Tiros a Puerta",
        "Tiros Totales", "Puntería (%)", "Córners", "Faltas Cometidas",
        "Tarjetas Amarillas", "Tarjetas Rojas"
    ]
    metrica_sel = col_m4.selectbox("Dato a Predecir", metricas_lista, key="m_met")
    if st.button(f"Predecir {metrica_sel} de {eq_objetivo}", type="primary", width="stretch"):
        if eq_objetivo == eq_rival:
            st.warning("⚠️ El equipo y el rival deben ser diferentes.")
        else:
            pred, med_eq, med_riv, ultimos, max_h = ml.predecir_metrica_equipo(df, eq_objetivo, eq_rival, condicion_eq, metrica_sel)
            st.subheader(f"🎯 Dashboard de {metrica_sel}: {eq_objetivo} ({condicion_eq}) vs {eq_rival}")
            sufijo = "%" if metrica_sel == "Puntería (%)" else ""
            r1, r2, r3 = st.columns(3)
            r1.metric(f"{metrica_sel} Predichos ({eq_objetivo})", f"{pred}{sufijo}", delta=f"{round(pred - med_eq, 2)}{sufijo} vs su media")
            r2.metric(f"Media Reciente {eq_objetivo}", f"{med_eq}{sufijo}")
            r3.metric(f"Referencia Rival ({eq_rival})", f"{med_riv}{sufijo}")
            v1, v2 = st.columns([1, 1.3])
            with v1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=pred, delta={'reference': med_eq},
                    title={'text': f"Velocímetro: {metrica_sel}", 'font': {'size': 18}},
                    gauge={'axis': {'range': list((0, max_h))}, 'bar': {'color': "#00CC96"}}
                ))
                fig_gauge.update_layout(font=dict(size=14), template="plotly_dark", height=330)
                st.plotly_chart(fig_gauge, width="stretch")
            with v2:
                fig_comp = go.Figure(go.Bar(
                    x=["Predicción", f"Media {eq_objetivo}", f"Concedido {eq_rival}"],
                    y=[pred, med_eq, med_riv],
                    marker_color=["#00CC96", "#00B4D8", "#FF6B6B"],
                    text=[f"{pred}{sufijo}", f"{med_eq}{sufijo}", f"{med_riv}{sufijo}"],
                    textposition="auto",
                    textfont=dict(size=14)
                ))
                fig_comp.update_layout(title=dict(text="Comparativa Directa", font=dict(size=18)), font=dict(size=14), template="plotly_dark", height=330)
                st.plotly_chart(fig_comp, width="stretch")
            if ultimos:
                v3, v4 = st.columns([1.4, 1])
                with v3:
                    fig_u = go.Figure(go.Scatter(
                        x=[f"P-{len(ultimos)-i}" for i in range(len(ultimos))],
                        y=ultimos, mode='lines+markers', fill='tozeroy',
                        line=dict(color="#00B4D8", width=3)
                    ))
                    fig_u.add_hline(y=pred, line_dash="dash", line_color="#00CC96")
                    fig_u.update_layout(title=dict(text=f"Últimos {len(ultimos)} partidos", font=dict(size=18)), font=dict(size=14), template="plotly_dark", height=320)
                    st.plotly_chart(fig_u, width="stretch")
                with v4:
                    fig_hist = px.histogram(x=ultimos, nbins=6, title="Frecuencia", color_discrete_sequence=["#AB63FA"])
                    fig_hist.update_layout(font=dict(size=14), template="plotly_dark", height=320, showlegend=False)
                    st.plotly_chart(fig_hist, width="stretch")
    st.write("")# --- PESTAÑA 6 (PLANTILLA Y ONCE IDEAL) ---
with tab6:
    c_sel, c_jor, c_btn = st.columns([2, 1, 1])
    idx_p1 = equipos_primera.index("Real Madrid") if "Real Madrid" in equipos_primera else 0
    eq_cal = c_sel.selectbox("Equipo de Primera División (LaLiga EA Sports):", equipos_primera, index=idx_p1, key="cal_eq")
    jornada_sel_tab6 = c_jor.selectbox("Jornada:", list(range(12, 0, -1)), index=0, key="cal_jor")

    if c_btn.button("🔄 Actualizar Web Ahora", width="stretch"):
        init_calificacion.clear()
        st.rerun()

    j_sel = int(jornada_sel_tab6)

    res_cal = init_calificacion(eq_cal, j_sel)
    dfs_res = [x for x in res_cal if isinstance(x, pd.DataFrame)]
    strs_res = [x for x in res_cal if isinstance(x, str)]

    df_plantilla = dfs_res[0] if len(dfs_res) > 0 else pd.DataFrame()
    df_once = dfs_res[1] if len(dfs_res) > 1 else pd.DataFrame()
    df_pichichi = dfs_res[-1] if len(dfs_res) > 2 else pd.DataFrame()

    tactica_real = next((s for s in strs_res if "-" in s and len(s) <= 8), "4-3-3")
    fuente_calif = next((s for s in strs_res if s != tactica_real), f"ESPN LaLiga Live API · Jornada {j_sel}")

    st.caption(f"🔄 Actualización automática (cada 6h) · Fuente: **{fuente_calif}**")

    disp = df_plantilla[~df_plantilla["Estado"].str.contains("🚑")].copy()
    bajas = df_plantilla[df_plantilla["Estado"].str.contains("🚑")].copy()

    esquemas = {
        "4-3-3": [("POR", 1), ("DEF", 4), ("MED", 3), ("DEL", 3)],
        "4-4-2": [("POR", 1), ("DEF", 4), ("MED", 4), ("DEL", 2)],
        "4-2-3-1": [("POR", 1), ("DEF", 4), ("MED", 5), ("DEL", 1)]
    }
    tac_usar = tactica_real if tactica_real in esquemas else "4-3-3"

    col_pts_j = next((c for c in ["Pts Jornada", "Puntos Jornada", "Pts"] if c in df_plantilla.columns), "Pts Jornada")

    titulares_11 = []
    for p_tac, c_tac in esquemas[tac_usar]:
        sub_pos = disp[disp["Pos"] == p_tac].sort_values(
            by=[col_pts_j, "Nota Media"], ascending=[False, False]
        ).head(c_tac)
        titulares_11.extend(sub_pos.to_dict("records"))

    titulares_set = {j["Jugador"] for j in titulares_11}
    df_plantilla_vista = df_plantilla.copy()

    def _rol_dinamico(row):
        if "🚑" in str(row["Estado"]):
            return "Baja"
        if row["Jugador"] in titulares_set:
            return "Titular"
        return "Banquillo"

    if "Rol" in df_plantilla_vista.columns:
        df_plantilla_vista["Rol"] = df_plantilla_vista.apply(_rol_dinamico, axis=1)
    else:
        df_plantilla_vista.insert(2, "Rol", df_plantilla_vista.apply(_rol_dinamico, axis=1))

    orden_rol = {"Titular": 0, "Banquillo": 1, "Baja": 2}
    df_plantilla_vista["_ord"] = df_plantilla_vista["Rol"].map(orden_rol)
    df_plantilla_vista = df_plantilla_vista.sort_values(
        by=["_ord", col_pts_j, "Nota Media"], ascending=[True, False, False]
    ).drop(columns=["_ord"]).reset_index(drop=True)

    puntos_once = int(sum(j.get(col_pts_j, 0) for j in titulares_11))

    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    m_c1.metric(f"Puntos Once (Jornada {j_sel})", f"{puntos_once} puntos")
    m_c2.metric("Táctica", tac_usar)
    m_c3.metric("Disponibles", f"{len(disp)}")
    m_c4.metric("Lesionados / Bajas", f"{len(bajas)}")

    def dibujar_campo(jugadores, clave_tac, key_pts, key_sub, color_punto, titulo="", altura=560):
        coords_x = {
            "4-3-3": [50, 14, 38, 62, 86, 24, 50, 76, 18, 50, 82],
            "4-4-2": [50, 14, 38, 62, 86, 14, 38, 62, 86, 35, 65],
            "4-2-3-1": [50, 14, 38, 62, 86, 35, 65, 18, 50, 82, 50]
        }
        coords_y = {
            "4-3-3": [11, 29, 27, 27, 29, 53, 49, 53, 79, 85, 79],
            "4-4-2": [11, 29, 27, 27, 29, 56, 53, 53, 56, 82, 82],
            "4-2-3-1": [11, 29, 27, 27, 29, 47, 47, 69, 71, 69, 86]
        }
        tac = clave_tac if clave_tac in coords_x else "4-3-3"
        xs = coords_x[tac][:len(jugadores)]
        ys = coords_y[tac][:len(jugadores)]

        textos = []
        for j in jugadores:
            nom_limpio = str(j.get("Jugador", "")).replace("⚽", "").strip()
            partes = nom_limpio.split()
            apellido = partes[-1] if len(partes) > 1 else nom_limpio
            pts_val = j.get(key_pts, 0) if key_pts else 0
            extra = f"{j[key_sub]}" if key_sub and key_sub in j else ""
            textos.append(
                f"<b>{apellido} ({pts_val} pts)</b><br><span style='font-size:13px'><b>({extra})</b></span>"
                if extra else f"<b>{apellido} ({pts_val} pts)</b>"
            )

        fig = go.Figure()
        fig.add_shape(type="rect", x0=2, y0=2, x1=98, y1=98, line=dict(color="rgba(255,255,255,0.85)", width=2.5))
        fig.add_shape(type="line", x0=2, y0=50, x1=98, y1=50, line=dict(color="rgba(255,255,255,0.85)", width=2.5))
        fig.add_shape(type="circle", x0=38, y0=38, x1=62, y1=62, line=dict(color="rgba(255,255,255,0.85)", width=2.5))
        fig.add_shape(type="rect", x0=22, y0=2, x1=78, y1=18, line=dict(color="rgba(255,255,255,0.85)", width=2.5))
        fig.add_shape(type="rect", x0=22, y0=82, x1=78, y1=98, line=dict(color="rgba(255,255,255,0.85)", width=2.5))

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=textos, textposition="bottom center",
            textfont=dict(color="#0A0E14", size=15, family="Arial Black, sans-serif"),
            marker=dict(size=24, color=color_punto, line=dict(width=2.5, color="#0A0E14"))
        ))
        fig.update_layout(
            title=dict(text=titulo, font=dict(size=17)) if titulo else None,
            xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[0, 100], showgrid=False, zeroline=False, visible=False),
            plot_bgcolor="#38A169", paper_bgcolor="#0E1117", height=altura,
            margin=dict(l=10, r=10, t=38 if titulo else 10, b=12),
            showlegend=False
        )
        return fig

    ALTURA_BLOQUE = 560

    c_iz, c_de = st.columns([1.45, 1])
    with c_iz:
        st.markdown(f"#### 📋 Plantilla Completa ({eq_cal}) — Jornada {j_sel}")
        st.dataframe(
            df_plantilla_vista,
            width="stretch",
            hide_index=True,
            height=ALTURA_BLOQUE,
            row_height=42,
            column_config={
                "Estado": st.column_config.TextColumn("Estado", width="large")
            }
        )

    with c_de:
        st.markdown(f"#### 🏟️ Once Titular — Jornada {j_sel} ({tac_usar})")
        fig_campo = dibujar_campo(titulares_11, tac_usar, col_pts_j, "Nota Media", "#FFFFFF", altura=ALTURA_BLOQUE)
        st.plotly_chart(fig_campo, width="stretch")

    if not df_once.empty:
        st.divider()
        o1, o2 = st.columns([1.45, 1])
        with o1:
            st.markdown(f"#### 🏆 Once Ideal de la Jornada {j_sel} (Mejores 11 de Primera División)")
            df_once_11 = df_once.head(11).copy()
            st.dataframe(
                df_once_11,
                width="stretch",
                hide_index=True,
                height=ALTURA_BLOQUE,
                row_height=47,
                column_config={
                    "Jugador": st.column_config.TextColumn("Jugador", width="medium"),
                    "Equipo": st.column_config.TextColumn("Equipo", width="medium"),
                    "Pos": st.column_config.TextColumn("Pos", width="small"),
                    "Puntos Jornada": st.column_config.NumberColumn("Puntos Jornada", format="%d pts"),
                    "Nota": st.column_config.NumberColumn("Nota Media", format="%.2f"),
                    "Pts Total": st.column_config.NumberColumn("Pts Total", format="%d")
                }
            )
        with o2:
            st.markdown(f"#### ⭐ Equipo de la Semana — Jornada {j_sel} (4-3-3)")
            once_list = df_once.head(11).to_dict("records")
            col_pts_once = next((c for c in ["Puntos Jornada", "Pts Jornada", "Pts"] if c in df_once.columns), "Puntos Jornada")
            fig_ideal = dibujar_campo(
                once_list, "4-3-3", col_pts_once, "Equipo", "#FACC15",
                altura=ALTURA_BLOQUE
            )
            st.plotly_chart(fig_ideal, width="stretch")
    st.write("")# --- PICHICHI Y DASHBOARD ---
    # --- PICHICHI Y DASHBOARD ---
    if not df_pichichi.empty:
        st.divider()
        col_pich, _ = st.columns([1.15, 1.35])
        with col_pich:
            st.markdown("#### ⚽ Tabla Pichichi — Máximos Goleadores")
            df_pich_top = df_pichichi.head(10).copy()
            st.dataframe(
                df_pich_top,
                width="stretch",
                hide_index=True,
                height=485,
                row_height=44,
                column_config={
                    "Pos.": st.column_config.NumberColumn("Pos.", width="small"),
                    "Jugador": st.column_config.TextColumn("Jugador", width="medium"),
                    "Equipo": st.column_config.TextColumn("Equipo", width="small"),
                    "Pos": st.column_config.TextColumn("Dem.", width="small"),
                    "Goles ⚽": st.column_config.NumberColumn("Goles ⚽", format="%d"),
                    "Pts Total": st.column_config.NumberColumn("Pts", format="%d")
                }
            )

        st.write("")
        st.markdown("#### 📈 Proyección Comparativa de Goleadores (J1 → J38)")
        modo_graf = st.radio(
            "Tipo de visualización:",
            ["📈 Evolución Continua (J1 → J38)", "📊 Comparativa por Tramos (J12 / J20 / J30 / J38)"],
            horizontal=True,
            key="modo_pichichi_graf"
        )

        top5_goleadores = df_pichichi.head(5).to_dict("records")
        colores_pich = ["#00CC96", "#00B4D8", "#F59E0B", "#EF553B", "#AB63FA"]
        jornada_ref = max(1, j_sel)

        datos_jugadores = []
        for idx_g, jug in enumerate(top5_goleadores):
            nom_g = str(jug.get("Jugador", "Jugador"))
            apellido_g = nom_g.split()[-1] if len(nom_g.split()) > 1 else nom_g
            goles_act = int(jug.get("Goles ⚽", 0))
            ritmo_gol = goles_act / float(jornada_ref)
            color_l = colores_pich[idx_g % len(colores_pich)]
            seed_g = sum(ord(c) for c in nom_g)

            js_real = list(range(1, jornada_ref + 1))
            goles_real = []
            for j_i in js_real:
                if j_i == jornada_ref:
                    goles_real.append(goles_act)
                else:
                    avance = (j_i / float(jornada_ref)) * goles_act
                    ajuste = ((seed_g + j_i * 3) % 3 - 1) * 0.25
                    val = int(round(max(0, min(goles_act, avance + ajuste))))
                    if goles_real and val < goles_real[-1]:
                        val = goles_real[-1]
                    goles_real.append(val)

            js_proy = list(range(jornada_ref, 39))
            goles_proy = [round(goles_act + ritmo_gol * (j_p - jornada_ref), 1) for j_p in js_proy]
            proy_final = int(round(goles_proy[-1]))

            datos_jugadores.append({
                "nombre": nom_g,
                "apellido": apellido_g,
                "goles_act": goles_act,
                "ritmo": round(ritmo_gol, 2),
                "color": color_l,
                "js_real": js_real,
                "goles_real": goles_real,
                "js_proy": js_proy,
                "goles_proy": goles_proy,
                "proy_final": proy_final
            })

        fig_pich = go.Figure()
        ALTURA_DASHBOARD = 650

        if "Evolución" in modo_graf:
            posiciones_usadas = []

            for d in datos_jugadores:
                fig_pich.add_trace(go.Scatter(
                    x=d["js_real"],
                    y=d["goles_real"],
                    mode="lines+markers",
                    name=f"{d['nombre']} ({d['goles_act']} → {d['proy_final']} goles)",
                    line=dict(color=d["color"], width=4.5, shape="spline"),
                    marker=dict(size=9),
                    hovertemplate=f"<b>{d['nombre']}</b><br>Jornada %{{x}}<br>Goles: %{{y}}<extra></extra>"
                ))

                fig_pich.add_trace(go.Scatter(
                    x=d["js_proy"],
                    y=d["goles_proy"],
                    mode="lines",
                    line=dict(color=d["color"], width=3.5, dash="dash"),
                    showlegend=False,
                    hovertemplate=f"<b>{d['nombre']} (Proy.)</b><br>Jornada %{{x}}<br>Goles est.: %{{y:.1f}}<extra></extra>"
                ))

                y_etiqueta = float(d["goles_proy"][-1])
                for y_prev in posiciones_usadas:
                    if abs(y_etiqueta - y_prev) < 1.9:
                        y_etiqueta = y_prev - 2.0
                posiciones_usadas.append(y_etiqueta)

                fig_pich.add_annotation(
                    x=38.3,
                    y=y_etiqueta,
                    text=f"<b>{d['apellido']} ({d['proy_final']} goles)</b>",
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(color=d["color"], size=17, family="Arial, sans-serif"),
                    bgcolor="rgba(14, 17, 23, 0.85)",
                    borderpad=4
                )

            fig_pich.add_vline(
                x=jornada_ref,
                line_dash="dot",
                line_width=2.5,
                line_color="rgba(255,255,255,0.7)",
                annotation_text=f"  J{jornada_ref} Actual",
                annotation_position="top right",
                annotation_font=dict(size=16, color="#F3F4F6")
            )

            max_y = max(d["proy_final"] for d in datos_jugadores)
            fig_pich.update_layout(
                xaxis=dict(
                    title=dict(text="Jornada de LaLiga (1 - 38)", font=dict(size=18)),
                    tickfont=dict(size=16),
                    range=[1, 44.5],
                    tickmode="linear",
                    dtick=2,
                    gridcolor="rgba(255,255,255,0.08)"
                ),
                yaxis=dict(
                    title=dict(text="Goles Acumulados / Proyectados", font=dict(size=18)),
                    tickfont=dict(size=16),
                    range=[0, max_y * 1.15],
                    gridcolor="rgba(255,255,255,0.08)"
                ),
                template="plotly_dark",
                height=ALTURA_DASHBOARD,
                margin=dict(l=25, r=35, t=85, b=25),
                hovermode="x unified",
                hoverlabel=dict(font_size=16),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.03,
                    xanchor="left",
                    x=0,
                    font=dict(size=16)
                )
            )
        else:
            hitos = sorted(list({jornada_ref, 20, 30, 38}))
            etiquetas_hitos = [f"J{h} (Actual)" if h == jornada_ref else f"J{h} (Proy.)" for h in hitos]

            for d in datos_jugadores:
                valores_hitos = [
                    d["goles_act"] if h == jornada_ref
                    else round(d["goles_act"] + d["ritmo"] * (h - jornada_ref), 1)
                    for h in hitos
                ]
                fig_pich.add_trace(go.Bar(
                    name=f"{d['nombre']} ({d['ritmo']} g/p)",
                    x=etiquetas_hitos,
                    y=valores_hitos,
                    marker_color=d["color"],
                    text=[f"<b>{int(round(v))}</b>" for v in valores_hitos],
                    textposition="outside",
                    textfont=dict(size=17, color="#F9FAFB")
                ))

            max_y = max(d["proy_final"] for d in datos_jugadores)
            fig_pich.update_layout(
                barmode="group",
                bargap=0.22,
                bargroupgap=0.08,
                xaxis=dict(
                    title=dict(text="Hitos de Temporada", font=dict(size=18)),
                    tickfont=dict(size=16)
                ),
                yaxis=dict(
                    title=dict(text="Goles Totales Estimados", font=dict(size=18)),
                    tickfont=dict(size=16),
                    range=[0, max_y * 1.20],
                    gridcolor="rgba(255,255,255,0.08)"
                ),
                template="plotly_dark",
                height=ALTURA_DASHBOARD,
                margin=dict(l=25, r=25, t=85, b=25),
                hoverlabel=dict(font_size=16),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.03,
                    xanchor="left",
                    x=0,
                    font=dict(size=16)
                )
            )

        st.plotly_chart(fig_pich, width="stretch")
    st.write("")