import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import motor_ml as ml

st.set_page_config(page_title="LaLiga Pro Analytics & ML", page_icon="⚽", layout="wide")

st.markdown("""
    <meta name="google" content="notranslate">
    <style>
    div[data-testid='stMetric'] {background-color: #161B22; border: 1px solid #30363D; padding: 15px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("⚽ LaLiga Pro Analytics: Predictor ML, Quiniela & Scouting")

@st.cache_data
def init_data():
    return ml.cargar_datos()

@st.cache_resource
def init_models():
    return ml.cargar_modelos()

@st.cache_data(ttl=3600)
def init_calendario(equipos_tuple):
    return ml.obtener_proximas_jornadas(list(equipos_tuple))

df = init_data()
m_loc, m_vis = init_models()
equipos = sorted(df['Equipo_Local'].unique())
jornadas_dict, idx_proxima, fuente_cal = init_calendario(tuple(equipos))

tab1, tab2, tab3 = st.tabs([
    "🔮 Predictor Individual",
    "🎟️ Quiniela Oficial (Modelo vs Realidad)",
    "📊 Scouting y Evolución"
])

with tab1:
    st.markdown("Selecciona dos equipos para calcular los **goles esperados** y las **probabilidades de resultado**.")
    c1, c2 = st.columns(2)
    local = c1.selectbox("Equipo Local 🏠", equipos, index=equipos.index("Real Madrid") if "Real Madrid" in equipos else 0, key="p_loc")
    visitante = c2.selectbox("Equipo Visitante ✈️", equipos, index=equipos.index("Barcelona") if "Barcelona" in equipos else 1, key="p_vis")

    if st.button("Predecir Resultado", type="primary", use_container_width=True):
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
                textposition='auto'
            ))
            fig_p.update_layout(title="Distribución de Probabilidad", xaxis_title="%", template="plotly_dark", height=250)
            st.plotly_chart(fig_p, use_container_width=True)

with tab2:
    st.caption(f"📡 Estado del calendario: **{fuente_cal}**")
    lista_jornadas = list(jornadas_dict.keys())
    jornada_sel = st.selectbox("📅 Elige la Jornada (Temporada 2026/27):", lista_jornadas, index=idx_proxima)
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
            res_real = "Por jugar"
            estado_eval = "⏳ Pendiente"

        filas_q.append({
            "Fecha": p["Fecha"],
            "Partido": f"{p['Local']} vs {p['Visitante']}",
            "Goles Esp. (ML)": f"{gl:.1f} - {gv:.1f}",
            "1 (%)": p1,
            "X (%)": px_v,
            "2 (%)": p2,
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

    df_q = pd.DataFrame(filas_q)
    st.dataframe(df_q, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("Selecciona **dos equipos** para comparar su rendimiento global, perfil táctico y evolución.")
    col_a, col_b = st.columns(2)
    eq1 = col_a.selectbox("Equipo A (Azul) 🔵", equipos, index=equipos.index("Malaga") if "Malaga" in equipos else 0, key="c_1")
    eq2 = col_b.selectbox("Equipo B (Coral) 🔴", equipos, index=equipos.index("Real Madrid") if "Real Madrid" in equipos else 1, key="c_2")

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

        st.markdown("#### 📊 Comparativa Directa de Promedios por Partido")
        cats_b = ['Goles a Favor', 'Goles en Contra', 'Tiros a Puerta', 'Córners', 'Puntos / Partido']
        v_b1 = [s1['gf'], s1['gc'], s1['tiros'], s1['corners'], s1['ppg']]
        v_b2 = [s2['gf'], s2['gc'], s2['tiros'], s2['corners'], s2['ppg']]

        fig_b = go.Figure([
            go.Bar(name=eq1, x=cats_b, y=v_b1, marker_color='#00B4D8', text=v_b1, textposition='auto'),
            go.Bar(name=eq2, x=cats_b, y=v_b2, marker_color='#FF6B6B', text=v_b2, textposition='auto')
        ])
        fig_b.update_layout(barmode='group', template="plotly_dark", height=340)
        st.plotly_chart(fig_b, use_container_width=True)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### 🕸️ Perfil Táctico (Escala 0 - 100)")
            cats_r = ['Ataque', 'Puntería', 'Presión', 'Regularidad', 'Defensa']
            rad1 = [
                min(100, round(s1['gf']/2.8*100)),
                min(100, round(s1['tiros']/7.5*100)),
                min(100, round(s1['corners']/7.0*100)),
                min(100, round(s1['ppg']/2.5*100)),
                min(100, max(10, round((2.2-s1['gc'])/1.6*100)))
            ]
            rad2 = [
                min(100, round(s2['gf']/2.8*100)),
                min(100, round(s2['tiros']/7.5*100)),
                min(100, round(s2['corners']/7.0*100)),
                min(100, round(s2['ppg']/2.5*100)),
                min(100, max(10, round((2.2-s2['gc'])/1.6*100)))
            ]
            fig_r = go.Figure([
                go.Scatterpolar(r=rad1, theta=cats_r, fill='toself', name=eq1, line_color='#00B4D8'),
                go.Scatterpolar(r=rad2, theta=cats_r, fill='toself', name=eq2, line_color='#FF6B6B')
            ])
            fig_r.update_layout(polar={'radialaxis': {'visible': True}}, template="plotly_dark", height=360)
            st.plotly_chart(fig_r, use_container_width=True)

        with g2:
            st.markdown("#### ⚖️ Balance de Resultados (%)")
            fig_res = go.Figure([
                go.Bar(name='Victorias %', y=[eq1, eq2], x=[s1['win'], s2['win']], orientation='h', marker_color='#00CC96', text=[f"{s1['win']}%", f"{s2['win']}%"], textposition='inside'),
                go.Bar(name='Empates %', y=[eq1, eq2], x=[s1['draw'], s2['draw']], orientation='h', marker_color='#FFA15A', text=[f"{s1['draw']}%", f"{s2['draw']}%"], textposition='inside'),
                go.Bar(name='Derrotas %', y=[eq1, eq2], x=[s1['loss'], s2['loss']], orientation='h', marker_color='#EF553B', text=[f"{s1['loss']}%", f"{s2['loss']}%"], textposition='inside')
            ])
            fig_res.update_layout(barmode='stack', template="plotly_dark", height=360)
            st.plotly_chart(fig_res, use_container_width=True)

        st.markdown("#### 📈 Evolución y Progresión por Temporada")
        df_evo = pd.concat([ml.evo_equipo(df, eq1), ml.evo_equipo(df, eq2)]).sort_values("Temporada")
        met_evo = st.radio(
            "Métrica:",
            ["Puntos_Promedio", "Goles_Promedio"],
            horizontal=True,
            format_func=lambda x: "Puntos por Partido" if x == "Puntos_Promedio" else "Goles por Partido"
        )
        temporadas_ordenadas = sorted(df_evo["Temporada"].unique())
        fig_e = px.line(
            df_evo,
            x="Temporada",
            y=met_evo,
            color="Equipo",
            markers=True,
            category_orders={"Temporada": temporadas_ordenadas},
            color_discrete_map={eq1: "#00B4D8", eq2: "#FF6B6B"}
        )
        fig_e.update_traces(line={'width': 3}, marker={'size': 9})
        fig_e.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_e, use_container_width=True)
