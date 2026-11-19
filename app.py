"""¿Es real esta oferta? · interfaz para probar el detector final. Ejecutar con `streamlit run app.py`.

Diseño de la plantilla del autor: página clara, tarjeta del formulario en negro con barra de avance, títulos en
monoespaciada y resultados en tarjetas blancas. El tema base está en `.streamlit/config.toml`; los detalles, en ESTILOS.
El modelo es el de `modelos/detector_lexico.joblib` (ver `scripts/detector.py`).
"""
import re
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scripts.detector import MODELO, explicar, limpiar, texto_de_campos

st.set_page_config(page_title="¿Es real esta oferta?", page_icon=":material/policy:", layout="centered")
ss = st.session_state
# Aporte mínimo para mostrar una palabra: debajo de esto su efecto en el resultado es despreciable.
MIN_SUBE, MIN_BAJA = 0.15, 0.05
EJEMPLOS = Path(__file__).parent / "ejemplos"

ESTILOS = """
<style>
:root {
  --tinta: #0b0b0c; --tinta-2: #5b5b61; --tinta-3: #8a8a90; --borde: #e4e4e1; --blanco: #ffffff;
  --negro-1: #141416; --negro-2: #1c1c1f; --negro-borde: #2b2b30; --negro-tinta: #ececf0; --negro-tinta-2: #a1a1aa;
  --azul: #1c5cab; --rojo: #b42318; --rojo-fondo: #fdecea; --ambar: #a15c07; --ambar-fondo: #fdf3e2;
  --verde: #18794e; --verde-fondo: #e7f5ee;
}
[data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 4rem; max-width: 760px; }

.pc-marca { font-family: "JetBrains Mono", monospace; font-weight: 600; font-size: 2.05rem; color: var(--tinta);
            margin: 0; line-height: 1.15; }
.pc-sub { color: var(--tinta-2); margin: .4rem 0 1.6rem; font-size: 1rem; }
.pc-seccion { font-family: "JetBrains Mono", monospace; font-weight: 600; font-size: 1.2rem; color: var(--tinta);
              margin: 2rem 0 .4rem; }

/* Streamlit fija tamaño y márgenes de los párrafos del markdown; estas clases los recuperan. */
[data-testid="stMarkdownContainer"] p.pc-marca { font-size: 2.05rem !important; margin: 0 !important; }
[data-testid="stMarkdownContainer"] p.pc-sub { font-size: 1rem !important; margin: .45rem 0 1.4rem !important; }
[data-testid="stMarkdownContainer"] p.pc-titulo { font-size: 1.35rem !important; margin: 0 !important; }
[data-testid="stMarkdownContainer"] p.pc-desc { font-size: .92rem !important; margin: .15rem 0 .7rem !important; }
[data-testid="stMarkdownContainer"] p.pc-veredicto { font-size: 1.45rem !important; margin: .8rem 0 .3rem !important; }
[data-testid="stMarkdownContainer"] p.pc-explica { margin: 0 0 1.1rem !important; }
[data-testid="stMarkdownContainer"] p.pc-posicion { font-size: .92rem !important; margin: .9rem 0 0 !important; }
[data-testid="stMarkdownContainer"] p.pc-sub2 { font-size: .72rem !important; margin: 1.6rem 0 .6rem !important; }
[data-testid="stMarkdownContainer"] p.pc-pie { font-size: .8rem !important; margin: 1.3rem 0 0 !important; }
[data-testid="stMarkdownContainer"] p.pc-seccion { font-size: 1.2rem !important; }

/* Tarjeta oscura del formulario */
.st-key-tarjeta { background: var(--negro-1); border: 1px solid var(--negro-borde); border-radius: 18px;
                  padding: 0 1.75rem 1.5rem; box-shadow: 0 28px 60px -24px rgba(0,0,0,.45); overflow: hidden; }
.pc-avance { height: 4px; background: var(--negro-borde); margin: 0 -1.75rem 1.4rem; }
.pc-avance > div { height: 100%; background: var(--blanco); border-radius: 0 4px 4px 0; transition: width .3s; }
.pc-paso { font-family: "JetBrains Mono", monospace; font-size: .72rem; color: var(--negro-tinta-2);
           letter-spacing: .08em; text-transform: uppercase; margin-bottom: .3rem; }
.pc-titulo { color: var(--negro-tinta); font-weight: 600; font-size: 1.35rem; margin: 0; }
.pc-desc { color: var(--negro-tinta-2); font-size: .92rem; margin: .15rem 0 .7rem; }
.st-key-tarjeta label p { color: #d4d4d8 !important; font-size: .82rem; font-weight: 500; }
.st-key-tarjeta [data-testid="stTextInputRootElement"], .st-key-tarjeta [data-testid="stTextAreaRootElement"] {
  background: var(--negro-2) !important; border-color: var(--negro-borde) !important; }
.st-key-tarjeta [data-testid="stTextInputRootElement"]:focus-within,
.st-key-tarjeta [data-testid="stTextAreaRootElement"]:focus-within { border-color: #6b6b73 !important; }
.st-key-tarjeta input, .st-key-tarjeta textarea { color: var(--negro-tinta) !important;
  -webkit-text-fill-color: var(--negro-tinta); background: transparent !important; }
.st-key-tarjeta input::placeholder, .st-key-tarjeta textarea::placeholder { color: #6b6b73 !important;
  -webkit-text-fill-color: #6b6b73; }
.st-key-tarjeta [data-testid="InputInstructions"] { color: #6b6b73; }
.st-key-tarjeta button { border-radius: 10px; font-weight: 600; }
.st-key-tarjeta button[kind="primary"] { background: var(--blanco); color: var(--tinta); border: none; }
.st-key-tarjeta button[kind="primary"]:hover { background: #e9e9ec; color: var(--tinta); }
.st-key-tarjeta button[kind="secondary"] { background: transparent; color: var(--negro-tinta-2);
  border: 1px solid var(--negro-borde); }
.st-key-tarjeta button[kind="secondary"]:hover { color: var(--negro-tinta); border-color: #6b6b73; }
.st-key-tarjeta button[kind="tertiary"] { color: var(--negro-tinta-2); }
.st-key-tarjeta button[kind="tertiary"]:hover { color: var(--negro-tinta); }

/* Resultado */
.pc-resultado { background: var(--blanco); border: 1px solid var(--borde); border-radius: 16px; padding: 1.5rem 1.6rem;
                margin-top: 1.6rem; box-shadow: 0 18px 40px -28px rgba(0,0,0,.25); }
.pc-etiqueta { display: inline-block; font-family: "JetBrains Mono", monospace; font-size: .74rem; font-weight: 600;
               letter-spacing: .08em; text-transform: uppercase; padding: .3rem .65rem; border-radius: 999px; }
.pc-veredicto { font-size: 1.5rem; font-weight: 650; color: var(--tinta); margin: .7rem 0 .25rem; line-height: 1.25; }
.pc-explica { color: var(--tinta-2); margin: 0 0 1.2rem; }
.pc-medidor { position: relative; height: 10px; border-radius: 999px; margin: 1.6rem 0 .5rem;
  background: linear-gradient(90deg, #dfe7f1 0%, #dfe7f1 70%, #f6dfc2 85%, #f3c7c2 100%); }
.pc-marca-medidor { position: absolute; top: -7px; width: 4px; height: 24px; background: var(--tinta); border-radius: 2px;
                    transform: translateX(-2px); }
.pc-escala { display: flex; justify-content: space-between; font-size: .76rem; color: var(--tinta-3); }
.pc-posicion { font-size: .92rem; color: var(--tinta); margin: .8rem 0 0; }
.pc-sub2 { font-family: "JetBrains Mono", monospace; font-size: .74rem; color: var(--azul); letter-spacing: .08em;
           text-transform: uppercase; margin: 1.5rem 0 .55rem; }
.pc-chips { display: flex; flex-wrap: wrap; gap: .4rem; }
.pc-chip { font-size: .85rem; padding: .25rem .6rem; border-radius: 8px; }
.pc-chip.malo { background: var(--rojo-fondo); color: var(--rojo); }
.pc-chip.bueno { background: var(--verde-fondo); color: var(--verde); }
.pc-vacio { color: var(--tinta-3); font-size: .88rem; }
.pc-texto { background: #fafaf8; border: 1px solid var(--borde); border-radius: 10px; padding: .9rem 1rem;
            font-size: .88rem; line-height: 1.55; color: var(--tinta-2); max-height: 280px; overflow-y: auto;
            white-space: pre-wrap; }
.pc-texto mark { background: var(--rojo-fondo); color: var(--rojo); padding: 0 .15rem; border-radius: 3px; }
.pc-aviso { background: var(--ambar-fondo); color: var(--ambar); border-radius: 10px; padding: .7rem .9rem;
            font-size: .88rem; margin-top: 1rem; }
.pc-pie { color: var(--tinta-3); font-size: .8rem; margin-top: 1.3rem; }
</style>
"""

NIVELES = {
    "alta": ("Alerta alta", "var(--rojo)", "var(--rojo-fondo)",
             "Este anuncio se parece mucho a las ofertas falsas que estudiamos.",
             "En nuestras pruebas, {p} de cada 100 anuncios con un resultado así eran falsos."),
    "media": ("Alerta", "var(--ambar)", "var(--ambar-fondo)",
              "Este anuncio tiene varias señales de las ofertas falsas. Conviene revisarlo con cuidado.",
              "En nuestras pruebas, {p} de cada 100 anuncios en esta zona o más arriba eran falsos."),
    "baja": ("Sin alerta", "var(--verde)", "var(--verde-fondo)",
             "No encontramos las señales típicas de las ofertas falsas.",
             "Aun así, revisa que la empresa exista antes de enviar tus datos."),
}


@st.cache_resource
def detector():
    return joblib.load(MODELO)


def clasificar(det, puntaje):
    alta, media = det["umbrales"][0.5], det["umbrales"][0.8]
    if puntaje >= alta["umbral"]:
        return "alta", alta["precision"]
    if puntaje >= media["umbral"]:
        return "media", media["precision"]
    return "baja", None


def resaltar(texto, terminos):
    """Marca en el texto del anuncio las palabras que más subieron la sospecha."""
    seguro = escape(texto)
    for t in sorted(terminos, key=len, reverse=True):
        patron = r"\b" + r"\W+".join(re.escape(escape(p)) for p in t.split()) + r"\b"
        seguro = re.sub(patron, lambda m: f"<mark>{m.group(0)}</mark>", seguro, flags=re.I)
    return seguro


def cargar_ejemplo(nombre):
    lineas = (EJEMPLOS / nombre).read_text().strip().split("\n", 1)
    ss.puesto, ss.anuncio = lineas[0], lineas[1] if len(lineas) > 1 else ""
    ss.analizado = True


def limpiar_campos():
    ss.puesto, ss.anuncio, ss.analizado = "", "", False


st.markdown(ESTILOS, unsafe_allow_html=True)
st.markdown('<p class="pc-marca">¿Es real esta oferta?</p>'
            '<p class="pc-sub">Pega un anuncio de empleo y te decimos si se parece a las ofertas falsas que estudiamos.</p>',
            unsafe_allow_html=True)
det = detector()
for clave, valor in (("puesto", ""), ("anuncio", ""), ("analizado", False)):
    ss.setdefault(clave, valor)

uno, varios = st.tabs(["Un anuncio", "Varios anuncios"])

with uno:
    with st.container(key="tarjeta"):
        llenos = bool(ss.puesto.strip()) + bool(ss.anuncio.strip())
        st.markdown(f'<div class="pc-avance"><div style="width:{[8, 50, 100][llenos]}%"></div></div>'
                    '<div class="pc-paso">Detector de ofertas falsas</div>'
                    '<p class="pc-titulo">El anuncio</p>'
                    '<p class="pc-desc">Cópialo completo, en inglés: mientras más texto, más confiable el resultado.</p>',
                    unsafe_allow_html=True)
        st.text_input("Puesto", key="puesto", placeholder="Ej. Data Entry Clerk")
        st.text_area("Texto del anuncio", key="anuncio", height=220,
                     placeholder="Empresa, descripción del puesto, requisitos y prestaciones…")
        a, b, c, d = st.columns([1.15, 1.25, 1.1, .8])
        if a.button("Analizar", type="primary", use_container_width=True, icon=":material/search:"):
            ss.analizado = True
        b.button("Ejemplo sospechoso", on_click=cargar_ejemplo, args=("anuncio_sospechoso.txt",),
                 use_container_width=True)
        c.button("Ejemplo real", on_click=cargar_ejemplo, args=("anuncio_legitimo.txt",), use_container_width=True)
        d.button("Borrar", type="tertiary", on_click=limpiar_campos, use_container_width=True)

    texto = f"{ss.puesto}\n{ss.anuncio}".strip()
    if ss.analizado and not texto:
        st.info("Escribe o pega un anuncio para analizarlo.")
    elif ss.analizado:
        puntaje, n_terminos, hacia_fraude, hacia_legitima = explicar(det, limpiar(texto), n=10)
        hacia_fraude = [(t, v) for t, v in hacia_fraude if v >= MIN_SUBE]
        hacia_legitima = [(t, v) for t, v in hacia_legitima if -v >= MIN_BAJA]
        clave, precision = clasificar(det, puntaje)
        nombre, color, fondo, veredicto, detalle = NIVELES[clave]
        if precision is not None:
            detalle = detalle.format(p=round(precision * 100))
        corpus = det["puntajes_corpus"]
        lugar = np.searchsorted(corpus, puntaje) / len(corpus)
        marca = min(max(lugar * 100, 1), 99)
        lugar_txt = f"{lugar:.1%}" if lugar >= 0.99 else f"{lugar:.0%}"

        malos = "".join(f'<span class="pc-chip malo">{escape(t)}</span>' for t, _ in hacia_fraude) or \
            '<span class="pc-vacio">Ninguna palabra del anuncio sube mucho la sospecha.</span>'
        buenos = "".join(f'<span class="pc-chip bueno">{escape(t)}</span>' for t, _ in hacia_legitima) or \
            '<span class="pc-vacio">Ninguna palabra del anuncio la baja de forma clara.</span>'
        donde = (f'<p class="pc-sub2">Dónde aparecen</p><div class="pc-texto">'
                 f'{resaltar(texto, [t for t, _ in hacia_fraude])}</div>') if hacia_fraude else ""
        aviso = ('<div class="pc-aviso">El anuncio es muy corto o no está en inglés, así que el resultado es poco '
                 'confiable. Pega el texto completo en inglés.</div>') if n_terminos < 20 else ""
        st.markdown(f"""
<div class="pc-resultado">
  <span class="pc-etiqueta" style="color:{color};background:{fondo}">{nombre}</span>
  <p class="pc-veredicto">{veredicto}</p>
  <p class="pc-explica">{detalle}</p>
  <div class="pc-medidor"><div class="pc-marca-medidor" style="left:{marca:.1f}%"></div></div>
  <div class="pc-escala"><span>Menos sospechoso</span><span>Más sospechoso</span></div>
  <p class="pc-posicion">Su nivel de sospecha es más alto que el de <b>{lugar_txt}</b> de los 14,265 anuncios que estudiamos.</p>
  {aviso}
  <p class="pc-sub2">Palabras que suben la sospecha</p>
  <div class="pc-chips">{malos}</div>
  <p class="pc-sub2">Palabras que la bajan</p>
  <div class="pc-chips">{buenos}</div>
  {donde}
  <p class="pc-pie">El detector sirve para decidir qué revisar primero; no prueba que un anuncio sea falso. Un empleo
  legítimo de captura de datos o de asistente puede salir con alerta.</p>
</div>""", unsafe_allow_html=True)

    with st.expander("¿Cómo decide?"):
        st.markdown(
            "Aprendió de **17,880 anuncios reales** publicados entre 2012 y 2014, revisados a mano: 680 eran falsos. "
            "Le da un peso a cada palabra y a cada par de palabras según qué tanto aparece en las ofertas falsas "
            "frente a las reales, y suma los pesos de las que encuentra en tu anuncio.\n\n"
            "Las ofertas falsas suelen ofrecer trabajo de oficina sin requisitos (*data entry*, *assistant*, *clerk*), "
            "prometer dinero (*earn*) y no decir nada de la empresa. Con la alerta, el detector encontró 8 de cada 10 "
            "ofertas falsas, y 3 de cada 4 alertas fueron correctas. Solo lee anuncios en inglés.")

with varios:
    st.markdown('<p class="pc-seccion">Analiza un archivo</p>', unsafe_allow_html=True)
    st.caption("Sube un CSV con las columnas title, company_profile, description, requirements y benefits "
               "(las que tengas). Hay uno de ejemplo en `ejemplos/anuncios.csv`.")
    archivo = st.file_uploader("Archivo CSV", type="csv", label_visibility="collapsed")
    if archivo is not None:
        tabla = pd.read_csv(archivo)
        textos = tabla.apply(texto_de_campos, axis=1)
        puntajes = det["modelo"].decision_function(det["vectorizador"].transform(textos))
        tabla.insert(0, "Resultado", [NIVELES[clasificar(det, p)[0]][0] for p in puntajes])
        tabla.insert(1, "Más sospechoso que", [np.searchsorted(det["puntajes_corpus"], p) / len(det["puntajes_corpus"])
                                               for p in puntajes])
        tabla = tabla.sort_values("Más sospechoso que", ascending=False)
        st.dataframe(tabla, hide_index=True, use_container_width=True,
                     column_config={"Más sospechoso que": st.column_config.ProgressColumn(
                         format="percent", min_value=0, max_value=1)})
        st.download_button("Descargar resultados", tabla.to_csv(index=False).encode(), "resultados_detector.csv",
                           "text/csv", icon=":material/download:")
