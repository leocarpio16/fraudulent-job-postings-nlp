"""Prueba el detector final del proyecto (léxico TF-IDF + regresión logística, C = 10) con anuncios nuevos.

Uso:
  python scripts/detector.py entrenar                  # una vez: entrena y guarda modelos/detector_lexico.joblib
  python scripts/detector.py probar --archivo anuncio.txt
  python scripts/detector.py probar --texto "Data entry clerk. Earn $500 a week from home..."
  python scripts/detector.py probar --csv anuncios.csv # columnas: title, company_profile, description, requirements, benefits

`entrenar` necesita corpus_procesado/ (cuadernos 01 a 05). Repite la validación cruzada por campañas del artículo para
fijar los umbrales de alerta y luego entrena con todo el corpus. `probar` solo necesita el archivo .joblib.
"""
import argparse, html, re, sys, unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
MODELO = RAIZ / "modelos" / "detector_lexico.joblib"
CAMPOS = ["title", "company_profile", "description", "requirements", "benefits"]
SEMILLA, C = 42, 10.0   # C elegida por la validación anidada en los cinco pliegues (Tabla 7)

# Misma limpieza que el proyecto (Práctica 1 y bloque A.2 del Anexo).
RE_ETIQUETA = re.compile(r"</?[a-zA-Z][^>]*>")
RE_BLOQUE = re.compile(r"</?(p|div|li|ul|ol|br|h[1-6]|tr|td|table|section)[^>]*>", re.I)
RE_ANONIMO = re.compile(r"#(?:URL|EMAIL|PHONE)_[0-9a-fA-F]+#")
RE_ESPACIOS = re.compile(r"[^\S\n]+")
RE_LINEAS = re.compile(r"\n\s*\n+")
TIPOGRAFICOS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...",
                              " ": " ", "​": ""})

def limpiar(texto):
    t = html.unescape(html.unescape(texto))
    t = RE_BLOQUE.sub("\n", t)
    t = RE_ETIQUETA.sub(" ", t)
    t = RE_ANONIMO.sub(" ", t)
    t = unicodedata.normalize("NFKC", t).translate(TIPOGRAFICOS)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = RE_LINEAS.sub("\n", RE_ESPACIOS.sub(" ", t))
    return "\n".join(linea.strip() for linea in t.split("\n")).strip()


def texto_de_campos(fila):
    return limpiar("\n".join(str(fila[c]).strip() for c in CAMPOS if c in fila and pd.notna(fila[c])))


def ruta_corpus():
    for p in [RAIZ / "notebooks" / "corpus_procesado", Path.home() / "Downloads" / "corpus_procesado"]:
        if p.exists():
            return p
    sys.exit("No encontré corpus_procesado/. Ejecuta antes los cuadernos 01 a 05.")


def entrenar():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    ruta = ruta_corpus()
    docs = pd.read_csv(ruta / "corpus_limpio.csv.gz")
    docs = docs.merge(pd.read_csv(ruta / "embeddings" / "campanas.csv.gz"), on="doc_id", how="left")
    textos, y = docs["texto_limpio"], (docs["categoria"] == "Fraudulenta").to_numpy().astype(int)

    def ajustar(tr):
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True,
                              token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#'-]+\b", max_features=60_000)
        X = vec.fit_transform(textos.iloc[tr])
        lr = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000,
                                random_state=SEMILLA).fit(X, y[tr])
        return vec, lr

    # Puntajes fuera de pliegue con la misma validación por campañas del artículo.
    s = np.zeros(len(y))
    pliegues = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA).split(
        np.zeros(len(y)), y, docs["campana"])
    for i, (tr, te) in enumerate(pliegues, 1):
        vec, lr = ajustar(tr)
        s[te] = lr.decision_function(vec.transform(textos.iloc[te]))
        print(f"  pliegue {i}/5 listo")
    print(f"Validación por campañas: AUC {roc_auc_score(y, s):.3f} | precisión promedio {average_precision_score(y, s):.3f}")

    # Umbral que da la mayor precisión detectando al menos esa fracción del fraude (sección 6.8).
    p, r, u = precision_recall_curve(y, s)
    umbrales = {}
    for objetivo in (0.5, 0.8):
        ok = np.where(r[:-1] >= objetivo)[0]
        k = ok[np.argmax(p[:-1][ok])]
        umbrales[objetivo] = {"umbral": float(u[k]), "precision": float(p[k]), "recall": float(r[k])}
        print(f"  detectar {r[k]:.0%} del fraude: umbral {u[k]:.2f}, precisión {p[k]:.2f}")

    vec, lr = ajustar(np.arange(len(y)))
    MODELO.parent.mkdir(exist_ok=True)
    joblib.dump({"vectorizador": vec, "modelo": lr, "umbrales": umbrales, "puntajes_corpus": np.sort(s)},
                MODELO, compress=3)
    print(f"Guardado en {MODELO.relative_to(RAIZ)} ({MODELO.stat().st_size / 1e6:.1f} MB)")


def explicar(det, texto, n=8):
    vec, lr = det["vectorizador"], det["modelo"]
    x = vec.transform([texto])
    puntaje = float(lr.decision_function(x)[0])
    aporte = x.multiply(lr.coef_[0]).tocsr()
    terminos = vec.get_feature_names_out()[aporte.indices]
    orden = np.argsort(aporte.data)
    hacia_fraude = [(terminos[i], aporte.data[i]) for i in orden[::-1][:n] if aporte.data[i] > 0]
    hacia_legitima = [(terminos[i], aporte.data[i]) for i in orden[:n // 2] if aporte.data[i] < 0]
    return puntaje, x.nnz, hacia_fraude, hacia_legitima


def nivel(det, puntaje):
    alta, media = det["umbrales"][0.5], det["umbrales"][0.8]
    if puntaje >= alta["umbral"]:
        return "ALERTA ALTA", f"en la validación, {alta['precision']:.0%} de los anuncios con este puntaje eran fraude"
    if puntaje >= media["umbral"]:
        return "ALERTA", f"en la validación, {media['precision']:.0%} de los anuncios por encima de este umbral eran fraude"
    return "SIN ALERTA", "el puntaje está por debajo del umbral que detecta el 80 % del fraude"


def probar(args):
    if not MODELO.exists():
        sys.exit("No existe el modelo. Ejecuta primero: python scripts/detector.py entrenar")
    det = joblib.load(MODELO)
    if args.csv:
        tabla = pd.read_csv(args.csv)
        textos = tabla.apply(texto_de_campos, axis=1)
        tabla["puntaje"] = det["modelo"].decision_function(det["vectorizador"].transform(textos))
        tabla["nivel"] = [nivel(det, p)[0] for p in tabla["puntaje"]]
        salida = Path(args.csv).with_name(Path(args.csv).stem + "_puntajes.csv")
        tabla.to_csv(salida, index=False)
        print(tabla[["title", "puntaje", "nivel"]].to_string(index=False) if "title" in tabla else tabla[["puntaje", "nivel"]])
        print(f"\nResultados en {salida}")
        return
    crudo = Path(args.archivo).read_text() if args.archivo else (args.texto or sys.stdin.read())
    puntaje, n_terminos, fraude, legitima = explicar(det, limpiar(crudo))
    etiqueta, detalle = nivel(det, puntaje)
    mas_alto = np.searchsorted(det["puntajes_corpus"], puntaje) / len(det["puntajes_corpus"])
    print(f"\n{etiqueta}  (puntaje {puntaje:+.2f}; {detalle})")
    print(f"Más sospechoso que el {mas_alto:.1%} de los 14,265 anuncios del corpus. Términos reconocidos: {n_terminos}")
    if n_terminos < 20:
        print("Aviso: el modelo reconoce pocos términos; el texto es muy corto o no está en inglés.")
    print("\nLo que empuja hacia 'fraudulenta':")
    for t, v in fraude:
        print(f"  {t:<28} {v:+.2f}")
    print("Lo que empuja hacia 'legítima':")
    for t, v in legitima:
        print(f"  {t:<28} {v:+.2f}")
    print("\nEl detector prioriza anuncios para revisión humana; no prueba que un anuncio sea fraude.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="accion", required=True)
    sub.add_parser("entrenar")
    pr = sub.add_parser("probar")
    g = pr.add_mutually_exclusive_group()
    g.add_argument("--archivo"); g.add_argument("--texto"); g.add_argument("--csv")
    args = ap.parse_args()
    entrenar() if args.accion == "entrenar" else probar(args)
