"""Descarga EMSCAD (DataSet.csv) desde Kaggle y lo copia a notebooks/.

Uso:  python scripts/descargar_datos.py
Requiere `pip install kagglehub` y una cuenta de Kaggle (kagglehub pide iniciar sesión la primera vez).
"""
import hashlib
import shutil
from pathlib import Path

import kagglehub

MD5_ESPERADO = "ad00c105b4d1146f9133a7b47b3ed63c"
DESTINO = Path(__file__).resolve().parent.parent / "notebooks" / "DataSet.csv"

origen = Path(kagglehub.dataset_download("amruthjithrajvr/recruitment-scam")) / "DataSet.csv"
shutil.copyfile(origen, DESTINO)

md5 = hashlib.md5(DESTINO.read_bytes()).hexdigest()
print(f"Copiado a {DESTINO}")
print("MD5 correcto" if md5 == MD5_ESPERADO else f"Aviso: MD5 distinto ({md5}); los números pueden variar")
