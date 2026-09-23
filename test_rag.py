# test_rag.py: evidencia de pruebas del sistema completo. Corre 3 casos EN
# PARALELO (asyncio.gather) y guarda el resultado en evidencia_pruebas.json:
#   1. Pregunta real: la respuesta está en /data -> debe responder y citar fuente.
#   2. Trampa fuera de dominio: la corta el umbral, ni siquiera llama al LLM.
#   3. Trampa dentro del dominio: pasa el umbral (habla de HNSW, que sí aparece
#      en los documentos) pero el dato puntual no está -> el LLM debe decir
#      "No lo sé." en vez de usar lo que sabe por su entrenamiento.
# Requiere ANTHROPIC_API_KEY en .env (casos 1 y 3 llaman al LLM).

# asyncio.gather corre las 3 consultas en paralelo.
import asyncio

# json guarda la evidencia de las pruebas en un archivo.
import json

# time mide cuánto tardan las 3 consultas juntas.
import time

# datetime deja registrada la fecha de la corrida en la evidencia.
from datetime import datetime

# Path arma la ruta del archivo de evidencia al lado de este script.
from pathlib import Path

from ingesta import ingestar_documentos
from rag import get_rag_response
from schemas import NO_LO_SE

CASOS = [
    ("real", "¿Qué es el chunk overlap y para qué sirve?"),
    ("trampa_fuera_de_dominio", "¿Cuál es la mejor receta de milanesas con puré?"),
    ("trampa_dentro_del_dominio", "¿Quién inventó el algoritmo HNSW y en qué año se publicó?"),
]

EVIDENCIA = Path(__file__).parent / "evidencia_pruebas.json"


async def main():
    ingestar_documentos()  # no hace nada si la base ya está poblada

    inicio = time.perf_counter()
    # Las 3 consultas corren concurrentemente: mientras una espera al LLM,
    # las otras avanzan. Es la ventaja concreta de la arquitectura async.
    respuestas = await asyncio.gather(*(get_rag_response(p) for _, p in CASOS))
    duracion = time.perf_counter() - inicio

    resultados = {nombre: r for (nombre, _), r in zip(CASOS, respuestas)}
    for nombre, r in resultados.items():
        print("\n" + "=" * 70)
        print(f"CASO: {nombre}")
        print("=" * 70)
        print(r.model_dump_json(indent=2))

    real = resultados["real"]
    fuera = resultados["trampa_fuera_de_dominio"]
    dentro = resultados["trampa_dentro_del_dominio"]

    # Asserts = evidencia automática: si algo se rompe, el script falla acá.
    assert real.respuesta_encontrada and real.respuesta != NO_LO_SE, "Caso real: debería responder"
    assert real.fuentes, "Caso real: debería citar al menos una fuente"
    assert fuera.respuesta == NO_LO_SE and not fuera.fragmentos_recuperados, "Fuera de dominio: lo debería cortar el umbral"
    assert dentro.fragmentos_recuperados, "Dentro de dominio: debería pasar el umbral"
    assert dentro.respuesta == NO_LO_SE and not dentro.fuentes, "Dentro de dominio: el LLM debería decir 'No lo sé.'"

    EVIDENCIA.write_text(
        json.dumps(
            {
                "fecha": datetime.now().isoformat(timespec="seconds"),
                "duracion_total_segundos": round(duracion, 2),
                "resultado": "OK - los 3 casos se comportaron como se esperaba",
                "casos": {n: r.model_dump() for n, r in resultados.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nOK: los 3 casos pasaron en {duracion:.1f}s (en paralelo). Evidencia guardada en {EVIDENCIA.name}")


if __name__ == "__main__":
    asyncio.run(main())
