# ingesta.py: recorre /data (.txt y .md), limpia y fragmenta cada documento
# con DocumentProcessor, genera un id determinístico por chunk y los persiste
# en ChromaDB (./vectorstore) con VectorMemoryManager (upsert). Si la
# colección ya tiene contenido, salta la ingesta (no vuelve a indexar todo).
#
# Uso:
#   python ingesta.py            -> indexa solo si la base está vacía
#   python ingesta.py --forzar   -> borra la colección y reindexa desde cero

# argparse lee el flag --forzar desde la terminal.
import argparse

# logging deja registro de cada paso de la ingesta (chunks por archivo, total).
import logging

# Path arma las rutas relativas a este archivo, así funciona desde cualquier carpeta.
from pathlib import Path

from document_processor import DocumentProcessor
from vector_memory_manager import VectorMemoryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
# Silencia el ruido de las librerías de terceros (descargas de HuggingFace, etc.)
for _ruidoso in ("httpx", "sentence_transformers", "huggingface_hub", "chromadb"):
    logging.getLogger(_ruidoso).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PERSIST_PATH = str(BASE_DIR / "vectorstore")
COLLECTION_NAME = "pre_entrega_3_rag"
EXTENSIONES = ("*.txt", "*.md")

# 500 tokens con 50 de overlap: el mínimo que pide la consigna, medido en
# TOKENS (no caracteres) gracias al length_function de DocumentProcessor.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def ingestar_documentos(forzar: bool = False) -> VectorMemoryManager:
    manager = VectorMemoryManager(persist_path=PERSIST_PATH, collection_name=COLLECTION_NAME)

    # Persistencia: si la base ya tiene contenido, no se recalculan chunking +
    # embeddings en cada ejecución (ahorra tiempo y cómputo).
    if manager.count() > 0 and not forzar:
        logger.info(
            "La colección ya tiene %d chunks persistidos: se salta la ingesta. "
            "Usá 'python ingesta.py --forzar' para reindexar.",
            manager.count(),
        )
        return manager

    # Se borra todo antes de reindexar para que no queden chunks "huérfanos"
    # de versiones viejas de un documento.
    if forzar:
        manager.reset()

    processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    archivos = sorted(p for patron in EXTENSIONES for p in DATA_DIR.glob(patron))
    if not archivos:
        logger.warning("No se encontraron archivos .txt/.md en %s", DATA_DIR)
        return manager

    ids, documentos, metadatas = [], [], []

    for archivo in archivos:
        raw_text = archivo.read_text(encoding="utf-8")
        chunks = processor.process_document(raw_text)
        tokens = [processor.calculate_tokens(c) for c in chunks]
        logger.info("%s: %d chunks (tokens por chunk: %s)", archivo.name, len(chunks), tokens)

        for i, chunk in enumerate(chunks):
            # Id determinístico (hash del contenido + nombre de archivo): si se
            # reindexa el mismo documento sin cambios, upsert pisa el mismo
            # registro en vez de duplicarlo.
            ids.append(VectorMemoryManager.generar_id_deterministico(chunk, prefijo=f"{archivo.stem}_"))
            documentos.append(chunk)
            # fuente = de qué archivo salió (lo que después se cita en la
            # respuesta); posicion = orden del chunk dentro del archivo.
            metadatas.append({"fuente": archivo.name, "posicion": i})

    manager.upsert_documents(ids=ids, documents=documentos, metadatas=metadatas)
    logger.info(
        "Ingesta completa: %d chunks de %d archivos. La colección tiene %d items.",
        len(ids), len(archivos), manager.count(),
    )
    return manager


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa los documentos de /data en ChromaDB.")
    parser.add_argument("--forzar", action="store_true", help="Borra la colección y reindexa desde cero.")
    args = parser.parse_args()
    ingestar_documentos(forzar=args.forzar)
