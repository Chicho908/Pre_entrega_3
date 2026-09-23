# hashlib genera el hash SHA-256 que usamos como id determinístico de cada chunk.
import hashlib

# logging deja registro de cada operación sobre la base (upsert, query, errores).
import logging

# lru_cache hace que el modelo de embeddings se cargue una sola vez por proceso.
from functools import lru_cache

# Tipos para documentar qué recibe y qué devuelve cada método.
from typing import Any, Dict, List, Optional

# chromadb es la base vectorial local; chromadb.errors trae sus excepciones propias.
import chromadb
import chromadb.errors

# SentenceTransformer carga el modelo de embeddings local (no necesita API key).
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Modelo de embeddings ÚNICO para todo el sistema: se usa tanto para indexar
# (ingesta) como para consultar (retriever). Tenerlo definido en un solo lugar
# evita el error #1 de un RAG: indexar con un modelo y consultar con otro.
#
# Por qué multilingual-e5-small y no all-MiniLM-L6-v2:
#   - all-MiniLM-L6-v2 está entrenado en inglés y trunca el texto a 256 tokens,
#     pero nuestros chunks son de ~450 tokens en español: la mitad de cada chunk
#     no llegaba ni a vectorizarse.
#   - multilingual-e5-small entiende español y acepta hasta 512 tokens, así que
#     el chunk entero queda representado en su embedding.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Los modelos E5 fueron entrenados con estos prefijos: "passage: " para los
# documentos indexados y "query: " para la pregunta. Omitirlos baja la calidad
# de la búsqueda.
PREFIJO_DOCUMENTO = "passage: "
PREFIJO_CONSULTA = "query: "


# Carga el modelo una sola vez por proceso aunque haya varios managers (la
# primera vez lo descarga de HuggingFace, ~470 MB; después queda cacheado en disco).
@lru_cache(maxsize=1)
def _cargar_modelo() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


# Encapsula el CRUD sobre una colección de ChromaDB persistente en disco:
# alta/actualización idempotente (upsert), búsqueda semántica con filtrado
# por metadata, y eliminación de vectores por id.
# Los embeddings se calculan acá mismo con el MISMO modelo para documentos y
# consultas, y el nombre del modelo queda guardado en la metadata de la
# colección para detectar si alguien intenta consultar con otro modelo.
class VectorMemoryManager:
    def __init__(self, persist_path: str, collection_name: str):
        # PersistentClient guarda la colección en disco (persist_path); sin
        # esto, todo viviría en memoria y se perdería al cerrar el proceso.
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection_name = collection_name
        self._modelo = _cargar_modelo()

        self.collection = self._abrir_coleccion()
        logger.info(
            "VectorMemoryManager listo (colección='%s', persistencia='%s', modelo='%s', %d items)",
            collection_name, persist_path, EMBEDDING_MODEL, self.collection.count(),
        )

    # Abre (o crea) la colección y verifica que haya sido indexada con el mismo modelo.
    def _abrir_coleccion(self):
        # hnsw:space="cosine" -> la distancia que devuelve Chroma es
        # 1 - similitud_coseno (0 = idéntico, más alto = menos relacionado).
        # embedding_function=None porque los embeddings los generamos nosotros.
        coleccion = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
        )

        # Guardia contra embeddings no coincidentes: si la colección ya existía
        # y fue indexada con otro modelo, las distancias no tendrían sentido.
        modelo_guardado = (coleccion.metadata or {}).get("embedding_model")
        if modelo_guardado != EMBEDDING_MODEL:
            raise RuntimeError(
                f"La colección '{self.collection_name}' fue indexada con el modelo "
                f"'{modelo_guardado}', pero el sistema consulta con '{EMBEDDING_MODEL}'. "
                f"Reindexá con: python ingesta.py --forzar"
            )
        return coleccion

    # Borra la colección completa y la vuelve a crear vacía (para reindexar desde cero).
    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass  # no existía, nada que borrar
        self.collection = self._abrir_coleccion()
        logger.info("Colección '%s' reiniciada", self.collection_name)

    # Convierte una lista de textos en vectores con el prefijo que corresponda.
    def _embed(self, textos: List[str], prefijo: str) -> List[List[float]]:
        # normalize_embeddings=True deja cada vector con norma 1, que es lo que
        # espera la distancia coseno.
        vectores = self._modelo.encode(
            [prefijo + t for t in textos],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectores.tolist()

    # Inserta o actualiza documentos de forma idempotente: si un id ya existe
    # en la colección, upsert lo actualiza en vez de duplicarlo o lanzar un
    # error de "ID duplicado" (a diferencia de collection.add, que sí falla).
    def upsert_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError(
                f"ids ({len(ids)}), documents ({len(documents)}) y metadatas "
                f"({len(metadatas)}) deben tener la misma longitud"
            )
        if not ids:
            logger.warning("upsert_documents llamado con listas vacías, no se hace nada")
            return

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=self._embed(documents, PREFIJO_DOCUMENTO),
            )
            logger.info("%d documentos insertados/actualizados (upsert)", len(ids))
        except chromadb.errors.ChromaError as e:
            logger.error("Error de ChromaDB al hacer upsert: %s", e)
            raise

    # Convierte la pregunta en embedding (mismo modelo que en la ingesta) y
    # devuelve los chunks más cercanos, ordenados de menor a mayor distancia.
    # `where` permite filtrar por metadata antes de rankear por similitud.
    def semantic_search(
        self,
        query_text: str,
        n_results: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            resultados = self.collection.query(
                query_embeddings=self._embed([query_text], PREFIJO_CONSULTA),
                n_results=n_results,
                where=where,
            )
        except chromadb.errors.ChromaError as e:
            logger.error("Error de ChromaDB al hacer query: %s", e)
            raise

        if not resultados["ids"][0]:
            return []

        return [
            {"id": id_, "documento": doc, "metadata": meta, "distancia": dist}
            for id_, doc, meta, dist in zip(
                resultados["ids"][0],
                resultados["documents"][0],
                resultados["metadatas"][0],
                resultados["distances"][0],
            )
        ]

    # Elimina documentos de la colección por id (operación 'delete' del CRUD).
    def delete_documents(self, ids: List[str]) -> None:
        if not ids:
            logger.warning("delete_documents llamado con una lista vacía, no se hace nada")
            return
        try:
            self.collection.delete(ids=ids)
            logger.info("%d documentos eliminados", len(ids))
        except chromadb.errors.ChromaError as e:
            logger.error("Error de ChromaDB al eliminar: %s", e)
            raise

    # Búsqueda exacta por filtro de metadata, sin ranking por similitud (operación 'read').
    def get_by_metadata(self, where: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            resultados = self.collection.get(where=where)
        except chromadb.errors.ChromaError as e:
            logger.error("Error de ChromaDB al hacer get: %s", e)
            raise
        return [
            {"id": id_, "documento": doc, "metadata": meta}
            for id_, doc, meta in zip(resultados["ids"], resultados["documents"], resultados["metadatas"])
        ]

    # Genera un id determinista a partir del contenido (hash SHA-256): volver a
    # indexar el mismo texto siempre produce el mismo id, así upsert_documents
    # actualiza el registro existente en vez de duplicarlo.
    @staticmethod
    def generar_id_deterministico(texto: str, prefijo: str = "") -> str:
        contenido_hash = hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]
        return f"{prefijo}{contenido_hash}" if prefijo else contenido_hash

    # Cantidad de items actualmente en la colección.
    def count(self) -> int:
        return self.collection.count()
