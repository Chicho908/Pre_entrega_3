# rag.py: get_rag_response(query) es el núcleo asíncrono del sistema.
#   1. Retriever: convierte la pregunta en embedding y busca en ChromaDB
#      (en un hilo aparte, para no bloquear el event loop).
#   2. Filtro de relevancia: descarta chunks por encima de DISTANCIA_MAXIMA.
#   3. Generación grounded: cadena LCEL prompt | LLM | PydanticOutputParser,
#      llamada con await y con reintentos que dejan log de cada fallo.

# asyncio: to_thread para la búsqueda vectorial y sleep para esperar entre reintentos.
import asyncio

# logging deja registro de cada paso: recuperación, cada intento al LLM, y errores.
import logging

# os se usa para verificar que la API key esté cargada antes de crear el modelo.
import os

# lru_cache hace que el manager y la cadena se creen una sola vez por proceso.
from functools import lru_cache

# List tipa la lista de fragmentos que devuelve el retriever.
from typing import List

# Errores del SDK de Anthropic que no tiene sentido reintentar (key inválida, request mal armado).
from anthropic import AuthenticationError, BadRequestError, PermissionDeniedError

# load_dotenv lee el .env y carga la API key como variable de entorno.
from dotenv import load_dotenv

# ChatAnthropic es el mismo proveedor que ya usamos en el Módulo 1 y la Pre-entrega 2.
from langchain_anthropic import ChatAnthropic

# PydanticOutputParser convierte el texto del LLM en un objeto Pydantic validado.
from langchain_core.output_parsers import PydanticOutputParser

# ChatPromptTemplate evita hardcodear f-strings: LangChain resuelve {contexto}/{pregunta} solo.
from langchain_core.prompts import ChatPromptTemplate

from ingesta import COLLECTION_NAME, PERSIST_PATH
from schemas import NO_LO_SE, FragmentoRecuperado, RAGQuery, RAGResponse, RespuestaLLM
from vector_memory_manager import VectorMemoryManager

# La API key se lee de .env (nunca se escribe en el código).
load_dotenv()

logger = logging.getLogger(__name__)

LLM_MODEL = "claude-sonnet-4-5"

# Umbral de distancia coseno (0 = idéntico) por encima del cual un chunk se
# considera irrelevante. Calibrado corriendo 10 preguntas del dominio y 8
# totalmente ajenas contra la base indexada (500/50 tokens, multilingual-e5-small):
#   preguntas del dominio -> distancia mínima entre 0.09 y 0.19
#   preguntas ajenas      -> distancia mínima entre 0.20 y 0.29
# Es la primera barrera (barata, sin llamar al LLM). La segunda barrera es el
# prompt: si una pregunta pasa el umbral pero el dato no está, el LLM dice
# "No lo sé.". Si cambian los documentos o el modelo, hay que recalibrarlo.
DISTANCIA_MAXIMA = 0.20

# Reintentos con backoff exponencial (1s, 2s) y un log por cada intento
# fallido: así se ve en consola qué falló y cuántas veces.
MAX_INTENTOS = 3

# Errores que no se arreglan reintentando (API key inválida, request mal
# armado): se corta en el primer intento en vez de esperar al pedo.
ERRORES_NO_REINTENTABLES = (AuthenticationError, PermissionDeniedError, BadRequestError)

_parser = PydanticOutputParser(pydantic_object=RespuestaLLM)

# Prompt "grounded": actúa como filtro de veracidad. {format_instructions}
# lo completa el parser con el esquema JSON de RespuestaLLM.
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un asistente técnico. Respondé ÚNICAMENTE en base al CONTEXTO "
            "que se te provee. Cada fragmento del contexto está etiquetado con "
            "el nombre del archivo del que proviene.\n"
            "Reglas:\n"
            "1. No uses conocimiento propio ni externo al CONTEXTO.\n"
            f"2. Si la respuesta no está en el CONTEXTO, respondé exactamente "
            f"\"{NO_LO_SE}\", con respuesta_en_contexto=false y fuentes_citadas vacía.\n"
            "3. Si la respuesta está, indicá en fuentes_citadas los nombres de "
            "archivo que usaste.\n\n"
            "{format_instructions}",
        ),
        ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ]
).partial(format_instructions=_parser.get_format_instructions())


# Una sola instancia del manager por proceso (cargar el modelo de embeddings es caro).
@lru_cache(maxsize=1)
def _get_manager() -> VectorMemoryManager:
    manager = VectorMemoryManager(PERSIST_PATH, COLLECTION_NAME)
    if manager.count() == 0:
        raise RuntimeError("La base vectorial está vacía. Corré primero: python ingesta.py")
    return manager


# Cadena LCEL: prompt -> LLM -> PydanticOutputParser. Se crea recién cuando
# hace falta, así una pregunta descartada por el umbral no necesita API key.
@lru_cache(maxsize=1)
def _get_chain():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("Falta ANTHROPIC_API_KEY. Copiá .env.example a .env y completala.")
    # temperature=0: en una respuesta grounded no se busca creatividad.
    # max_retries=0: los reintentos los manejamos nosotros (con logs).
    llm = ChatAnthropic(model=LLM_MODEL, temperature=0, max_tokens=1024, max_retries=0)
    return PROMPT | llm | _parser


# Retriever: búsqueda semántica + filtro de relevancia.
async def _recuperar(query: RAGQuery) -> List[FragmentoRecuperado]:
    manager = _get_manager()
    # La búsqueda (embedding + ChromaDB) es sincrónica: se manda a un hilo
    # con to_thread para que el event loop siga libre mientras tanto.
    resultados = await asyncio.to_thread(manager.semantic_search, query.pregunta, query.top_k)

    fragmentos = [
        FragmentoRecuperado(texto=r["documento"], fuente=r["metadata"]["fuente"], distancia=round(r["distancia"], 4))
        for r in resultados
        if r["distancia"] <= DISTANCIA_MAXIMA
    ]
    logger.info(
        "Retriever: %d/%d fragmento(s) pasaron el umbral %.2f (distancias: %s)",
        len(fragmentos), len(resultados), DISTANCIA_MAXIMA,
        [round(r["distancia"], 3) for r in resultados],
    )
    return fragmentos


# Llama a la cadena LCEL de forma asíncrona, reintentando con un log por intento.
async def _generar_con_reintentos(contexto: str, pregunta: str) -> RespuestaLLM:
    chain = _get_chain()
    for intento in range(1, MAX_INTENTOS + 1):
        try:
            logger.info("LLM: intento %d/%d", intento, MAX_INTENTOS)
            respuesta = await chain.ainvoke({"contexto": contexto, "pregunta": pregunta})
            if intento > 1:
                logger.info("LLM: respondió bien en el intento %d", intento)
            return respuesta
        except ERRORES_NO_REINTENTABLES as e:
            logger.error("LLM: error no reintentable (%s): %s", type(e).__name__, e)
            raise
        except Exception as e:
            # Acá caen tanto errores de red/rate limit como salidas que no
            # cumplen el esquema (OutputParserException).
            if intento == MAX_INTENTOS:
                logger.error("LLM: falló tras %d intentos. Último error (%s): %s", MAX_INTENTOS, type(e).__name__, e)
                raise
            espera = 2 ** (intento - 1)
            logger.warning(
                "LLM: intento %d/%d falló (%s: %s). Reintento en %ds",
                intento, MAX_INTENTOS, type(e).__name__, str(e)[:200], espera,
            )
            await asyncio.sleep(espera)


# Pipeline RAG completo y asíncrono:
# pregunta -> retriever -> filtro de relevancia -> LLM grounded -> RAGResponse.
async def get_rag_response(query: str, top_k: int = 4) -> RAGResponse:
    # Validación de entrada (pregunta no vacía, top_k entre 3 y 5).
    consulta = RAGQuery(pregunta=query, top_k=top_k)
    logger.info("Pregunta: %r (top_k=%d)", consulta.pregunta, consulta.top_k)

    fragmentos = await _recuperar(consulta)

    # Barrera 1: nada relevante -> "No lo sé." sin gastar una llamada al LLM.
    if not fragmentos:
        logger.warning("Sin contexto relevante: se responde '%s' sin llamar al LLM", NO_LO_SE)
        return RAGResponse(pregunta=consulta.pregunta, respuesta=NO_LO_SE, respuesta_encontrada=False)

    contexto = "\n\n---\n\n".join(f"[{f.fuente}]\n{f.texto}" for f in fragmentos)
    salida = await _generar_con_reintentos(contexto, consulta.pregunta)

    # Barrera 2: el LLM dice que el dato no está en el contexto.
    if not salida.respuesta_en_contexto:
        logger.info("El LLM indicó que la respuesta no está en el contexto")
        return RAGResponse(
            pregunta=consulta.pregunta,
            respuesta=NO_LO_SE,
            respuesta_encontrada=False,
            fragmentos_recuperados=fragmentos,
        )

    # Solo se aceptan fuentes que realmente se le pasaron al modelo: si cita
    # un archivo que no estaba en el contexto, se descarta.
    recuperadas = {f.fuente for f in fragmentos}
    fuentes = [f for f in dict.fromkeys(salida.fuentes_citadas) if f in recuperadas]
    descartadas = set(salida.fuentes_citadas) - recuperadas
    if descartadas:
        logger.warning("Se descartaron fuentes citadas que no estaban en el contexto: %s", descartadas)

    logger.info("Respuesta generada. Fuentes: %s", fuentes)
    return RAGResponse(
        pregunta=consulta.pregunta,
        respuesta=salida.respuesta,
        respuesta_encontrada=True,
        fuentes=fuentes,
        fragmentos_recuperados=fragmentos,
    )
