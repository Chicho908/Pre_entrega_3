# List tipa las listas del contrato (fuentes citadas, fragmentos recuperados).
from typing import List

# Pydantic valida y documenta la estructura de datos de entrada y salida del sistema.
from pydantic import BaseModel, Field

# Frase fija que el sistema usa cuando la respuesta no está en los documentos.
# Tenerla en una constante permite detectarla de forma confiable en los tests.
NO_LO_SE = "No lo sé."


# Input validado para una consulta al sistema RAG.
class RAGQuery(BaseModel):
    # min_length=3 evita aceptar una pregunta vacía o de un par de letras sueltas.
    pregunta: str = Field(..., min_length=3, max_length=500, description="Pregunta del usuario")
    # ge=3 / le=5: rango que pide la consigna para evitar el "contexto infinito"
    # (límite de tokens y degradación de atención tipo "Lost in the Middle").
    top_k: int = Field(default=4, ge=3, le=5, description="Cantidad de chunks a recuperar")


# Un chunk recuperado de la base vectorial, con su distancia a la pregunta.
class FragmentoRecuperado(BaseModel):
    texto: str
    fuente: str
    # La distancia queda expuesta para poder auditar por qué un chunk se
    # consideró relevante o no (ver DISTANCIA_MAXIMA en rag.py).
    distancia: float


# Contrato que el LLM tiene que devolver. PydanticOutputParser inyecta este
# esquema en el prompt y después valida el JSON de la respuesta contra él:
# si no cumple, se reintenta (con logs de cada intento).
class RespuestaLLM(BaseModel):
    respuesta: str = Field(
        ...,
        description=f"Respuesta basada SOLO en el contexto. Si no está en el contexto: exactamente '{NO_LO_SE}'",
    )
    respuesta_en_contexto: bool = Field(
        ...,
        description="true si la respuesta se encontró en el contexto, false si no",
    )
    # default_factory=list: si el modelo no encuentra la respuesta, devuelve
    # una lista vacía en vez de fallar la validación.
    fuentes_citadas: List[str] = Field(
        default_factory=list,
        description="Nombres de archivo del contexto usados para responder (lista vacía si no se encontró la respuesta)",
    )


# Output final del sistema: respuesta + referencias + evidencia de recuperación.
class RAGResponse(BaseModel):
    pregunta: str
    respuesta: str
    # False cuando el sistema respondió "No lo sé." (sea por el umbral de
    # relevancia o porque el LLM no encontró el dato en el contexto).
    respuesta_encontrada: bool
    # Fuentes que el LLM citó, filtradas contra las realmente recuperadas
    # (una fuente inventada por el modelo no pasa).
    fuentes: List[str] = Field(default_factory=list)
    # Todos los fragmentos que pasaron el umbral y se enviaron como contexto.
    fragmentos_recuperados: List[FragmentoRecuperado] = Field(default_factory=list)
