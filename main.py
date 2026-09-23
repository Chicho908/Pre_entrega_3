# main.py: modo interactivo por consola para probar el sistema RAG a mano.
# Corré primero 'python ingesta.py' una vez para poblar la base vectorial.

# asyncio corre el loop asíncrono y lee el input sin bloquearlo (to_thread).
import asyncio

# ValidationError es lo que tira RAGQuery si la pregunta no es válida.
from pydantic import ValidationError

from rag import get_rag_response


async def main():
    print("Sistema RAG listo. Escribí una pregunta (o 'salir' para terminar).\n")
    while True:
        pregunta = (await asyncio.to_thread(input, "> ")).strip()
        if pregunta.lower() in ("salir", "exit", "quit"):
            break

        try:
            respuesta = await get_rag_response(pregunta)
        except ValidationError:
            print("Pregunta inválida: tiene que tener entre 3 y 500 caracteres.\n")
            continue
        except RuntimeError as e:
            print(f"Error: {e}\n")
            break

        print(f"\n{respuesta.respuesta}\n")
        if respuesta.fuentes:
            print("Fuentes:", ", ".join(respuesta.fuentes))
        print()


if __name__ == "__main__":
    asyncio.run(main())
