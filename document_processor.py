# re se usa para limpiar el texto con expresiones regulares (espacios, saltos de linea).
import re

# tiktoken es el tokenizador de OpenAI: permite contar tokens reales en vez de caracteres.
import tiktoken

# List tipa la lista de chunks que devuelve process_document.
from typing import List

# RecursiveCharacterTextSplitter es el splitter que vimos en la Unidad 2: corta
# respetando una jerarquia de separadores (parrafo -> linea -> oracion -> palabra).
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Limpia y fragmenta documentos en chunks medidos en TOKENS (no caracteres).
class DocumentProcessor:
    def __init__(
        self,
        model_encoding: str = "cl100k_base",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        # Inicializar el encoding de tiktoken: es el mismo tokenizador que usan
        # los modelos de OpenAI (cl100k_base = GPT-3.5/GPT-4), asi que contar
        # tokens con esto refleja el limite real del modelo, no una aproximacion.
        self.tokenizer = tiktoken.get_encoding(model_encoding)

        # Configurar RecursiveCharacterTextSplitter usando una funcion propia
        # (self.calculate_tokens) como length_function, en vez del conteo por
        # caracteres que trae por defecto. Asi el chunk_size se respeta en
        # TOKENS, que es lo que realmente limita al modelo de embeddings.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.calculate_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # Limpia el texto eliminando espacios duplicados y saltos de linea innecesarios.
    def clean_text(self, text: str) -> str:
        text = re.sub(r"\r\n?", "\n", text)      # normaliza CRLF/CR a LF
        text = re.sub(r"[ \t]+", " ", text)      # colapsa espacios y tabs repetidos
        text = re.sub(r"\n{3,}", "\n\n", text)   # colapsa mas de 2 saltos de linea seguidos
        return text.strip()

    # Calcula la cantidad de tokens usando el tokenizer de tiktoken.
    def calculate_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # Pipeline: limpieza -> fragmentacion.
    def process_document(self, raw_text: str) -> List[str]:
        # 1. Limpiar el texto
        texto_limpio = self.clean_text(raw_text)

        if not texto_limpio:
            return []

        # 2. Aplicar el splitter (ya configurado para medir en tokens)
        chunks = self.splitter.split_text(texto_limpio)

        # 3. Retornar los chunks procesados
        return chunks
