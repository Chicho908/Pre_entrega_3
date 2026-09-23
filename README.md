# Pre-entrega 3: Sistema de recuperación semántica local (RAG)

Sistema RAG end-to-end: indexa documentos propios en una base vectorial local
(ChromaDB) y responde preguntas con Claude (Anthropic) usando **solo** la
información recuperada. Si la respuesta no está en los documentos, responde
`"No lo sé."`.

```
pregunta ──► RAGQuery (Pydantic) ──► retriever async (embedding + ChromaDB, top_k=4)
         ──► filtro de relevancia (distancia ≤ 0.20) ──► ¿hay contexto?
                 │ no ──► "No lo sé." (sin llamar al LLM)
                 │ sí ──► cadena LCEL: prompt | ChatAnthropic | PydanticOutputParser
                          (await + reintentos con logs) ──► RAGResponse (texto + fuentes)
```

## Cómo ejecutarlo (Windows / PowerShell)

```powershell
# 1. Entorno virtual y dependencias
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. API key (el archivo .env NO se sube al repo)
copy .env.example .env      # y completar ANTHROPIC_API_KEY

# 3. Ingesta: /data -> chunks -> embeddings -> ./vectorstore
python ingesta.py           # si la base ya existe, la saltea
python ingesta.py --forzar  # reindexa desde cero (si cambiaste /data)

# 4. Pruebas (caso real + 2 preguntas trampa) -> genera evidencia_pruebas.json
python test_rag.py

# 5. Modo interactivo
python main.py
```

> La primera ejecución descarga el modelo de embeddings (~470 MB) y queda cacheado.

En Linux/Mac: `python3 -m venv venv` y `source venv/bin/activate`.

## Estructura

```
.
├── data/                              # dataset de ejemplo (4 documentos .txt/.md)
│   ├── rag_fundamentos.txt
│   ├── embeddings_y_bases_vectoriales.txt
│   ├── estrategias_de_chunking.txt
│   └── prompting_grounded_y_evaluacion.md
├── document_processor.py              # limpieza + chunking por tokens (RecursiveCharacterTextSplitter)
├── vector_memory_manager.py           # ChromaDB persistente + modelo de embeddings único
├── ingesta.py                         # script de ingesta, con verificación de persistencia
├── schemas.py                         # modelos Pydantic de entrada y salida
├── rag.py                             # get_rag_response(query: str) asíncrona + cadena LCEL
├── test_rag.py                        # pruebas: caso real + trampas, en paralelo
├── evidencia_pruebas.json             # salida de test_rag.py (evidencia)
├── main.py                            # consola interactiva
├── requirements.txt
├── .env.example
└── .gitignore                         # excluye .env, venv/, vectorstore/, __pycache__/
```

## Cómo cumple cada criterio

### 1. Chunking e ingesta (25%)
- `RecursiveCharacterTextSplitter` con **500 tokens y 50 de overlap**, medidos en
  tokens reales (tiktoken) y no en caracteres. Separadores jerárquicos
  (`\n\n`, `\n`, `. `, espacio) para cortar por párrafo/oración y no a la mitad de una idea.
- Lee `.txt` y `.md` de `/data`. Cada chunk guarda `fuente` y `posicion` como metadata.
- Ids determinísticos (SHA-256 del contenido) + `upsert`: reindexar no duplica.
- **Persistencia**: si `./vectorstore` ya tiene datos, la ingesta se saltea.
- **Mismo modelo para indexar y consultar**: `intfloat/multilingual-e5-small`,
  definido en una sola constante y guardado en la metadata de la colección. Si
  alguien consulta una base indexada con otro modelo, el sistema lo detecta y corta.
  Se eligió porque entiende español y acepta 512 tokens (el anterior,
  `all-MiniLM-L6-v2`, es solo inglés y truncaba los chunks a 256 tokens).

### 2. Arquitectura asíncrona y recuperación (35%)
- `async def get_rag_response(query: str)` en `rag.py`.
- La búsqueda vectorial corre con `asyncio.to_thread` (no bloquea el event loop) y
  el LLM se llama con `await chain.ainvoke(...)`.
- `top_k` validado entre 3 y 5 (default 4) para evitar el "contexto infinito".
- `test_rag.py` corre las 3 consultas **en paralelo** con `asyncio.gather`.
- Reintentos con backoff exponencial y **un log por cada intento fallido**
  (tipo de error, intento N/3, espera). Los errores que no se arreglan
  reintentando (API key inválida) cortan en el primer intento.

### 3. Validación con Pydantic (20%)
- `RAGQuery`: valida la entrada (largo de la pregunta, rango de `top_k`).
- `RespuestaLLM`: la cadena LCEL termina en `PydanticOutputParser`, que inyecta
  el esquema en el prompt y valida el JSON del modelo (si no cumple, se reintenta).
- `RAGResponse`: salida final con `respuesta`, `respuesta_encontrada`, `fuentes`
  y `fragmentos_recuperados` (texto, archivo y distancia de cada chunk).
- Las fuentes que cita el LLM se cruzan con las realmente recuperadas: una
  fuente inventada se descarta.

### 4. Robustez y evidencia (20%)
Doble barrera contra alucinaciones:
1. **Umbral de distancia** (`DISTANCIA_MAXIMA = 0.20`, calibrado con 10
   preguntas del dominio y 8 ajenas): lo irrelevante ni llega al LLM.
2. **Prompt grounded**: si la pregunta es del tema pero el dato no está, el
   modelo debe responder exactamente `"No lo sé."`.

`test_rag.py` prueba las dos barreras con asserts y guarda el resultado en
`evidencia_pruebas.json`:

| Caso | Pregunta | Resultado esperado |
|---|---|---|
| Real | ¿Qué es el chunk overlap y para qué sirve? | Responde y cita `estrategias_de_chunking.txt` |
| Trampa fuera de dominio | ¿Cuál es la mejor receta de milanesas con puré? | `No lo sé.` (la corta el umbral, sin llamar al LLM) |
| Trampa dentro del dominio | ¿Quién inventó el algoritmo HNSW y en qué año se publicó? | `No lo sé.` (pasa el umbral, el LLM no inventa) |

**Credenciales**: la API key se lee de `.env` con `python-dotenv`; `.env` está en
`.gitignore` y solo se versiona `.env.example`. Si falta la key, el sistema
avisa con un mensaje claro.
