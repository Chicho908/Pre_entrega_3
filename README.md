# Pre-entrega 3: Sistema de recuperacion semantica local (RAG)

Sistema RAG end-to-end (Modulo 3, CoderHouse - Desarrollo de aplicaciones con
LLMs). Indexa documentos propios en una base vectorial local (ChromaDB) y
responde preguntas con Claude usando unicamente la informacion recuperada. Si
la respuesta no esta en los documentos, responde "No lo sé." en vez de inventar.

## Que hace

1. **`document_processor.py`** — limpia el texto y lo fragmenta con
   `RecursiveCharacterTextSplitter` (Unidad 2): chunks de 500 tokens con 50 de
   overlap, medidos en tokens reales con `tiktoken` y no en caracteres.
2. **`vector_memory_manager.py`** — maneja la coleccion de ChromaDB persistente
   en disco (`upsert`, busqueda semantica, borrado). Genera los embeddings con un
   unico modelo (`intfloat/multilingual-e5-small`) tanto para indexar como para
   consultar, y lo guarda en la metadata de la coleccion para detectar si alguien
   intenta consultar con otro modelo.
3. **`ingesta.py`** — script de ingesta: recorre `/data` (`.txt` y `.md`),
   fragmenta cada documento y lo guarda en `./vectorstore` con ids
   deterministicos (hash del contenido). Si la base ya existe, no reindexa.
4. **`schemas.py`** — modelos Pydantic: `RAGQuery` (valida la pregunta y el
   `top_k` entre 3 y 5), `FragmentoRecuperado`, `RespuestaLLM` (lo que tiene que
   devolver el modelo) y `RAGResponse` (salida final: texto + fuentes).
5. **`rag.py`** — `get_rag_response(query: str)`, la funcion asincrona central:
   busca los fragmentos mas parecidos (en un hilo aparte con `asyncio.to_thread`),
   descarta los que superan un umbral de distancia, y genera la respuesta con la
   cadena LCEL `prompt | ChatAnthropic | PydanticOutputParser` llamada con
   `.ainvoke()`. Cada reintento al LLM queda registrado en los logs.
6. **`test_rag.py`** — pruebas: una pregunta real y dos preguntas trampa, corridas
   en paralelo con `asyncio.gather`. Guarda el resultado en `evidencia_pruebas.json`.
7. **`main.py`** — modo interactivo por consola para hacer preguntas a mano.

## Como correrlo

```bash
python -m venv venv
venv\Scripts\activate          # en Windows
# source venv/bin/activate     # en Linux/Mac
pip install -r requirements.txt
copy .env.example .env         # completar con tu ANTHROPIC_API_KEY real
python ingesta.py              # indexa /data en ./vectorstore (una sola vez)
python test_rag.py             # corre las 3 pruebas y genera evidencia_pruebas.json
python main.py                 # modo interactivo
```

La primera vez, `ingesta.py` descarga el modelo de embeddings (~470 MB). Para
reindexar desde cero despues de cambiar los documentos: `python ingesta.py --forzar`.

## Estructura del repositorio

```
.
├── data/                        # dataset de ejemplo (4 documentos sobre RAG)
│   ├── rag_fundamentos.txt
│   ├── embeddings_y_bases_vectoriales.txt
│   ├── estrategias_de_chunking.txt
│   └── prompting_grounded_y_evaluacion.md
├── document_processor.py        # limpieza + chunking por tokens
├── vector_memory_manager.py     # ChromaDB persistente + modelo de embeddings unico
├── ingesta.py                   # script de ingesta (/data -> ./vectorstore)
├── schemas.py                   # modelos Pydantic de entrada y salida
├── rag.py                       # get_rag_response async + cadena LCEL + reintentos con logs
├── test_rag.py                  # pruebas: pregunta real + 2 preguntas trampa
├── evidencia_pruebas.json       # resultado de test_rag.py
├── main.py                      # modo interactivo por consola
├── requirements.txt             # dependencias con versiones fijadas
├── .env.example                 # plantilla de variables de entorno (sin la key real)
└── .gitignore                   # excluye .env, venv/, vectorstore/ y __pycache__/
```

## Ejemplo de salida esperada

`test_rag.py` corre tres casos. El resultado real completo esta en `evidencia_pruebas.json`.

**Pregunta real** — la respuesta esta en los documentos:

```json
{
  "pregunta": "¿Qué es el chunk overlap y para qué sirve?",
  "respuesta": "El chunk overlap (superposición) es la cantidad de tokens que se repiten entre un chunk y el siguiente. Su función es evitar que una idea importante quede cortada exactamente en el límite entre dos chunks...",
  "respuesta_encontrada": true,
  "fuentes": ["estrategias_de_chunking.txt"]
}
```

**Pregunta trampa fuera del tema** — ningun fragmento pasa el umbral, no se llama al LLM:

```json
{
  "pregunta": "¿Cuál es la mejor receta de milanesas con puré?",
  "respuesta": "No lo sé.",
  "respuesta_encontrada": false,
  "fuentes": [],
  "fragmentos_recuperados": []
}
```

**Pregunta trampa dentro del tema** — HNSW aparece en los documentos, pero no quien
lo invento: el fragmento llega al LLM y aun asi no alucina:

```json
{
  "pregunta": "¿Quién inventó el algoritmo HNSW y en qué año se publicó?",
  "respuesta": "No lo sé.",
  "respuesta_encontrada": false,
  "fuentes": []
}
```

## Decisiones de diseno

- **`ChatAnthropic` como proveedor**: la consigna permite elegir el provider; se
  mantuvo Anthropic, igual que en la Pre-entrega 1 y 2, con la API key ya probada.
- **`PydanticOutputParser` en vez de `with_structured_output`**: esta vez la
  consigna pide explicitamente que la salida pase por un `PydanticOutputParser`.
  El parser agrega el esquema JSON al prompt y valida la respuesta del modelo; si
  no cumple, se reintenta.
- **Reintentos con logs** (sugerencia de la correccion de la Pre-entrega 2): en
  vez de `.with_retry()`, que reintenta sin dejar rastro, cada intento fallido
  queda registrado con su numero, el tipo de error y la espera (1s, 2s). Los errores
  que no se arreglan reintentando (API key invalida) cortan en el primer intento.
- **Chunks de 500 tokens con 50 de overlap**: los valores minimos que pide la
  consigna, medidos en tokens para respetar el limite real del modelo. Los
  separadores jerarquicos (parrafo, linea, oracion) evitan cortar una idea a la mitad.
- **`multilingual-e5-small` como modelo de embeddings**: el modelo por defecto
  (`all-MiniLM-L6-v2`) esta entrenado en ingles y corta el texto a 256 tokens,
  y los chunks tienen ~450 tokens en español. Este entiende español y acepta 512
  tokens. Esta definido en una sola constante y se verifica al abrir la coleccion,
  para evitar indexar con un modelo y consultar con otro.
- **Doble barrera contra alucinaciones**:
  1. Umbral de distancia (`DISTANCIA_MAXIMA = 0.20`), calibrado con 10 preguntas
     del tema (distancia 0.09 a 0.19) y 8 ajenas (0.20 a 0.29). Lo irrelevante se
     corta antes y no gasta una llamada al LLM.
  2. Prompt "grounded": si la pregunta es del tema pero el dato no esta, el modelo
     tiene que responder exactamente "No lo sé.".
- **`top_k = 4`, validado entre 3 y 5**: evita el "contexto infinito" (limite de
  tokens y el problema de "Lost in the Middle").
- **Fuentes verificadas**: si el modelo cita un archivo que no estaba en el
  contexto, se descarta.
- **Asincronia real**: la busqueda en ChromaDB corre en un hilo aparte
  (`asyncio.to_thread`) y el LLM con `.ainvoke()`, asi el event loop no se bloquea.
  `test_rag.py` corre las 3 preguntas en paralelo para demostrarlo.
- **Persistencia**: `ingesta.py` verifica si `./vectorstore` ya tiene datos antes
  de reindexar, para no recalcular embeddings sin necesidad.
- **`temperature=0`**: en una respuesta basada en contexto no se busca creatividad.
- **Versiones fijadas en `requirements.txt`**: las mismas probadas en este entorno,
  para que el repo sea reproducible.
- **`.env` fuera del repo**: la API key real nunca se sube (esta en `.gitignore`);
  solo se versiona `.env.example` como plantilla.
