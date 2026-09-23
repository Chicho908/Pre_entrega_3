# Prompting grounded, parámetros de recuperación y evaluación de un RAG

## El prompt de sistema como filtro de veracidad

En un sistema RAG, el prompt de sistema cumple el rol de un filtro de veracidad: le indica al modelo de lenguaje que debe responder únicamente en base al contexto recuperado y no con el conocimiento general que aprendió durante su entrenamiento. Una instrucción típica es: "Respondé solo basándote en el CONTEXTO proporcionado. Si la respuesta no está allí, respondé exactamente: No lo sé". Definir una frase fija para el caso negativo tiene una ventaja práctica: el resto del sistema puede detectarla de forma confiable y, por ejemplo, no mostrar fuentes cuando el modelo no pudo responder.

Esta instrucción no reemplaza al umbral de relevancia en la etapa de recuperación, sino que lo complementa. El umbral descarta preguntas que no tienen nada que ver con la base de conocimiento; el prompt grounded cubre el caso más sutil, en el que la pregunta sí pertenece al dominio y se recuperan fragmentos relacionados, pero ninguno contiene el dato puntual que se pide.

## Cuántos fragmentos pasarle al modelo: top_k y "Lost in the Middle"

El parámetro top_k define cuántos fragmentos se recuperan de la base vectorial para cada consulta. Un valor muy bajo puede dejar afuera información necesaria. Un valor muy alto genera el problema del "contexto infinito": aumenta el costo y la latencia de cada llamada, puede superar el límite de tokens del modelo y, sobre todo, degrada la calidad de la respuesta. Diversos estudios mostraron que los modelos de lenguaje prestan más atención a la información ubicada al principio y al final del contexto, y tienden a ignorar la que queda en el medio, un fenómeno conocido como "Lost in the Middle". Por eso, en la práctica se recomienda un top_k de entre 3 y 5 fragmentos.

## Consistencia del modelo de embeddings

El error más común al construir un RAG es usar un modelo de embeddings para indexar los documentos y otro distinto para convertir la pregunta del usuario en vector. Cada modelo genera su propio espacio vectorial, con dimensiones y geometría diferentes, por lo que comparar vectores de dos modelos distintos no tiene sentido y los resultados de la búsqueda son prácticamente aleatorios. Una buena práctica es definir el nombre del modelo de embeddings en un único lugar de la configuración y guardarlo también como metadata de la colección, para poder verificar al consultar que se está usando el mismo modelo con el que se indexó.

## Salida estructurada

Para que la respuesta del sistema sea fácil de consumir por otros programas, conviene que el modelo no devuelva texto libre sino un objeto con una estructura definida, por ejemplo un JSON con la respuesta y la lista de fuentes utilizadas. Herramientas como PydanticOutputParser agregan al prompt las instrucciones de formato derivadas de un modelo Pydantic y luego validan la salida del modelo contra ese esquema. Si la salida no cumple el esquema, se produce un error de parseo que puede manejarse con reintentos, idealmente dejando registro en los logs de cada intento fallido para tener visibilidad de qué está fallando.

## Cómo evaluar que el sistema no alucina

Un RAG se evalúa con dos tipos de preguntas. Las preguntas positivas tienen su respuesta en los documentos y permiten verificar que la recuperación encuentra el fragmento correcto y que la respuesta cita la fuente adecuada. Las preguntas trampa no tienen respuesta en los documentos y permiten verificar que el sistema admite que no sabe en lugar de inventar. Conviene incluir dos variantes de pregunta trampa: una totalmente ajena al dominio, que debería ser descartada por el umbral de relevancia, y otra del mismo dominio pero con un dato que no figura en la base, que pone a prueba la instrucción del prompt de sistema.
