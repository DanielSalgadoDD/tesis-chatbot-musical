## **Herramientas, Librerías y Tecnologías**

A continuación, se describen las principales herramientas, APIs y librerías de software utilizadas para el desarrollo de este proyecto, desde el análisis de datos hasta la implementación del prototipo conversacional.

---

### **APIs y Servicios Externos**

#### **OpenAI API**
**Descripción General:**  
Interfaz de programación de aplicaciones que proporciona acceso a los modelos de lenguaje a gran escala (LLMs) de OpenAI, como la familia **GPT (Generative Pre-trained Transformer)**.  
Permite integrar capacidades avanzadas de comprensión y generación de lenguaje natural en aplicaciones.

**Uso en este Proyecto:**  
Constituyó el núcleo de la interfaz conversacional.  
Se utilizó el modelo **GPT-4 Turbo** para:
1. Interpretar las solicitudes del usuario en lenguaje natural.  
2. Estructurar la intención y parámetros en formato JSON para el backend.  
3. Generar una respuesta final coherente y empática que acompañara las recomendaciones musicales.

---

#### **Spotify API**
**Descripción General:**  
API que permite acceder al catálogo musical de Spotify, incluyendo información sobre canciones, artistas, álbumes y playlists.

**Uso en este Proyecto:**  
Se utilizó para **enriquecer la experiencia de usuario**.  
Permitió obtener las **carátulas de los álbumes** y habilitar la **reproducción de audio directa** desde el chatbot, mejorando la presentación de los resultados.

---

### **Motor de Búsqueda y Machine Learning**

#### **FAISS (Facebook AI Similarity Search)**
**Descripción General:**  
Librería de código abierto desarrollada por Facebook AI para la búsqueda y el clustering de vectores densos de alta eficiencia.  
Optimizada para manejar grandes volúmenes de datos y realizar búsquedas de vecinos más cercanos en milisegundos.

**Uso en este Proyecto:**  
Fue el **componente central del motor de recomendación**.  
Se utilizó para **indexar los vectores ponderados** de las ~650.000 canciones del corpus, permitiendo comparar consultas en tiempo real y recuperar las canciones más similares semántica y emocionalmente.

---

#### **Scikit-learn**
**Descripción General:**  
Librería de aprendizaje automático que ofrece herramientas para clasificación, regresión, clustering y preprocesamiento.

**Uso en este Proyecto:**  
Se utilizó su implementación del algoritmo **K-Nearest Neighbors (KNN)** para imputar géneros musicales faltantes y para calcular métricas de evaluación durante la comparación de modelos (baselines vs. modelo propuesto).

---

#### **Torch / TensorFlow**
**Descripción General:**  
Frameworks de aprendizaje profundo ampliamente utilizados para construir, entrenar y desplegar redes neuronales.

**Uso en este Proyecto:**  
Se emplearon indirectamente como **bases de ejecución** de los modelos preentrenados de **Hugging Face** (DistilBERT, RoBERTa, etc.).  
Además, se usaron en la fase experimental para construir y entrenar el modelo predictivo **LSTM/MLP** comparado con los baselines.

---

### **Desarrollo del Backend y Orquestación**

#### **FastAPI**
**Descripción General:**  
Framework web moderno y de alto rendimiento para construir APIs con Python.  
Destaca por su rapidez, su tipado fuerte y la generación automática de documentación interactiva.

**Uso en este Proyecto:**  
Constituyó el **backend** de la aplicación.  
Se utilizó para crear los **endpoints** que reciben las solicitudes del frontend del chatbot, gestionar las llamadas a OpenAI y FAISS, y devolver los resultados procesados.

---

#### **LangChain**
**Descripción General:**  
Framework diseñado para simplificar el desarrollo de aplicaciones basadas en **modelos de lenguaje (LLMs)**.  
Permite gestionar prompts, mantener memoria conversacional y encadenar múltiples llamadas a modelos o APIs.

**Uso en este Proyecto:**  
Facilitó la **orquestación del flujo conversacional** con el modelo de OpenAI.  
Gestionó el **contexto del diálogo**, los **prompts estructurados** y el **parseo del JSON** que representaba la intención del usuario.

---

#### **AsyncIO**
**Descripción General:**  
Librería de Python para escribir código concurrente usando `async/await`.  
Ideal para aplicaciones con muchas operaciones de entrada/salida (I/O), como las llamadas a APIs externas.

**Uso en este Proyecto:**  
Fue esencial para la **eficiencia y capacidad de respuesta del chatbot**.  
Permitió que el backend realizara múltiples peticiones (a OpenAI, Spotify, FAISS) **sin bloquear el servidor**, habilitando la **respuesta en streaming**.

---

### **Análisis y Manipulación de Datos**

#### **Pandas**
**Descripción General:**  
Librería estándar en Python para la manipulación y análisis de datos tabulares mediante estructuras tipo **DataFrame**.

**Uso en este Proyecto:**  
Indispensable en el **preprocesamiento y análisis exploratorio (EDA)**.  
Se utilizó para limpiar las letras, fusionar datasets, manejar valores faltantes e implementar las operaciones estadísticas del sistema.

---

#### **NumPy**
**Descripción General:**  
Paquete fundamental para la computación científica en Python.  
Proporciona estructuras de datos y operaciones matemáticas de alto rendimiento sobre **matrices y vectores**.

**Uso en este Proyecto:**  
Base de las operaciones numéricas del sistema.  
Se utilizó en la **creación y normalización de los vectores ponderados** y en la preparación de datos para el indexado con FAISS.

---

### **Visualización y Librerías Estándar**

#### **Matplotlib / Seaborn**
**Descripción General:**  
Herramientas de visualización en Python.  
**Matplotlib** proporciona gráficos básicos, mientras que **Seaborn** ofrece una interfaz de alto nivel para análisis estadístico y visualizaciones más atractivas.

**Uso en este Proyecto:**  
Se emplearon para los gráficos del **Análisis Exploratorio de Datos (EDA)**, incluyendo histogramas, mapas de calor y diagramas de caja.

---

#### **Spotipy**
**Descripción General:**  
Cliente ligero de Python para la **API Web de Spotify**, que simplifica la autenticación y las consultas al catálogo musical.

**Uso en este Proyecto:**  
Facilitó la conexión con Spotify, permitiendo obtener carátulas, nombres de artistas y datos de reproducción de manera eficiente.

---

#### **JSON / re / os / dotenv**
**Descripción General:**  
Conjunto de librerías estándar de Python:  
- **JSON**: Intercambio estructurado de datos.  
- **re**: Expresiones regulares para procesar texto.  
- **os** y **dotenv**: Gestión de variables de entorno y configuración segura.

**Uso en este Proyecto:**  
- **JSON**: Formato de comunicación entre el modelo y el backend.  
- **re**: Limpieza de texto de las letras musicales.  
- **os / dotenv**: Gestión de claves y credenciales de forma segura mediante variables de entorno.

---
