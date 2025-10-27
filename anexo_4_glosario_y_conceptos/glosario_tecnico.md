# Glosario Técnico

## A. Inteligencia Artificial y Aprendizaje Automático

### **Inteligencia Artificial (IA)**
La Inteligencia Artificial es un campo amplio de la informática dedicado a la creación de sistemas capaces de realizar tareas que normalmente requieren inteligencia humana. Esto incluye actividades como el razonamiento, el aprendizaje, la percepción y la resolución de problemas.  
**Referencia:** Russell, S. J., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

### **Aprendizaje Automático (Machine Learning)**
Es una rama de la IA que se centra en el desarrollo de algoritmos que permiten a las computadoras aprender de los datos y mejorar su rendimiento en una tarea sin ser explícitamente programadas para ello. En lugar de seguir reglas predefinidas, estos modelos identifican patrones en los datos de entrenamiento para hacer predicciones o tomar decisiones.  
**Referencia:** Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

### **Aprendizaje Profundo (Deep Learning)**
Es un subcampo del Aprendizaje Automático que utiliza Redes Neuronales Artificiales con múltiples capas (arquitecturas "profundas") para aprender representaciones de datos con varios niveles de abstracción. Ha demostrado ser especialmente eficaz en tareas complejas como el reconocimiento de imágenes y el procesamiento del lenguaje.  
**Referencia:** Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [http://www.deeplearningbook.org](http://www.deeplearningbook.org)

### **Red Neuronal Artificial**
Un modelo computacional inspirado en la estructura y función de las redes neuronales biológicas. Consiste en nodos interconectados ("neuronas") organizados en capas. Cada conexión tiene un peso asociado que se ajusta durante el entrenamiento para que la red aprenda a mapear entradas a salidas deseadas.  
**Referencia:** Haykin, S. (2009). *Neural Networks and Learning Machines* (3rd ed.). Pearson.

### **Perceptrón Multicapa (MLP)**
Un tipo de red neuronal artificial de alimentación hacia adelante (*feedforward*) que consta de al menos tres capas de nodos: una capa de entrada, una o más capas ocultas y una capa de salida. Los MLP son capaces de aprender funciones no lineales, lo que los hace aplicables a una amplia gama de problemas de clasificación y regresión.  
**Referencia:** Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning* (Chapter 6). MIT Press.

### **Red Recurrente (RNN) y LSTM (Long Short-Term Memory)**
Una RNN es un tipo de red neuronal diseñada para trabajar con datos secuenciales, como texto o series temporales, ya que sus conexiones forman un ciclo dirigido que le permite mantener una "memoria" de la información previa.  
La LSTM es una arquitectura avanzada de RNN diseñada para superar el problema de la desaparición del gradiente en secuencias largas, utilizando un mecanismo de "compuertas" (*gates*) que regulan el flujo de información, permitiéndole recordar dependencias a largo plazo.  
**Referencia:** Olah, C. (2015). *Understanding LSTM Networks.* Colah's Blog. [https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### **Embedding Vectorial**
Es una representación numérica de datos categóricos o textuales en un espacio vectorial denso y de baja dimensión. En este espacio, las entidades con significados o características similares se ubican cerca unas de otras, permitiendo que los algoritmos de aprendizaje automático procesen y comparen estas entidades de manera efectiva.  
**Referencia:** TensorFlow. (n.d.). *Embeddings.* [https://www.tensorflow.org/text/guide/word_embeddings](https://www.tensorflow.org/text/guide/word_embeddings)

### **Normalización L2**
Una técnica de regularización utilizada para prevenir el *overfitting* en modelos de aprendizaje automático. Añade un término a la función de pérdida que penaliza la suma de los cuadrados de los pesos del modelo, incentivando a la red a mantener pesos más pequeños y a evitar soluciones demasiado complejas que no generalizan bien a datos nuevos.  
**Referencia:** Ng, A. (2017). *Regularization.* Coursera - Deep Learning Specialization.

### **Función de Activación (ReLU, Sigmoid, Softmax, Tanh)**
Funciones matemáticas aplicadas a la salida de cada neurona para introducir no linealidad en el modelo, permitiéndole aprender relaciones complejas:

- **ReLU (Rectified Linear Unit):** Devuelve 0 si la entrada es negativa y la propia entrada si es positiva. Es computacionalmente eficiente y muy utilizada.  
- **Sigmoid:** Comprime cualquier valor de entrada en un rango entre 0 y 1, útil para predecir probabilidades.  
- **Softmax:** Generaliza la función sigmoide a múltiples clases, convirtiendo un vector de valores en una distribución de probabilidad.  
- **Tanh (Tangente Hiperbólica):** Comprime los valores en un rango entre -1 y 1.  

**Referencia:** Sharma, S., Sharma, S., & Athaiya, A. (2020). *Activation functions in neural networks.* *International Journal of Engineering Applied Sciences and Technology*, 4(12), 310–316.

### **Error Cuadrático Medio (MSE) o Función de Pérdida**
Una función que mide la discrepancia entre los valores predichos por un modelo y los valores reales. El MSE, por ejemplo, calcula el promedio de los errores al cuadrado. El objetivo del entrenamiento es minimizar el valor de esta función.  
**Referencia:** Janocha, K., & Czarnecki, W. M. (2017). *On loss functions for deep neural networks in classification.* *arXiv preprint arXiv:1702.05659.*

### **Gradiente y Retropropagación (Backpropagation)**
El gradiente es un vector que apunta en la dirección del máximo incremento de la función de pérdida. Durante el entrenamiento, se utiliza el descenso del gradiente para ajustar los pesos del modelo en la dirección opuesta al gradiente, reduciendo así el error.  
La retropropagación es el algoritmo que permite calcular de manera eficiente el gradiente de la función de pérdida con respecto a todos los pesos de la red, propagando el error desde la capa de salida hacia atrás.  
**Referencia:** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors.* *Nature*, 323(6088), 533–536.

### **Entrenamiento, Validación y Prueba**
Son las tres fases del desarrollo de un modelo:

- **Entrenamiento:** El modelo aprende los patrones a partir de un conjunto de datos de entrenamiento.  
- **Validación:** Se utiliza un conjunto de datos de validación para ajustar los hiperparámetros del modelo (ej. tasa de aprendizaje) y monitorear el *overfitting*.  
- **Prueba:** Se evalúa el rendimiento final del modelo en un conjunto de datos de prueba, que el modelo nunca ha visto antes, para obtener una estimación imparcial de su capacidad de generalización.  

**Referencia:** Brownlee, J. (2017). *What is the Difference Between Test and Validation Datasets?* Machine Learning Mastery.

### **Overfitting y Underfitting**
- **Overfitting (Sobreajuste):** Ocurre cuando un modelo aprende demasiado bien los datos de entrenamiento, incluyendo su ruido y detalles específicos, pero pierde la capacidad de generalizar a datos nuevos.  
- **Underfitting (Subajuste):** Ocurre cuando un modelo es demasiado simple para capturar los patrones subyacentes en los datos, resultando en un bajo rendimiento tanto en el conjunto de entrenamiento como en el de prueba.  
**Referencia:** IBM. (n.d.). *Overfitting.* [https://www.ibm.com/cloud/learn/overfitting](https://www.ibm.com/cloud/learn/overfitting)

---

## B. Procesamiento del Lenguaje Natural (PLN)

### **Procesamiento del Lenguaje Natural (PLN / NLP)**
Es un campo interdisciplinario de la IA y la lingüística que se ocupa de la interacción entre las computadoras y el lenguaje humano. El objetivo del PLN es capacitar a las máquinas para que procesen, comprendan, interpreten y generen lenguaje natural de una manera que sea valiosa y útil.  
**Referencia:** Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed. draft). [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)

### **Tokenización, Lematización y Stopwords**
Son pasos fundamentales del preprocesamiento de texto:

- **Tokenización:** Es el proceso de segmentar un texto en unidades más pequeñas llamadas "tokens", que pueden ser palabras, caracteres o sub-palabras.  
- **Lematización:** Es el proceso de reducir una palabra a su forma base o raíz, conocida como "lema". Por ejemplo, "caminando" y "caminé" se lematizan a "caminar".  
- **Stopwords:** Son palabras muy comunes en un idioma (como "el", "y", "un") que a menudo se eliminan del texto porque aportan poco significado semántico y pueden introducir ruido en el análisis.  

**Referencia:** Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.

### **Embeddings Semánticos**
Son embeddings vectoriales diseñados específicamente para capturar el significado o la semántica del texto.  
Modelos como Word2Vec o GloVe generan vectores para palabras donde la distancia y dirección entre ellos codifican relaciones semánticas (ej. `vector('Rey') - vector('Hombre') + vector('Mujer') ≈ vector('Reina')`).  
**Referencia:** Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed representations of words and phrases and their compositionality.* *Advances in Neural Information Processing Systems, 26.*

### **Modelos Transformers**
La arquitectura **Transformer** revolucionó el PLN al introducir el mecanismo de **auto-atención (*self-attention*)**.  
A diferencia de las arquitecturas recurrentes (RNN) que procesan el texto de manera secuencial, el mecanismo de atención permite al modelo **sopesar la importancia de todas las palabras en la secuencia de entrada de forma simultánea**, capturando dependencias complejas y de largo alcance entre términos, sin importar su distancia.  
**Referencia Fundacional:** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). *Attention is All You Need.* *Advances in Neural Information Processing Systems, 30.*


### **Modelos Base (BERT, RoBERTa, DistilBERT, Robertuito)**
A partir de la arquitectura Transformer, surgieron modelos de lenguaje **pre-entrenados de gran escala** que sirven como base para una multitud de tareas de PLN:

- **BERT (Bidirectional Encoder Representations from Transformers):** Aprende representaciones de texto profundamente bidireccionales, considerando el contexto tanto a la izquierda como a la derecha de cada palabra de forma simultánea, lo que permite una comprensión contextual más rica.  
- **RoBERTa (A Robustly Optimized BERT Pretraining Approach):** Es una variante que optimiza el proceso de preentrenamiento de BERT, utilizando más datos y ajustando los hiperparámetros de entrenamiento para mejorar significativamente su rendimiento.  
- **DistilBERT:** Es una versión más pequeña y rápida de BERT, creada mediante un proceso llamado *destilación de conocimiento*, que conserva gran parte de la capacidad de BERT con un tamaño considerablemente menor.  
- **Robertuito:** Es un modelo tipo RoBERTa **preentrenado desde cero sobre un corpus masivo en español**, lo que lo hace especialmente efectivo para tareas de PLN en este idioma.  


### **Modelos Específicos Utilizados en esta Investigación**

En la presente investigación, no se utilizaron los modelos base directamente, sino **versiones especializadas y previamente ajustadas (*fine-tuned*)** alojadas en la plataforma **Hugging Face**, seleccionadas por su alto rendimiento en tareas de análisis de emoción y sentimiento en inglés y español.


#### **1. Extracción de Emociones en Inglés**
- **Modelo:** `bhadresh-savani/distilbert-base-uncased-emotion`  
- **Descripción:** Modelo basado en la arquitectura DistilBERT, ajustado para clasificar texto en seis emociones: *tristeza (sadness)*, *alegría (joy)*, *amor (love)*, *ira (anger)*, *miedo (fear)* y *sorpresa (surprise)*.  
Su naturaleza ligera y eficiente fue ideal para procesar el gran corpus de canciones en inglés.  
**Referencia:** Savani, B. (2021). *DistilBERT Base Uncased Emotion.* Hugging Face. [https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)


#### **2. Extracción de Emociones en Español**
- **Modelo:** `pysentimiento/robertuito-emotion-analysis`  
- **Descripción:** Modelo basado en la arquitectura Robertuito, optimizado para el español y ajustado para clasificar emociones en un corpus de redes sociales.  
Detecta las emociones: *alegría (joy)*, *tristeza (sadness)*, *ira (anger)*, *sorpresa (surprise)*, *disgusto (disgust)* y *miedo (fear)*.  
**Referencia:** Pérez, J. C., & Furman, D. (2021). *Robertuito Emotion Analysis.* Hugging Face. [https://huggingface.co/pysentimiento/robertuito-emotion-analysis](https://huggingface.co/pysentimiento/robertuito-emotion-analysis)


#### **3. Cálculo de la Valencia Textual en Inglés**
- **Modelo:** `cardiffnlp/twitter-roberta-base-sentiment-latest`  
- **Descripción:** Modelo RoBERTa ajustado sobre un corpus masivo de tuits para análisis de sentimiento.  
Proporciona probabilidades para las clases *positivo*, *negativo* y *neutro*, utilizadas para calcular una **métrica continua de valencia** mediante la fórmula *(p_pos - p_neg)*.  
**Referencia:** Barbieri, F., et al. (2020). *Twitter RoBERTa Base for Sentiment Analysis - Latest.* Hugging Face. [https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)


#### **4. Cálculo de la Valencia Textual en Español**
- **Modelo:** `pysentimiento/robertuito-sentiment-analysis`  
- **Descripción:** Al igual que su contraparte de emociones, este modelo se basa en Robertuito y fue ajustado para la **clasificación de sentimiento (positivo, negativo, neutro)** en español.  
Permite el cálculo de la valencia textual para el corpus de canciones en este idioma.  
**Referencia:** Pérez, J. C., & Furman, D. (2021). *Robertuito Sentiment Analysis.* Hugging Face. [https://huggingface.co/pysentimiento/robertuito-sentiment-analysis](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis)

### **Modelos Pre-entrenados y Fine-tuning (Ajuste Fino)**
Un **modelo pre-entrenado** es aquel que ha sido entrenado en una tarea a gran escala con una cantidad masiva de datos (por ejemplo, toda Wikipedia).  
Este proceso le permite adquirir un **conocimiento general del lenguaje**, que luego puede reutilizarse en otras tareas.  
El **fine-tuning** o **ajuste fino** consiste en tomar ese modelo pre-entrenado y **volver a entrenarlo** en un conjunto de datos más pequeño y específico de una tarea (por ejemplo, clasificar emociones en letras de canciones), adaptando su conocimiento general al problema concreto.  

**Referencia:** Howard, J., & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification.* ArXiv preprint arXiv:1801.06146.


### **Vectorización de Texto y Similaridad Semántica**
La **vectorización** es el proceso de convertir texto en **representaciones numéricas (embeddings)** que pueden ser procesadas por algoritmos de aprendizaje automático.  
La **similaridad semántica** mide cuán cercanas en significado son dos piezas de texto.  
En el espacio vectorial, esto se evalúa comúnmente usando la **similitud del coseno**, una métrica que mide el ángulo entre dos vectores:  
- Valor **1** indica vectores idénticos (significados muy similares).  
- Valor **0** indica vectores ortogonales (sin relación semántica).  

**Referencia:** Gomaa, W. H., & Fahmy, A. A. (2013). *A Survey of Text Similarity Approaches.* *International Journal of Computer Applications, 68*(13).


### **Modelo de Lenguaje (Language Model)**
Un **modelo de lenguaje** es un sistema estadístico o neuronal que aprende a **predecir la probabilidad de una secuencia de palabras**.  
Los **modelos de lenguaje modernos a gran escala (LLMs)**, como **GPT-4**, son la base de las interfaces conversacionales actuales, ya que pueden **generar texto coherente, contextual y relevante** en respuesta a una entrada.  

**Referencia:** Brown, T., et al. (2020). *Language Models Are Few-Shot Learners.* *Advances in Neural Information Processing Systems, 33,* 1877–1901.


### **Prompt, Prompt Engineering y Contexto Conversacional**
- **Prompt:** Es la entrada de texto que se proporciona a un modelo de lenguaje para que genere una respuesta.  
- **Prompt Engineering:** Es la práctica de **diseñar, estructurar y refinar los prompts** para obtener respuestas más precisas, útiles o específicas del modelo.  
- **Contexto Conversacional:** Es la **información adicional** (como el historial de mensajes previos) incluida en el prompt para que el modelo **mantenga coherencia y continuidad** en la conversación.  

**Referencia:** White, J., Fu, Q., Hays, S., et al. (2023). *A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT.* ArXiv preprint arXiv:2302.11382.


