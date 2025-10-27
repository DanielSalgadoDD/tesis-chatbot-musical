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

