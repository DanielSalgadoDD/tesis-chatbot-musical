# Sistema de Recomendación Musical Basado en Análisis Emocional y Semántico de Letras

**Autor:** Daniel Salgado
**Trabajo de Grado - Ciencia de Datos**  
**Año:** 2025  

---

##  Descripción General

Este repositorio contiene los anexos técnicos, cuadernos y componentes implementados para el proyecto de tesis **"Sistema de recomendación musical basado en análisis emocional y semántico de letras"**.  
El trabajo propone un enfoque multimodal que integra procesamiento del lenguaje natural (PLN), análisis acústico y modelado de emociones para generar recomendaciones musicales personalizadas según el estado emocional y las preferencias del usuario.

El sistema combina **modelos preentrenados BERT** para la extracción de características líricas y emocionales, con **FAISS** como motor de búsqueda vectorial, implementado dentro de una **interfaz conversacional tipo chatbot**.

---

##   Estructura del Repositorio


---

##  Contenido de los Anexos

### **Anexo 1 – Preprocesamiento y Análisis Exploratorio (EDA)**
Incluye los cuadernos utilizados para la limpieza de datos, normalización lingüística, detección de idioma, tokenización y análisis exploratorio de la base de datos musical.  
Se describen las principales variables, distribuciones y correlaciones que fundamentan el diseño del recomendador.

### **Anexo 2 – Chatbot: Backend y Frontend**
Contiene la implementación modular del sistema conversacional.  
Incluye la lógica de flujos, integración con el modelo GPT, generación de prompts y la interfaz de usuario.  
La arquitectura se basa en una estructura de orquestación que conecta el procesamiento emocional, el motor FAISS y la capa de interacción.

### **Anexo 3 – Dependencias, Librerías y APIs**
Documenta las librerías, entornos y credenciales utilizadas (sin exponer claves privadas).  
Incluye las descripciones técnicas de las APIs externas (Spotify, Hugging Face, etc.).

### **Anexo 4 – Glosario y Conceptos Técnicos**
Reúne explicaciones breves de los principales términos utilizados en la tesis: embeddings, valencia emocional, FAISS, BERT, vector ponderado, entre otros.  
También se incluyen referencias cruzadas con el documento principal.

---

##  Requisitos y Entorno

- Python 3.10+
- Librerías principales: `transformers`, `torch`, `faiss`, `numpy`, `pandas`, `matplotlib`, `spacy`, `nltk`
- Requiere acceso a API de Spotify y modelos Hugging Face (DistilBERT, RoBERTa, Robertuito)
- Se requiere un LLM, en este caso GPT 4o a partir de tokens de OPENAI

Para instalar dependencias:
```bash
pip install -r requirements.txt
