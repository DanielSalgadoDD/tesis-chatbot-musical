import asyncio
import json
import random
import re
import typing as t
from collections import defaultdict
import os

# Librerías de datos y API
import faiss
import numpy as np
import pandas as pd
import spotipy
import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# ==============================================================================
# 1. CONFIGURACIÓN Y VARIABLES DE ENTORNO
# ==============================================================================

# Cargar variables del archivo .env
load_dotenv()

# Recuperar claves de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Configuración del Modelo (OpenAI GPT-4o)
LLM_API_URL = "https://api.openai.com/v1/chat/completions"
LLM_MODEL = "gpt-4o" 

# Validaciones de seguridad
if not OPENAI_API_KEY:
    print("⚠️  ERROR CRÍTICO: Falta OPENAI_API_KEY en el archivo .env")
if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    print("⚠️  Advertencia: Faltan claves de Spotify en el archivo .env")

app = FastAPI(title="Recomendador Musical Inteligente")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def cargar_prompt(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠️ No se encontró el prompt en: {ruta}")
        return ""

RAW_PROMPT_EXTRACCION = cargar_prompt("prompts/prompt_extraccion.txt")
RAW_PROMPT_RESPUESTA_FINAL = cargar_prompt("prompts/prompt_respuesta_final.txt")

# ==============================================================================
# 2. CARGA DE DATOS Y FAISS (SIN CLUSTERS DE ARTISTAS)
# ==============================================================================

DF_PATH = "data/canciones_total.csv"

try:
    # 1. Carga del DataFrame Principal
    df = pd.read_csv(DF_PATH)
    
    # Normalizamos nombres de canciones para facilitar el filtrado posterior
    df['name_norm'] = df['name'].astype(str).str.lower().str.strip()
    
    lista_generos_validos = [str(g).strip() for g in df['tag'].dropna().unique().tolist()]
    lista_emociones_validas = df['emocion_label'].dropna().unique().tolist()
    print("✅ Datos de canciones cargados.")

except Exception as e:
    print(f"❌ Error crítico cargando CSV: {e}")
    df = pd.DataFrame()
    lista_generos_validos = []
    lista_emociones_validas = []

def parse_vector_from_string(x, dim=8):
    if isinstance(x, str):
        try:
            clean_str = re.sub(r'\s+', ' ', x.replace('\n', '')).strip()
            vec = np.fromstring(clean_str.strip("[]"), sep=" ", dtype=np.float32)
            if vec.shape[0] == dim: return vec
        except: pass
        try: return np.array(json.loads(x), dtype=np.float32)
        except: return np.zeros(dim, dtype=np.float32)
    elif isinstance(x, (np.ndarray, list)):
        return np.array(x, dtype=np.float32)
    return np.zeros(dim, dtype=np.float32)

# Preparar FAISS
if not df.empty:
    df['vector_ponderado'] = df['vector_ponderado'].apply(lambda x: parse_vector_from_string(x, dim=8))
    X = np.vstack(df['vector_ponderado'].values).astype('float32')
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    print("✅ DataFrame y FAISS listos.")
else:
    index = None
    print("⚠️ DataFrame vacío, FAISS no inicializado.")

# Inicialización Spotify
try:
    if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET))
    else:
        sp = None
except Exception as e:
    sp = None
    print(f"⚠️ Spotify no inicializado: {e}")

# Caché
historial_chat = defaultdict(list)
id_cache = {}
album_art_cache = {}

try:
    df_con_ids = pd.read_csv("data/canciones_filtradas.csv")
    for _, row in df_con_ids.iterrows():
        nombre_cancion, nombre_artista = row.get('name'), row.get('artists')
        if isinstance(nombre_cancion, str) and isinstance(nombre_artista, str):
            cache_key = f"{nombre_cancion.lower()}|{nombre_artista.lower()}"
            id_cache[cache_key] = row['id']
    print(f"✅ Caché de IDs cargado: {len(id_cache)} entradas.")
except FileNotFoundError:
    print("ℹ️ Caché vacío.")

# ==============================================================================
# 3. FUNCIONES AUXILIARES
# ==============================================================================

async def llamar_llm(messages: list, temperature: float = 0.1) -> str:
    """Función genérica para llamar a OpenAI GPT-4o."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(LLM_API_URL, headers=headers, json=payload)
            if response.status_code != 200:
                print(f"❌ Error API LLM ({response.status_code}): {response.text}")
                return ""
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"❌ Error de conexión LLM: {e}")
            return ""

def limpiar_json_llm(texto_respuesta: str) -> dict:
    if not texto_respuesta: return {}
    texto_limpio = re.sub(r'```json\s*', '', texto_respuesta)
    texto_limpio = re.sub(r'```\s*', '', texto_limpio).strip()
    try:
        return json.loads(texto_limpio)
    except:
        try:
            inicio = texto_limpio.find('{')
            fin = texto_limpio.rfind('}') + 1
            if inicio != -1 and fin != -1:
                return json.loads(texto_limpio[inicio:fin])
        except: pass
        return {}

def obtener_spotify_album_art(track_id: str, sp_client: spotipy.Spotify) -> t.Optional[str]:
    if not track_id or not sp_client: return None
    if track_id in album_art_cache: return album_art_cache[track_id]
    try:
        track_info = sp_client.track(track_id)
        if track_info and track_info['album']['images']:
            image_url = track_info['album']['images'][0]['url']
            album_art_cache[track_id] = image_url
            return image_url
    except:
        album_art_cache[track_id] = None
    return None

def obtener_spotify_id(nombre_cancion: str, nombre_artista: str, sp_client: spotipy.Spotify) -> t.Optional[str]:
    if not sp_client or not nombre_cancion or not nombre_artista: return None
    primer_artista = nombre_artista.split(',')[0].strip()
    cache_key = f"{nombre_cancion.lower()}|{primer_artista.lower()}"
    if cache_key in id_cache: return id_cache[cache_key]
    try:
        query = f"track:{nombre_cancion} artist:{primer_artista}"
        results = sp_client.search(q=query, type='track', limit=1)
        if results and results['tracks']['items']:
            track_id = results['tracks']['items'][0]['id']
            id_cache[cache_key] = track_id
            return track_id
    except:
        id_cache[cache_key] = None
    return None

def gestion_historial(session_id: str, autor: str, texto: str) -> str:
    historial = historial_chat[session_id]
    historial.append(f"{autor}: {texto}")
    if len(historial) > 6:
        historial = historial[-6:]
    historial_chat[session_id] = historial
    return "\n".join(historial)

def generar_sugerencias_contextuales(df_recs: pd.DataFrame, lista_master_generos: list, lista_master_emociones: list) -> list:
    sugerencias = []
    if not df_recs.empty and 'tag' in df_recs.columns:
        genero_actual = df_recs['tag'].iloc[0]
        sugerencias.append(f"Más de '{genero_actual}'")
        otros_generos = [g for g in random.sample(lista_master_generos, min(len(lista_master_generos), 5)) if g != genero_actual]
        if len(otros_generos) >= 2:
            sugerencias.append(f"Probar con '{otros_generos[0]}'")
            sugerencias.append(f"Probar con '{otros_generos[1]}'")
    else:
        if len(lista_master_emociones) >= 2:
            emociones_sugeridas = random.sample(lista_master_emociones, 2)
            sugerencias.append(f"Sentimiento: '{emociones_sugeridas[0]}'")
            sugerencias.append(f"Sentimiento: '{emociones_sugeridas[1]}'")
    return sugerencias

def diversificar_resultados_por_artista(df_recs: pd.DataFrame, top_k: int = 5, max_por_artista: int = 2) -> pd.DataFrame:
    if df_recs.empty: return df_recs
    artistas_contados = defaultdict(int)
    indices_a_mantener = []
    for index, row in df_recs.iterrows():
        artista_principal = row['artists'].split(',')[0].strip()
        if artistas_contados[artista_principal] < max_por_artista:
            indices_a_mantener.append(index)
            artistas_contados[artista_principal] += 1
        if len(indices_a_mantener) >= top_k:
            break
    return df_recs.loc[indices_a_mantener]

# ==============================================================================
# 4. LÓGICA DE RECOMENDACIÓN (SIMPLIFICADA SIN CLUSTERS)
# ==============================================================================

def recomendar_por_genero(genero: t.Optional[str], df_local: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if not genero or not isinstance(genero, str): return pd.DataFrame()
    resultado = df_local[df_local['tag'].str.lower() == genero.lower()]
    return resultado.sample(n=min(len(resultado), top_k)) if not resultado.empty else pd.DataFrame()

def recomendar_por_artista(artista: str, df_local: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    resultado = df_local[df_local['artists'].str.contains(artista, case=False, na=False)]
    return resultado.sample(n=min(len(resultado), top_k)) if not resultado.empty else pd.DataFrame()

def recomendar_por_similitud(vector_query: np.ndarray, df_local: pd.DataFrame, idx_local: faiss.Index, top_k: int = 10, filtro_genero: str = None) -> pd.DataFrame:
    if idx_local is None: return pd.DataFrame()
    k_busqueda = top_k * 5 if filtro_genero else top_k
    vector_query = vector_query.reshape(1, -1).astype('float32')
    faiss.normalize_L2(vector_query)
    distancias, indices = idx_local.search(vector_query, k_busqueda)
    valid_indices = [i for i in indices[0] if i < len(df_local)]
    if not valid_indices: return pd.DataFrame()
    recs = df_local.iloc[valid_indices].copy()
    recs['similaridad'] = distancias[0][:len(valid_indices)]
    if filtro_genero:
        boost = recs['tag'].str.lower() == filtro_genero.lower()
        recs['similaridad'] += boost * 0.2
        recs = recs.sort_values(by='similaridad', ascending=False)
    return recs.head(top_k)

def recomendar_por_emocion(emocion: str, df_local: pd.DataFrame, idx_local: faiss.Index, top_k: int = 5, filtro_genero: str = None) -> pd.DataFrame:
    df_emocion = df_local[df_local['emocion_label'].str.contains(emocion, case=False, na=False)]
    if df_emocion.empty: return pd.DataFrame()
    vector_promedio_emocion = np.mean(np.vstack(df_emocion['vector_ponderado'].values), axis=0)
    return recomendar_por_similitud(vector_promedio_emocion, df_local, idx_local, top_k, filtro_genero)

def recomendar_por_historial(canciones: list, df_local: pd.DataFrame, idx_local: faiss.Index, top_k: int = 5, filtro_genero: str = None, modo_estricto: bool = False) -> pd.DataFrame:
    """
    Recomendador híbrido con exclusión de la canción original.
    - Modo Estricto: Filtra solo por GÉNERO exacto de la semilla.
    - Modo Descubrimiento: Busca en todo el catálogo.
    """
    vectores_usuario = []
    semilla_genero = None
    nombres_a_excluir = set()
    
    # 1. Obtener vectores y datos semilla de las canciones input
    for cancion_item in canciones:
        nombre_cancion, nombre_artista = "", ""
        if isinstance(cancion_item, dict):
            nombre_cancion, nombre_artista = cancion_item.get("name", "").lower(), cancion_item.get("artist", "")
        elif isinstance(cancion_item, str):
            nombre_cancion = cancion_item.lower()
            if " - " in nombre_cancion:
                nombre_cancion = nombre_cancion.split(" - ")[0].strip()

        if not nombre_cancion: continue
        
        # Guardamos el nombre normalizado para excluirlo después
        nombres_a_excluir.add(nombre_cancion.strip().lower())
        
        # Buscar en el DF
        mask = (df_local["name_norm"] == nombre_cancion.strip().lower())
        if nombre_artista:
            mask &= (df_local["artists"].str.lower().str.contains(nombre_artista.lower(), na=False))
        
        fila = df_local[mask]
        if not fila.empty:
            fila_data = fila.iloc[0]
            vectores_usuario.append(fila_data['vector_ponderado'])
            
            # Guardamos datos de la primera canción encontrada para usar de semilla estricta
            if semilla_genero is None:
                semilla_genero = fila_data.get('tag', '')

    if not vectores_usuario:
        return pd.DataFrame()

    vector_promedio_usuario = np.mean(vectores_usuario, axis=0).astype('float32')
    vector_query = vector_promedio_usuario.reshape(1, -1)
    faiss.normalize_L2(vector_query)

    recomendaciones_finales = pd.DataFrame()

    # ---------------------------------------------------------
    # RAMA A: MODO ESTRICTO (Filtro Duro por Género)
    # ---------------------------------------------------------
    if modo_estricto and semilla_genero:
        # Filtro: Solo canciones del mismo género exacto
        df_subset = df_local[df_local['tag'] == semilla_genero].copy()
        
        if not df_subset.empty:
            # --- FILTRO DE EXCLUSIÓN DE LA PROPIA CANCIÓN ---
            df_subset = df_subset[~df_subset['name_norm'].isin(nombres_a_excluir)]
            
            if not df_subset.empty:
                subset_matrix = np.vstack(df_subset['vector_ponderado'].values)
                faiss.normalize_L2(subset_matrix)
                
                # Cálculo manual de similitud
                scores = np.dot(vector_query, subset_matrix.T).flatten()
                df_subset['similaridad'] = scores
                
                recomendaciones_finales = df_subset.sort_values(by='similaridad', ascending=False).head(top_k)

    # ---------------------------------------------------------
    # RAMA B: MODO DISCOVERY / FALLBACK (Relleno)
    # ---------------------------------------------------------
    faltantes = top_k - len(recomendaciones_finales)
    
    if faltantes > 0:
        # Buscamos margen extra para borrar duplicados
        k_search = (top_k + len(nombres_a_excluir)) * 3
        distancias, indices = idx_local.search(vector_query, k_search)
        valid_indices = [i for i in indices[0] if i < len(df_local)]
        
        if valid_indices:
            recs_globales = df_local.iloc[valid_indices].copy()
            recs_globales['similaridad'] = distancias[0][:len(valid_indices)]
            
            # --- FILTRO DE EXCLUSIÓN ---
            recs_globales = recs_globales[~recs_globales['name_norm'].isin(nombres_a_excluir)]
            
            if filtro_genero:
                boost = recs_globales['tag'].str.lower() == filtro_genero.lower()
                recs_globales['similaridad'] += boost * 0.2
                recs_globales = recs_globales.sort_values(by='similaridad', ascending=False)
            
            # Eliminamos las que ya estaban en recomendaciones_finales
            if not recomendaciones_finales.empty:
                ids_ya_recomendados = recomendaciones_finales.index.tolist()
                recs_globales = recs_globales[~recs_globales.index.isin(ids_ya_recomendados)]
            
            recomendaciones_finales = pd.concat([recomendaciones_finales, recs_globales.head(faltantes)])

    return recomendaciones_finales

def ejecutar_flujo_principal(intencion: dict, df_completo: pd.DataFrame, idx_completo: faiss.Index, modo_estricto_global: bool = False) -> pd.DataFrame:
    modo = intencion.get('modo_principal')
    params = intencion.get('parametros_filtrado', {}) or {}
    emocion_ctx = intencion.get('contexto_emocional', {}) or {}
    entidades = intencion.get('entidades_extraidas', {}) or {}

    idioma = params.get('idioma')
    genero = params.get('genero')
    top_k = 5

    df_filtrado = df_completo.copy()
    if idioma and 'language' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['language'] == idioma]
        if df_filtrado.empty: return pd.DataFrame()

    # Reconstrucción de índice si se filtró (simplificado)
    idx_filtrado = idx_completo 
    if idioma and 'language' in df_filtrado.columns:
        df_indices = df_filtrado.index.to_numpy()
        if len(df_indices) > 0:
            temp_index = faiss.IndexFlatIP(X.shape[1])
            temp_index.add(X[df_indices])
            idx_filtrado = temp_index
        else:
            return pd.DataFrame()

    k_busqueda_ampliada = top_k * 5

    if modo == 'emocion':
        emocion = emocion_ctx.get('emocion_detectada')
        if emocion_ctx.get('intencion_emocional') == 'calmar':
            mapa_opuestos = {"tristeza": "alegría", "frustración": "esperanza frágil", "ira": "amor"}
            emocion = mapa_opuestos.get(emocion, emocion)
        recomendaciones_crudas = recomendar_por_emocion(emocion, df_filtrado, idx_filtrado, top_k=k_busqueda_ampliada, filtro_genero=genero)
        return diversificar_resultados_por_artista(recomendaciones_crudas, top_k=top_k, max_por_artista=2)

    elif modo == 'historial_canciones':
        canciones_lista = entidades.get('canciones', [])
        if not canciones_lista: return pd.DataFrame()
        # --- AQUÍ PASAMOS EL MODO ESTRICTO ---
        recomendaciones_crudas = recomendar_por_historial(
            canciones_lista, 
            df_filtrado, 
            idx_filtrado, 
            top_k=top_k, 
            filtro_genero=genero,
            modo_estricto=modo_estricto_global
        )
        return diversificar_resultados_por_artista(recomendaciones_crudas, top_k=top_k, max_por_artista=2)

    elif modo == 'directo_artista':
        return recomendar_por_artista(entidades.get('artista_principal'), df_filtrado, top_k=top_k)

    elif modo == 'directo_genero':
        genero_principal = entidades.get('genero_principal')
        if not genero_principal: return pd.DataFrame()
        return recomendar_por_genero(genero_principal, df_filtrado, top_k=top_k)

    return pd.DataFrame()

# ==============================================================================
# VARIABLES Y FUNCIONES DE SEGURIDAD (RIESGO EMOCIONAL)
# ==============================================================================

emociones_riesgosas = {
    "tristeza", "ira", "miedo", "ansiedad", "desesperanza", "alarma",
    "resentimiento", "desesperación furiosa", "pánico violento",
    "ansiedad exaltada", "culpa"
}

emocion_pendiente_confirmacion = {}

def es_emocion_riesgosa(etiqueta_emocion: str) -> bool:
    if not etiqueta_emocion: return False
    return etiqueta_emocion.lower() in emociones_riesgosas

def buscar_emocion_en_db(nombre_cancion: str, nombre_artista: str, df_data: pd.DataFrame) -> t.Optional[str]:
    if not nombre_cancion: return None
    nombre_clean = nombre_cancion.lower().strip()
    matches = df_data[df_data['name'].str.lower() == nombre_clean]
    if matches.empty: return None
    if nombre_artista:
        artista_clean = nombre_artista.lower().strip()
        matches_artista = matches[matches['artists'].str.lower().str.contains(artista_clean, na=False)]
        if not matches_artista.empty: matches = matches_artista
    return matches.iloc[0]['emocion_label']

# ==============================================================================
# 5. STREAMING Y ENDPOINT PRINCIPAL
# ==============================================================================

async def stream_generator(input_usuario: str, session_id: str, modo_estricto: bool):
    historial_formateado = gestion_historial(session_id, "Usuario", input_usuario)
    intencion = None
    
    # --- FASE 1: SEGURIDAD / CONFIRMACIÓN ---
    if session_id in emocion_pendiente_confirmacion:
        input_clean = input_usuario.strip().lower()
        palabras_confirmacion = ["si", "sí", "seguro", "estoy seguro", "adelante", "ok", "continuar", "confirmar", "dale", "yes"]
        
        if any(p in input_clean for p in palabras_confirmacion):
            datos_pendientes = emocion_pendiente_confirmacion.pop(session_id)
            
            # CASO A: Canción riesgosa
            if isinstance(datos_pendientes, dict) and datos_pendientes.get('tipo') == 'cancion':
                emocion_aviso = datos_pendientes['emocion']
                canciones_originales = datos_pendientes['datos']
                
                intencion = {
                    "modo_principal": "historial_canciones",
                    "parametros_filtrado": {"idioma": None, "genero": None},
                    "contexto_emocional": {"emocion_detectada": emocion_aviso, "intencion_emocional": "explorar"},
                    "entidades_extraidas": {"canciones": canciones_originales, "artista_principal": None, "genero_principal": None},
                    "respuesta_conversacional": None,
                    "_ya_confirmado": True,
                    "_bypass_filtro_idioma": True
                }
                print(f"⚠️ RIESGO CONFIRMADO (Canción): {canciones_originales}")

            # CASO B: Emoción explícita
            else:
                emocion_guardada = datos_pendientes if isinstance(datos_pendientes, str) else datos_pendientes['emocion']
                intencion = {
                    "modo_principal": "emocion",
                    "parametros_filtrado": {"idioma": "es", "genero": None},
                    "contexto_emocional": {"emocion_detectada": emocion_guardada, "intencion_emocional": "explorar"},
                    "entidades_extraidas": {"canciones": [], "artista_principal": None, "genero_principal": None},
                    "respuesta_conversacional": None,
                    "_ya_confirmado": True
                }
                print(f"⚠️ RIESGO CONFIRMADO (Emoción): {emocion_guardada}")
        else:
            del emocion_pendiente_confirmacion[session_id]
            intencion = None 

    # --- FASE 2: EXTRACCIÓN CON GPT-4o ---
    if intencion is None:
        print(f"\n--- PROCESANDO: '{input_usuario}' (Estricto: {modo_estricto}) ---")
        prompt_extraccion = RAW_PROMPT_EXTRACCION.replace("{lista_generos_validos}", json.dumps(lista_generos_validos, ensure_ascii=False))
        prompt_extraccion = prompt_extraccion.replace("{lista_emociones_validas}", json.dumps(lista_emociones_validas, ensure_ascii=False))
        prompt_extraccion = prompt_extraccion.replace("{historial_chat}", historial_formateado)
        prompt_extraccion = prompt_extraccion.replace("{input_usuario}", input_usuario)

        messages_extraccion = [
            {"role": "system", "content": "Eres un motor de extracción de datos backend API. Tu salida DEBE ser EXCLUSIVAMENTE un objeto JSON minificado."},
            {"role": "user", "content": prompt_extraccion}
        ]
        
        # Llamada genérica al LLM (OpenAI)
        json_str = await llamar_llm(messages_extraccion, temperature=0.0)
        intencion = limpiar_json_llm(json_str)
        
        if not intencion:
            intencion = {"modo_principal": "conversacional", "respuesta_conversacional": "Lo siento, no pude procesar tu solicitud correctamente."}

        # --- FASE 3: DETECCIÓN DE RIESGO (POST-LLM) ---
        if not intencion.get("_ya_confirmado"):
            emocion_riesgosa_detectada = None
            tipo_riesgo = None
            
            if intencion.get("modo_principal") == "emocion":
                emo = intencion.get("contexto_emocional", {}).get("emocion_detectada")
                if es_emocion_riesgosa(emo):
                    emocion_riesgosa_detectada = emo
                    tipo_riesgo = 'directo'

            elif intencion.get("modo_principal") == "historial_canciones":
                lista_canciones = intencion.get("entidades_extraidas", {}).get("canciones", [])
                for cancion in lista_canciones:
                    emo_cancion = buscar_emocion_en_db(cancion.get("name"), cancion.get("artist"), df)
                    if es_emocion_riesgosa(emo_cancion):
                        emocion_riesgosa_detectada = emo_cancion
                        tipo_riesgo = 'cancion'
                        break

            if emocion_riesgosa_detectada:
                emocion_pendiente_confirmacion[session_id] = {
                    'tipo': tipo_riesgo,
                    'datos': intencion.get("entidades_extraidas", {}).get("canciones", []) if tipo_riesgo == 'cancion' else None,
                    'emocion': emocion_riesgosa_detectada
                }
                
                mensaje_alerta = (
                    f"Entiendo que buscas conectar con la **{emocion_riesgosa_detectada}**. "
                    "Sin embargo, es importante cuidarte: sumergirse demasiado en estos estados a veces puede intensificar el malestar "
                    "o incluso llevar a pensamientos autodestructivos.\n\n"
                    "Como recomendación personal, quizás sería positivo buscar algo que ayude a calmar ese sentimiento. "
                    "Dicho esto, si sientes que necesitas explorar esta emoción ahora, estoy aquí para acompañarte. "
                    "¿Estás seguro de que deseas continuar? (Escribe 'Sí' para confirmar)."
                )
                yield f"text: {json.dumps(mensaje_alerta)}\n\n"
                return

    # --- FASE 4: EJECUCIÓN ---
    bypass_idioma = intencion.get("_bypass_filtro_idioma", False)
    if not bypass_idioma:
        if "español" in input_usuario.lower() or intencion.get("parametros_filtrado", {}).get("idioma") is None:
             if not intencion.get("parametros_filtrado"): intencion["parametros_filtrado"] = {}
             intencion["parametros_filtrado"]["idioma"] = "es"

    if intencion.get('modo_principal') in ['conversacional', None]:
        recomendaciones_df = pd.DataFrame()
        mensaje_conversacional = intencion.get('respuesta_conversacional', "¡Hola! ¿En qué puedo ayudarte hoy?")
    else:
        recomendaciones_df = ejecutar_flujo_principal(intencion, df, index, modo_estricto_global=modo_estricto)
        mensaje_conversacional = None

    # --- FASE 5: DATA FRONTEND ---
    recs_records_para_frontend = []
    if not recomendaciones_df.empty:
        df_limpio = recomendaciones_df.copy().fillna('')
        for _, row in df_limpio.iterrows():
            spotify_id = obtener_spotify_id(row.get("name", ""), row.get("artists", ""), sp)
            recs_records_para_frontend.append({
                "name": str(row.get("name", "")), 
                "artists": str(row.get("artists", "")),
                "similaridad": float(row.get("similaridad", 1.0)), 
                "spotify_id": spotify_id,
                "album_art_url": obtener_spotify_album_art(spotify_id, sp), 
                "tag": str(row.get("tag", "")),
                "emocion_label": str(row.get("emocion_label", ""))
            })

    sugerencias = generar_sugerencias_contextuales(recomendaciones_df, lista_generos_validos, lista_emociones_validas)
    initial_data = {"recomendaciones": recs_records_para_frontend, "sugerencias": sugerencias}
    yield f"data: {json.dumps(initial_data)}\n\n"
    
    if not recs_records_para_frontend and not mensaje_conversacional:
        mensaje_conversacional = f"Entendí que buscas '{intencion.get('modo_principal')}', pero no encontré coincidencias exactas."

    # --- FASE 6: CHAT ---
    prompt_final = RAW_PROMPT_RESPUESTA_FINAL.replace("{modo_ejecutado}", str(intencion.get('modo_principal', 'tu petición')))
    prompt_final = prompt_final.replace("{num_recomendaciones}", str(len(recs_records_para_frontend)))
    prompt_final = prompt_final.replace("{mensaje_conversacional}", str(mensaje_conversacional if mensaje_conversacional else ""))
    prompt_final = prompt_final.replace("{sugerencias_contextuales}", ", ".join(s.split(': ')[-1].strip("'") for s in sugerencias[:2]))

    messages_final = [
        {"role": "system", "content": "Eres un DJ experto y empático. Responde brevemente."},
        {"role": "user", "content": prompt_final}
    ]
    
    texto_final = await llamar_llm(messages_final, temperature=0.6)
    if texto_final:
        yield f"text: {json.dumps(texto_final)}\n\n"
        gestion_historial(session_id, "Bot", texto_final)
    else:
        yield f"text: {json.dumps('Error generando respuesta.')}\n\n"

@app.get("/")
def raiz():
    return FileResponse("frontend/index.html")

@app.get("/chat")
async def chat(input_usuario: str, session_id: str = Query("default_session"), modo_estricto: bool = Query(False)):
    return StreamingResponse(stream_generator(input_usuario, session_id, modo_estricto), media_type="text/event-stream")
