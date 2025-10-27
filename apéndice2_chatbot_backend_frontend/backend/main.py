import asyncio
import json
import random
import re
import typing as t
from collections import defaultdict
import faiss
import numpy as np
import pandas as pd
import spotipy
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

# --- Cargar variables de entorno ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# --- Validación opcional (solo para depuración) ---
if not OPENAI_API_KEY or not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    print("⚠️ Advertencia: Faltan claves de API en el archivo .env o en las variables de entorno.")

def cargar_prompt(ruta):
    """Lee un prompt desde un archivo .txt con codificación UTF-8."""
    with open(ruta, "r", encoding="utf-8") as f:
        return f.read()

# Cargar los prompts externos
PROMPT_EXTRACCION = ChatPromptTemplate.from_template(
    cargar_prompt("prompts/prompt_extraccion.txt")
)

PROMPT_RESPUESTA_FINAL = ChatPromptTemplate.from_template(
    cargar_prompt("prompts/prompt_respuesta_final.txt")
)

# ==============================================================================
# 2. INICIALIZACIÓN DE LA APLICACIÓN Y DATOS
# ==============================================================================

app = FastAPI(title="Recomendador Musical Inteligente")

# --- Carga y preparación de datos ---
DF_PATH = "data/canciones_total.csv"
df = pd.read_csv(DF_PATH)
lista_generos_validos = df['tag'].dropna().unique().tolist()
lista_emociones_validas = df['emocion_label'].dropna().unique().tolist()
print("✅ Datos de anclaje listos.")

def parse_vector_from_string(x, dim=8):
    """Convierte una representación de string de un vector a un array de numpy."""
    if isinstance(x, str):
        try:
            clean_str = re.sub(r'\s+', ' ', x.replace('\n', '')).strip()
            vec = np.fromstring(clean_str.strip("[]"), sep=" ", dtype=np.float32)
            if vec.shape[0] == dim:
                return vec
        except Exception:
            pass
        try:
            return np.array(json.loads(x), dtype=np.float32)
        except Exception:
            return np.zeros(dim, dtype=np.float32)
    elif isinstance(x, (np.ndarray, list)):
        return np.array(x, dtype=np.float32)
    return np.zeros(dim, dtype=np.float32)

df['vector_ponderado'] = df['vector_ponderado'].apply(lambda x: parse_vector_from_string(x, dim=8))

# --- Creación del índice FAISS para búsqueda de similitud ---
X = np.vstack(df['vector_ponderado'].values).astype('float32')
faiss.normalize_L2(X)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
print("✅ DataFrame y FAISS listos.")

# --- Inicialización de Modelos y Clientes ---
historial_chat = defaultdict(list)
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.5)

try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET))
except Exception as e:
    sp = None
    print(f"⚠️ Spotify no inicializado: {e}")

# --- Configuración de Middleware (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Sistema de Caché ---
id_cache = {}
album_art_cache = {}
try:
    df_con_ids = pd.read_csv("data/canciones_filtradas.csv")
    for _, row in df_con_ids.iterrows():
        nombre_cancion, nombre_artista = row.get('name'), row.get('artists')
        if isinstance(nombre_cancion, str) and isinstance(nombre_artista, str):
            cache_key = f"{nombre_cancion.lower()}|{nombre_artista.lower()}"
            id_cache[cache_key] = row['id']
    print(f"✅ Pre-cargado el caché con {len(id_cache)} IDs de Spotify.")
except FileNotFoundError:
    print("ℹ️ No se encontró 'canciones_filtradas.csv', el caché empezará vacío.")


# ==============================================================================
# 3. HELPERS Y FUNCIONES DE UTILIDAD
# ==============================================================================

def obtener_spotify_album_art(track_id: str, sp_client: spotipy.Spotify) -> t.Optional[str]:
    """Busca la URL de la carátula de una canción y la guarda en caché."""
    if not track_id or not sp_client:
        return None
    if track_id in album_art_cache:
        return album_art_cache[track_id]
    try:
        track_info = sp_client.track(track_id)
        if track_info and track_info['album']['images']:
            image_url = track_info['album']['images'][0]['url']
            album_art_cache[track_id] = image_url
            return image_url
    except Exception as e:
        print(f"❌ Error buscando carátula: {e}")
        album_art_cache[track_id] = None
    return None

def obtener_spotify_id(nombre_cancion: str, nombre_artista: str, sp_client: spotipy.Spotify) -> t.Optional[str]:
    """Obtiene el ID de una canción en Spotify, usando caché."""
    if not sp_client or not nombre_cancion or not nombre_artista:
        return None
    primer_artista = nombre_artista.split(',')[0].strip()
    cache_key = f"{nombre_cancion.lower()}|{primer_artista.lower()}"
    if cache_key in id_cache:
        return id_cache[cache_key]
    try:
        query = f"track:{nombre_cancion} artist:{primer_artista}"
        results = sp_client.search(q=query, type='track', limit=1)
        if results and results['tracks']['items']:
            track_id = results['tracks']['items'][0]['id']
            id_cache[cache_key] = track_id
            return track_id
    except Exception as e:
        print(f"❌ Error buscando en Spotify: {e}")
        id_cache[cache_key] = None
    return None

def gestion_historial(session_id: str, autor: str, texto: str) -> str:
    """Gestiona el historial de la conversación para mantener el contexto."""
    historial = historial_chat[session_id]
    historial.append(f"{autor}: {texto}")
    if len(historial) > 4:
        historial = historial[-4:]
    historial_chat[session_id] = historial
    return "\n".join(historial)

def generar_sugerencias_contextuales(df_recs: pd.DataFrame, lista_master_generos: list, lista_master_emociones: list) -> list:
    """Genera una lista de sugerencias de continuación."""
    sugerencias = []
    if not df_recs.empty and 'tag' in df_recs.columns:
        genero_actual = df_recs['tag'].iloc[0]
        sugerencias.append(f"Más de '{genero_actual}'")
        otros_generos = [g for g in random.sample(lista_master_generos, min(len(lista_master_generos), 5)) if g != genero_actual]
        if len(otros_generos) >= 2:
            sugerencias.append(f"Probar con '{otros_generos[0]}'")
            sugerencias.append(f"Probar con '{otros_generos[1]}'")
    else:
        emociones_sugeridas = random.sample(lista_master_emociones, min(len(lista_master_emociones), 2))
        sugerencias.append(f"Sentimiento: '{emociones_sugeridas[0]}'")
        sugerencias.append(f"Sentimiento: '{emociones_sugeridas[1]}'")
    return sugerencias

def diversificar_resultados_por_artista(df_recs: pd.DataFrame, top_k: int = 5, max_por_artista: int = 2) -> pd.DataFrame:
    """Evita la repetición excesiva de un mismo artista en los resultados."""
    if df_recs.empty:
        return df_recs
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
# 4. FUNCIONES DE RECOMENDACIÓN (CORE LOGIC)
# ==============================================================================

def recomendar_por_genero(genero: t.Optional[str], df_local: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if not genero or not isinstance(genero, str):
        return pd.DataFrame()
    resultado = df_local[df_local['tag'].str.lower() == genero.lower()]
    return resultado.sample(n=min(len(resultado), top_k)) if not resultado.empty else pd.DataFrame()

def recomendar_por_artista(artista: str, df_local: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    resultado = df_local[df_local['artists'].str.contains(artista, case=False, na=False)]
    return resultado.sample(n=min(len(resultado), top_k)) if not resultado.empty else pd.DataFrame()

def recomendar_por_similitud(vector_query: np.ndarray, df_local: pd.DataFrame, idx_local: faiss.Index, top_k: int = 10, filtro_genero: str = None) -> pd.DataFrame:
    k_busqueda = top_k * 5 if filtro_genero else top_k
    vector_query = vector_query.reshape(1, -1).astype('float32')
    faiss.normalize_L2(vector_query)
    distancias, indices = idx_local.search(vector_query, k_busqueda)
    valid_indices = [i for i in indices[0] if i < len(df_local)]
    if not valid_indices:
        return pd.DataFrame()
    recs = df_local.iloc[valid_indices].copy()
    recs['similaridad'] = distancias[0][:len(valid_indices)]
    if filtro_genero:
        boost = recs['tag'].str.lower() == filtro_genero.lower()
        recs['similaridad'] += boost * 0.2
        recs = recs.sort_values(by='similaridad', ascending=False)
    return recs.head(top_k)

def recomendar_por_emocion(emocion: str, df_local: pd.DataFrame, idx_local: faiss.Index, top_k: int = 5, filtro_genero: str = None) -> pd.DataFrame:
    df_emocion = df_local[df_local['emocion_label'].str.contains(emocion, case=False, na=False)]
    if df_emocion.empty:
        return pd.DataFrame()
    vector_promedio_emocion = np.mean(np.vstack(df_emocion['vector_ponderado'].values), axis=0)
    return recomendar_por_similitud(vector_promedio_emocion, df_local, idx_local, top_k, filtro_genero)

def recomendar_por_historial(canciones: list, df_local: pd.DataFrame, idx_local: faiss.Index, top_k: int = 5, filtro_genero: str = None) -> pd.DataFrame:
    vectores_usuario = []
    for cancion_item in canciones:
        nombre_cancion, nombre_artista = "", ""
        if isinstance(cancion_item, dict):
            nombre_cancion, nombre_artista = cancion_item.get("name", "").lower(), cancion_item.get("artist", "")
        elif isinstance(cancion_item, str):
            nombre_cancion = cancion_item.lower()
            for sep in [" de ", " - "]:
                if sep in nombre_cancion:
                    partes = nombre_cancion.split(sep)
                    nombre_cancion, nombre_artista = partes[0].strip(), partes[1].strip()
                    break
        if not nombre_cancion:
            continue
        mask = (df_local["name"].str.lower() == nombre_cancion)
        if nombre_artista:
            mask &= (df_local["artists"].str.lower().str.contains(nombre_artista, case=False, na=False))
        fila = df_local[mask]
        if not fila.empty:
            vectores_usuario.append(fila.iloc[0]['vector_ponderado'])
    if not vectores_usuario:
        return pd.DataFrame()
    vector_promedio_usuario = np.mean(vectores_usuario, axis=0)
    return recomendar_por_similitud(vector_promedio_usuario, df_local, idx_local, top_k, filtro_genero)

# ==============================================================================
# 6. LÓGICA CENTRAL DEL ENDPOINT Y FLUJO PRINCIPAL
# ==============================================================================

def ejecutar_flujo_principal(intencion: dict, df_completo: pd.DataFrame, idx_completo: faiss.Index) -> pd.DataFrame:
    """Ejecuta la lógica de recomendación basada en la intención extraída."""
    modo = intencion.get('modo_principal')
    params = intencion.get('parametros_filtrado', {})
    emocion_ctx = intencion.get('contexto_emocional', {})
    entidades = intencion.get('entidades_extraidas', {})

    idioma = params.get('idioma')
    genero = params.get('genero')
    top_k = 5

    df_filtrado = df_completo.copy()
    if idioma:
        df_filtrado = df_filtrado[df_filtrado['language'] == idioma]
        if df_filtrado.empty:
            return pd.DataFrame()

    idx_filtrado = idx_completo
    if idioma:
        df_indices = df_filtrado.index.to_numpy()
        if len(df_indices) > 0:
            temp_index = faiss.IndexFlatIP(X.shape[1])
            temp_index.add(X[df_indices])
            idx_filtrado = temp_index
        else:
            return pd.DataFrame()

    recomendaciones_crudas = pd.DataFrame()
    k_busqueda_ampliada = top_k * 5

    if modo == 'emocion':
        emocion = emocion_ctx.get('emocion_detectada')
        if emocion_ctx.get('intencion_emocional') == 'calmar':
            mapa_opuestos = {"tristeza": "alegría", "frustración": "esperanza frágil", "ira": "amor"}
            emocion = mapa_opuestos.get(emocion, emocion)
        recomendaciones_crudas = recomendar_por_emocion(emocion, df_filtrado, idx_filtrado, top_k=k_busqueda_ampliada, filtro_genero=genero)
        return diversificar_resultados_por_artista(recomendaciones_crudas, top_k=top_k, max_por_artista=2)

    elif modo == 'historial_canciones':
        recomendaciones_crudas = recomendar_por_historial(entidades.get('canciones'), df_filtrado, idx_filtrado, top_k=k_busqueda_ampliada, filtro_genero=genero)
        return diversificar_resultados_por_artista(recomendaciones_crudas, top_k=top_k, max_por_artista=2)

    elif modo == 'directo_artista':
        return recomendar_por_artista(entidades.get('artista_principal'), df_filtrado, top_k=top_k)

    elif modo == 'directo_genero':
        genero_principal = entidades.get('genero_principal')
        if not genero_principal:
            return pd.DataFrame()
        return recomendar_por_genero(genero_principal, df_filtrado, top_k=top_k)

    return pd.DataFrame()

# ==============================================================================
# 7. ENDPOINTS DE LA API
# ==============================================================================

@app.get("/")
def raiz():
    """Sirve la página principal del frontend."""
    return FileResponse("frontend/index.html")

# === FUNCIÓN DE STREAMING CORREGIDA ===
async def stream_generator(input_usuario: str, session_id: str):
    """Genera la respuesta en trozos: `data:` para datos de tarjetas y `text:` para el mensaje del chat."""
    historial_formateado = gestion_historial(session_id, "Usuario", input_usuario)

    cadena_extraccion = PROMPT_EXTRACCION | llm | JsonOutputParser()
    intencion = await cadena_extraccion.ainvoke({
        "lista_generos_validos": lista_generos_validos,
        "lista_emociones_validas": lista_emociones_validas,
        "historial_chat": historial_formateado,
        "input_usuario": input_usuario
    })

    if "español" in input_usuario.lower() and intencion.get("parametros_filtrado", {}).get("idioma") is None:
        if "parametros_filtrado" not in intencion:
            intencion["parametros_filtrado"] = {}
        intencion["parametros_filtrado"]["idioma"] = "es"

    if intencion.get('modo_principal') in ['conversacional', None]:
        recomendaciones_df = pd.DataFrame()
        mensaje_conversacional = intencion.get('respuesta_conversacional', "¡Hola! ¿En qué puedo ayudarte hoy?")
    else:
        recomendaciones_df = ejecutar_flujo_principal(intencion, df, index)
        mensaje_conversacional = None

    recs_records_para_frontend = []
    if not recomendaciones_df.empty:
        df_limpio = recomendaciones_df.copy().fillna('')
        for _, row in df_limpio.iterrows():
            spotify_id = obtener_spotify_id(row.get("name", ""), row.get("artists", ""), sp)
            album_art_url = obtener_spotify_album_art(spotify_id, sp)
            recs_records_para_frontend.append({
                "name": str(row.get("name", "")), "artists": str(row.get("artists", "")),
                "similaridad": float(row.get("similaridad", 1.0)), "spotify_id": spotify_id,
                "album_art_url": album_art_url, "tag": str(row.get("tag", "")),
                "emocion_label": str(row.get("emocion_label", ""))
            })

    sugerencias = generar_sugerencias_contextuales(recomendaciones_df, lista_generos_validos, lista_emociones_validas)
    
    # Enviar el chunk de datos para las tarjetas y botones
    initial_data = {"recomendaciones": recs_records_para_frontend, "sugerencias": sugerencias}
    yield f"data: {json.dumps(initial_data)}\n\n"
    
    if not recs_records_para_frontend and not mensaje_conversacional:
        mensaje_conversacional = f"No encontré recomendaciones para tu petición. ¿Quieres que intentemos con otra cosa? Podemos probar con {sugerencias[0]} o {sugerencias[1]}."

    # Preparar el contexto y generar el texto del chat
    cadena_respuesta_stream = PROMPT_RESPUESTA_FINAL | llm
    full_response_text = ""
    sugerencias_texto = ", ".join(s.split(': ')[-1].strip("'") for s in sugerencias[:2])

    async for chunk in cadena_respuesta_stream.astream({
        "modo_ejecutado": intencion.get('modo_principal', 'tu petición'),
        "num_recomendaciones": len(recs_records_para_frontend),
        "mensaje_conversacional": mensaje_conversacional,
        "sugerencias_contextuales": sugerencias_texto
    }):
        if chunk.content:
            full_response_text += chunk.content
            escaped_content = json.dumps(chunk.content)
            yield f"text: {escaped_content}\n\n"

    gestion_historial(session_id, "Bot", full_response_text)


@app.get("/chat")
async def chat(input_usuario: str, session_id: str = Query("default_session")):
    """Endpoint principal que maneja las solicitudes de chat y devuelve un stream."""
    return StreamingResponse(stream_generator(input_usuario, session_id), media_type="text/event-stream")