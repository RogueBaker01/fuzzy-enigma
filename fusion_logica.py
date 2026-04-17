"""
fusion_logica.py
──────────────────────────────────────────────────────────────────────────
Módulo de audio y fusión matemática. No tiene punto de entrada propio.
Importado por main_vision.py.

Exporta:
  - AudioWorker       : Worker asíncrono ElevenLabs (queue + threading)
  - GestorCooldown    : Anti-spam por clase de objeto
  - extraer_profundidad_roi()  : Correlación espacial YOLO → MiDaS
  - calcular_distancia_metros(): Heurística de distancia
  - posicion_en_frame()        : Clasificación lateral (izq/frente/der)
  - construir_alerta()         : Construye la frase narrada por ElevenLabs

CÓMO CALIBRAR CONSTANTE_FOCAL (antes del demo):
    1. Coloca un objeto a exactamente 1.0 metro de la cámara.
    2. Descomenta la línea [CALIB] en main_vision.py.
    3. Lee en consola el valor "profundidad_media" que imprime MiDaS.
    4. Ese número ES tu CONSTANTE_FOCAL. Pégalo abajo.
    5. Verifica: a 2 metros debe reportar ~2.0 m.
"""

import io
import os
import time
import queue
import threading
import numpy as np
import pygame
from dotenv import load_dotenv

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# Leer API key desde .env
load_dotenv()
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")

# ID de voz en tu cuenta de ElevenLabs
# Busca IDs en: https://api.elevenlabs.io/v1/voices
ELEVENLABS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

# Constante de calibración de distancia (ver instrucciones en docstring)
CONSTANTE_FOCAL = 350.0

# Cooldown entre alertas de la misma clase de objeto (ahorra créditos de API)
COOLDOWN_ALERTA_SEG = 4.0

# Área normalizada mínima para considerar un objeto relevante
UMBRAL_AREA_RELEVANTE = 0.03


# ──────────────────────────────────────────────────────────────────────────
# AudioWorker: cola asíncrona de texto → ElevenLabs → pygame
# ──────────────────────────────────────────────────────────────────────────

class AudioWorker:
    """
    Separa el TTS del CV loop completamente.
    El main nunca se bloquea esperando audio.
    """

    def __init__(self):
        self._cola: queue.Queue[str | None] = queue.Queue()
        self._activo = True

        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

        if not ELEVENLABS_API_KEY:
            print("[AudioWorker] ADVERTENCIA: ELEVENLABS_API_KEY vacía. Audio desactivado.")
            self._cliente = None
        else:
            self._cliente = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            print("[AudioWorker] Cliente ElevenLabs inicializado ✓")

        self._hilo = threading.Thread(
            target=self._loop_audio, daemon=True, name="AudioWorker"
        )
        self._hilo.start()

    def encolar(self, texto: str):
        """Encola un mensaje de voz. No bloquea al hilo que llama."""
        self._cola.put(texto)

    def detener(self):
        """Cierra el worker limpiamente con una poison pill."""
        self._activo = False
        self._cola.put(None)

    def _loop_audio(self):
        while self._activo:
            texto = self._cola.get()

            if texto is None:
                break

            print(f"[TTS] >> {texto}")

            if self._cliente is None:
                self._cola.task_done()
                continue

            try:
                # eleven_turbo_v2_5: modelo de baja latencia, ideal para demos en vivo
                audio_stream = self._cliente.text_to_speech.convert(
                    text=texto,
                    voice_id=ELEVENLABS_VOICE_ID,
                    model_id="eleven_turbo_v2_5",
                    voice_settings=VoiceSettings(
                        stability=0.4,
                        similarity_boost=0.85,
                        style=0.0,
                        use_speaker_boost=True,
                    ),
                    output_format="mp3_44100_64",
                )

                # Reproducir desde RAM, sin escribir a disco
                audio_bytes  = b"".join(audio_stream)
                buffer_audio = io.BytesIO(audio_bytes)
                pygame.mixer.music.load(buffer_audio, "mp3")
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)

            except Exception as e:
                print(f"[AudioWorker] Error ElevenLabs: {e}")

            self._cola.task_done()


# ──────────────────────────────────────────────────────────────────────────
# GestorCooldown: anti-spam por clase de objeto
# ──────────────────────────────────────────────────────────────────────────

class GestorCooldown:
    """
    Cronómetro independiente por clase. Evita:
      - Gasto excesivo de créditos de ElevenLabs.
      - Solapamiento de audio que desorienta al usuario.
    Clase -1 reservada para alertas de suelo (MiDaS).
    """

    def __init__(self, cooldown_seg: float = COOLDOWN_ALERTA_SEG):
        self._cd = cooldown_seg
        self._registro: dict[int, float] = {}

    def puede_alertar(self, clase_id: int) -> bool:
        return (time.time() - self._registro.get(clase_id, 0.0)) >= self._cd

    def registrar(self, clase_id: int):
        self._registro[clase_id] = time.time()


# Funciones matemáticas de fusión espacial YOLO + MiDaS

def extraer_profundidad_roi(
    depth_map: np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
) -> float:
    """
    Extrae la mediana de profundidad del bbox de YOLO sobre el depth map de MiDaS.
    Mediana en lugar de media: ignora bordes ruidosos y reflexiones.
    """
    h, w = depth_map.shape[:2]
    x1 = max(0, min(x1, w - 1));  x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1));  y2 = max(y1 + 1, min(y2, h))
    roi = depth_map[y1:y2, x1:x2]
    return float(np.median(roi)) if roi.size > 0 else 0.0


def calcular_distancia_metros(profundidad_media: float) -> float:
    """
    MiDaS produce disparidad inversa (mayor valor = más cercano).
    Fórmula pinhole: distancia = CONSTANTE_FOCAL / disparidad.
    Calibrar CONSTANTE_FOCAL con cinta métrica antes del demo.
    """
    return round(CONSTANTE_FOCAL / (profundidad_media + 1e-6), 1)


def posicion_en_frame(centro_x: float, ancho: int) -> str:
    t = ancho // 3
    if centro_x < t:       return "a tu izquierda"
    elif centro_x < 2 * t: return "al frente"
    else:                  return "a tu derecha"


def construir_alerta(nombre: str, distancia: float, posicion: str) -> str:
    if distancia < 1.0:    prefijo = "¡Cuidado! "
    elif distancia < 2.0:  prefijo = "Atención. "
    else:                  prefijo = ""
    return f"{prefijo}{nombre} {posicion}, a {distancia} metros"
