"""
Módulo de audio y fusión matemática. No tiene punto de entrada propio.
Importado por main_vision.py.

Exporta:
  - AudioWorker            : Worker asíncrono local MP3 → pygame (queue + threading)
  - GestorCooldown         : Anti-spam por clase de objeto
  - extraer_profundidad_roi()    : Correlación espacial YOLO → MiDaS
  - calcular_distancia_metros()  : Heurística de distancia
  - posicion_en_frame()          : Clasificación lateral (izq/frente/der)
  - construir_alerta()           : Frase de debug para overlay de OpenCV
  - obtener_archivo_audio()      : Enrutador con Binning de distancia → AudioWorker

CÓMO CALIBRAR CONSTANTE_FOCAL (antes del demo):
    1. Coloca un objeto a exactamente 1.0 metro de la cámara.
    2. Descomenta la línea [CALIB] en main_vision.py.
    3. Lee en consola el valor "profundidad_media" que imprime MiDaS.
    4. Ese número ES tu CONSTANTE_FOCAL. Pégalo abajo.
    5. Verifica: a 2 metros debe reportar ~2.0 m.
"""

import os
import cv2
import time
import queue
import threading
import numpy as np
import pygame

# Constante de calibración de distancia (ver instrucciones en docstring)
CONSTANTE_FOCAL = 350.0

# Cooldown entre alertas de la misma clase de objeto
COOLDOWN_ALERTA_SEG = 4.0

# Cooldown especial para alertas críticas de suelo (clase -1: escalón/precipicio).
# Debe coincidir aproximadamente con la duración del MP3 "escalon_frente.mp3"
# para evitar el "efecto metralleta" sin silenciar la alerta más de lo necesario.
COOLDOWN_CRITICO_SEG = 2.0

# Área normalizada mínima para considerar un objeto relevante
UMBRAL_AREA_RELEVANTE = 0.03

# Directorio donde están los MP3 pregrabados generados por generador_audios.py.

DIRECTORIO_AUDIO = "audios"

# Escalones de distancia disponibles, si MiDaS da un valor intermedio,
# obtener_archivo_audio() lo aproximará al escalón más cercano.
ESCALONES_DISTANCIA = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Caché de archivos de audio en memoria
ARCHIVOS_AUDIO_DISPONIBLES: set[str] = set()

def inicializar_cache_audio(directorio: str = DIRECTORIO_AUDIO) -> None:

    global ARCHIVOS_AUDIO_DISPONIBLES
    ARCHIVOS_AUDIO_DISPONIBLES.clear()

    if not os.path.isdir(directorio):
        print(f"[CACHE] Directorio '{directorio}' no encontrado. Cache vacío.")
        return

    for nombre in os.listdir(directorio):
        ruta_completa = os.path.join(directorio, nombre)
        if os.path.isfile(ruta_completa):
            # Guardamos la ruta completa relativa
            ARCHIVOS_AUDIO_DISPONIBLES.add(ruta_completa)

    print(f"[CACHE] {len(ARCHIVOS_AUDIO_DISPONIBLES)} archivos de audio cacheados.")


def _audio_existe(ruta: str) -> bool:

    if ARCHIVOS_AUDIO_DISPONIBLES:
        return ruta in ARCHIVOS_AUDIO_DISPONIBLES
    return os.path.exists(ruta)  # Fallback seguro


# AudioWorker: cola asíncrona de archivo MP3 → pygame

class AudioWorker:

    def __init__(self):
        self._cola: queue.Queue[str | None] = queue.Queue(maxsize=3)
        self._activo = True
        self._lock = threading.Lock()
        self._interrupcion_activa = threading.Event()

        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

        self._hilo = threading.Thread(
            target=self._loop_audio, daemon=True, name="AudioWorker"
        )
        self._hilo.start()

    def encolar(self, archivo: str, es_critico: bool = False):

        if es_critico:
            with self._lock:
                self._interrupcion_activa.set()

                # Vaciar toda la cola existente
                descartados = 0
                while not self._cola.empty():
                    try:
                        self._cola.get_nowait()
                        self._cola.task_done()
                        descartados += 1
                    except queue.Empty:
                        break

                # Detener audio en curso
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass

                if descartados:
                    print(f"[AUDIO][PREEMPT]  {descartados} alertas descartadas → prioridad: {archivo}")
                else:
                    print(f"[AUDIO][PREEMPT] Prioridad crítica: {archivo}")

        # Si la cola está llena, descartar el audio más antiguo antes de insertar.
        with self._lock:
            if self._cola.full():
                try:
                    self._cola.get_nowait()
                    self._cola.task_done()
                    print(f"[AUDIO][DROP] Cola llena → descartado audio antiguo para insertar: {archivo}")
                except queue.Empty:
                    pass
            self._cola.put_nowait(archivo)

    def detener(self):
        self._activo = False
        self._cola.put(None)

    def _loop_audio(self):
        while self._activo:
            archivo = self._cola.get()

            if archivo is None:
                break

            print(f"[AUDIO] >> {archivo}")

            try:
                if not _audio_existe(archivo):
                    # Archivo no grabado → bip genérico si existe, si no silencio
                    bip = os.path.join(DIRECTORIO_AUDIO, "beep.mp3")
                    if _audio_existe(bip):
                        archivo = bip
                    else:
                        self._cola.task_done()
                        continue

                # el nuevo audio, para no abortar inmediatamente su propia reproducción.
                self._interrupcion_activa.clear()
                pygame.mixer.music.load(archivo)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    if self._interrupcion_activa.is_set():
                        break
                    time.sleep(0.05)

            except Exception as e:
                print(f"[AudioWorker] Error al reproducir '{archivo}': {e}")

            self._cola.task_done()


# GestorCooldown: anti-spam por clase de objeto

class GestorCooldown:
    """Anti-spam de alertas por clase de objeto.

    Parámetros
    ----------
    cooldown_seg : float
        Cooldown general (segundos) entre alertas de la misma clase.
    cooldowns_especiales : dict[int, float] | None
        Mapa opcional {clase_id: segundos} para clases que necesitan
        un cooldown distinto al general.  Por defecto se incluye:
          -1 → COOLDOWN_CRITICO_SEG (escalón/precipicio, 2.0 s)
    """

    def __init__(
        self,
        cooldown_seg: float = COOLDOWN_ALERTA_SEG,
        cooldowns_especiales: dict | None = None,
    ):
        self._cd = cooldown_seg
        self._registro: dict[int, float] = {}

        # Cooldowns especiales: permiten configurar tiempos distintos por clase.
        # La clase -1 (peligro de suelo) usa el cooldown crítico por defecto.
        self._especiales: dict[int, float] = {-1: COOLDOWN_CRITICO_SEG}
        if cooldowns_especiales:
            self._especiales.update(cooldowns_especiales)

    def _cooldown_para(self, clase_id: int) -> float:
        """Devuelve el cooldown aplicable a esta clase (especial o general)."""
        return self._especiales.get(clase_id, self._cd)

    def puede_alertar(self, clase_id: int) -> bool:
        cd = self._cooldown_para(clase_id)
        return (time.time() - self._registro.get(clase_id, 0.0)) >= cd

    def registrar(self, clase_id: int):
        self._registro[clase_id] = time.time()


# Funciones matemáticas de fusión espacial YOLO + MiDaS

def extraer_profundidad_roi(
    depth_map: np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
) -> float:
    h, w = depth_map.shape[:2]
    x1 = max(0, min(x1, w - 1));  x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1));  y2 = max(y1 + 1, min(y2, h))
    roi = depth_map[y1:y2, x1:x2]
    return float(np.median(roi)) if roi.size > 0 else 0.0


def calcular_distancia_metros(profundidad_media: float) -> float:
    return round(CONSTANTE_FOCAL / (profundidad_media + 1e-6), 1)


def posicion_en_frame(centro_x: float, ancho: int) -> str:
    t = ancho // 3
    if centro_x < t:       return "a tu izquierda"
    elif centro_x < 2 * t: return "al frente"
    else:                  return "a tu derecha"


def construir_alerta(nombre: str, distancia: float, posicion: str) -> str:
    # Solo para la pantalla de OpenCV (debug). NO se envía al audio.
    if distancia < 1.0:    prefijo = "¡Cuidado! "
    elif distancia < 2.0:  prefijo = "Atención. "
    else:                  prefijo = ""
    return f"{prefijo}{nombre} {posicion}, a {distancia} metros"


def construir_nombre_audio(nombre_clase: str, posicion_str: str) -> str:
    mapa_pos = {
        "a tu izquierda": "izquierda",
        "al frente":      "frente",
        "a tu derecha":   "derecha",
    }
    pos_simple = mapa_pos.get(posicion_str, "frente")
    return os.path.join(DIRECTORIO_AUDIO, f"{nombre_clase}_{pos_simple}.mp3")


def _redondear_escalon(distancia_metros: float) -> float:

    if not ESCALONES_DISTANCIA:
        return distancia_metros

    # Tope superior: nunca superar el máximo pregrabado
    if distancia_metros >= ESCALONES_DISTANCIA[-1]:
        return ESCALONES_DISTANCIA[-1]

    # Tope inferior
    if distancia_metros <= ESCALONES_DISTANCIA[0]:
        return ESCALONES_DISTANCIA[0]

    # Buscar el escalón más cercano por mínima distancia absoluta
    return min(ESCALONES_DISTANCIA, key=lambda e: abs(e - distancia_metros))


def obtener_archivo_audio(
    audio_worker,           # Instancia de AudioWorker a la que encolar el MP3
    clase_nombre: str,
    posicion: str,          # Valor devuelto por posicion_en_frame() (texto largo)
    distancia_metros: float,
) -> None:

    # 0. Traducir el nombre de clase YOLO al prefijo real del archivo MP3.
    # Los nombres en CLASES_OBSTACULOS usan tildes/abreviaturas del español,
    # pero los MP3 pregrabados usan prefijos distintos (sin tildes, nombres completos).
    MAPA_CLASE_AUDIO = {
        "persona":      "persona",
        "bicicleta":    "bicicleta",
        "coche":        "automovil",
        "moto":         "motocicleta",
        "autob\u00fas":       "automovil",   # no hay MP3 de autobús, usar automóvil
        "cami\u00f3n":       "automovil",   # no hay MP3 de camión, usar automóvil
        "silla":        "silla",
        "sof\u00e1":         "sofa",
        "cama":         "silla",        # fallback
        "mesa":         "mesa",
        "televisi\u00f3n":   "televisor",
        "laptop":       "laptop",
        "tel\u00e9fono":     "bolso",        # fallback más parecido
        "mochila":      "mochila",
        "maleta":       "maleta",
    }
    prefijo_audio = MAPA_CLASE_AUDIO.get(clase_nombre, clase_nombre)

    # 1. Traducir posición larga a clave de nombre de archivo
    mapa_pos = {
        "a tu izquierda": "izquierda",
        "al frente":      "frente",
        "a tu derecha":   "derecha",
    }
    pos_clave = mapa_pos.get(posicion, "frente")

    # 2. Redondeo Escalón: aproximar distancia al step pregrabado más cercano
    dist_binned = _redondear_escalon(distancia_metros)

    # 3. Construir nombre del archivo usando el prefijo traducido
    dist_sufijo    = f"{dist_binned:.1f}".replace(".", "_")
    nombre_archivo = f"{prefijo_audio}_{pos_clave}_{dist_sufijo}.mp3"
    ruta_audio     = os.path.join(DIRECTORIO_AUDIO, nombre_archivo)

    # 4. Determinar si es alerta crítica (escalones o distancia < 1.0 m)
    es_critico = ("escalon" in clase_nombre.lower()) or (distancia_metros < 1.0)

    # 5. Verificar existencia (O(1) con caché) y encolar
    if _audio_existe(ruta_audio):
        # Caso normal: archivo pregrabado encontrado → reproducir
        audio_worker.encolar(ruta_audio, es_critico=es_critico)

    else:
        # 6. Fallback: beep genérico si el MP3 específico no existe
        print(
            f"[AUDIO][WARN] Archivo no encontrado: '{ruta_audio}'. "
            f"Ejecuta generador_audios.py para generarlo."
        )
        ruta_beep = os.path.join(DIRECTORIO_AUDIO, "beep.mp3")
        if _audio_existe(ruta_beep):
            audio_worker.encolar(ruta_beep, es_critico=es_critico)
        # Si tampoco existe el beep → silencio total (pass implícito)


# MonitorSaludCamara: detecta condiciones degradadas del entorno visual

class MonitorSaludCamara:
    UMBRAL_OSCURIDAD  = 35.0   # Brillo promedio de píxel (0–255)
    UMBRAL_BORROSIDAD = 40.0   # Varianza del Laplaciano
    FPS_MINIMOS       = 5.0    # FPS por debajo de esto = alerta de lentitud
    COOLDOWN_SALUD    = 15.0   # Segundos entre alertas de salud repetidas

    def __init__(self):
        import cv2 as _cv2
        self._cv2 = _cv2
        self._ultimo_chequeo = 0.0

    def verificar(self, frame, fps_actuales: float) -> str | None:

        ahora = time.time()
        if (ahora - self._ultimo_chequeo) < self.COOLDOWN_SALUD:
            return None

        gris = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        brillo = self._cv2.mean(gris)[0]
        borrosidad = self._cv2.Laplacian(gris, 6).var()  # 6 = CV_64F

        alerta = None
        if brillo < self.UMBRAL_OSCURIDAD:
            print(f"[SALUD] Poca luz. Brillo={brillo:.1f}")
            alerta = os.path.join(DIRECTORIO_AUDIO, "oscuro.mp3")
        elif borrosidad < self.UMBRAL_BORROSIDAD:
            print(f"[SALUD] Cámara borrosa. Varianza={borrosidad:.1f}")
            alerta = os.path.join(DIRECTORIO_AUDIO, "camara_sucia.mp3")
        elif fps_actuales < self.FPS_MINIMOS:
            print(f"[SALUD] FPS bajos. FPS={fps_actuales:.1f}")
            alerta = os.path.join(DIRECTORIO_AUDIO, "procesando_lento.mp3")

        if alerta:
            self._ultimo_chequeo = ahora
        return alerta


# MonitorMovimiento: detecta velocidad, giros y frenadas usando Optical Flow

class MonitorMovimiento:

    # Umbrales ajustables (en píxeles por frame a 30 FPS)
    UMBRAL_VELOCIDAD_ALTA  = 18.0   # Desplazamiento medio alto → va muy rápido
    UMBRAL_GIRO_BRUSCO     = 12.0   # Componente horizontal neta → giro lateral
    UMBRAL_FRENADA         = 10.0   # Caída de velocidad entre frames consecutivos
    COOLDOWN_MOVIMIENTO    = 3.0    # Segundos entre alertas de movimiento

    # Parámetros de Lucas-Kanade
    LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    FEATURE_PARAMS = dict(maxCorners=80, qualityLevel=0.3, minDistance=10, blockSize=7)

    def __init__(self):
        self._frame_anterior_gris = None
        self._puntos_anteriores   = None
        self._velocidad_anterior  = 0.0
        self._ultimo_chequeo      = 0.0
        self._refresco_puntos     = 0     # Contador para redetectar puntos cada N frames

    def analizar(self, frame) -> str | None:

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Primer frame: solo guardar referencia
        if self._frame_anterior_gris is None:
            self._frame_anterior_gris = gris
            self._puntos_anteriores = cv2.goodFeaturesToTrack(gris, **self.FEATURE_PARAMS)
            return None

        # Re-detectar puntos cada 20 frames o si quedan muy pocos
        self._refresco_puntos += 1
        if self._puntos_anteriores is None or len(self._puntos_anteriores) < 10 \
                or self._refresco_puntos >= 20:
            self._puntos_anteriores = cv2.goodFeaturesToTrack(gris, **self.FEATURE_PARAMS)
            self._refresco_puntos = 0
            self._frame_anterior_gris = gris
            return None

        # Calcular flujo óptico sparse (Lucas-Kanade)
        puntos_nuevos, status, _ = cv2.calcOpticalFlowPyrLK(
            self._frame_anterior_gris, gris,
            self._puntos_anteriores, None,
            **self.LK_PARAMS,
        )

        # Filtrar solo puntos con tracking exitoso
        buenos_ant = self._puntos_anteriores[status == 1]
        buenos_nue = puntos_nuevos[status == 1]

        self._frame_anterior_gris = gris
        self._puntos_anteriores   = buenos_nue.reshape(-1, 1, 2) if len(buenos_nue) > 0 else None

        if len(buenos_ant) < 5:
            return None

        # Vectores de desplazamiento (dx, dy) por punto
        delta     = buenos_nue - buenos_ant
        magnitud  = np.linalg.norm(delta, axis=1)          # Velocidad por punto
        vel_media = float(np.mean(magnitud))                # Velocidad global (píxeles/frame)
        dx_medio  = float(np.mean(delta[:, 0]))             # Componente horizontal neta
        dy_medio  = float(np.mean(delta[:, 1]))             # Componente vertical neta

        # Verificar cooldown antes de alertar
        ahora = time.time()
        if (ahora - self._ultimo_chequeo) < self.COOLDOWN_MOVIMIENTO:
            self._velocidad_anterior = vel_media
            return None

        alerta = None

        # 1. Velocidad excesiva
        if vel_media > self.UMBRAL_VELOCIDAD_ALTA:
            print(f"[MOVIMIENTO] Velocidad alta. Flujo={vel_media:.1f}px/frame")
            alerta = os.path.join(DIRECTORIO_AUDIO, "rapido.mp3")

        # 2. Giro brusco (componente horizontal neta dominante)
        elif abs(dx_medio) > self.UMBRAL_GIRO_BRUSCO:
            lado = "derecha" if dx_medio > 0 else "izquierda"
            print(f"[MOVIMIENTO] Giro brusco {lado}. dx={dx_medio:.1f}")
            alerta = os.path.join(DIRECTORIO_AUDIO, f"giro_{lado}.mp3")

        # 3. Frenada súbita
        elif (self._velocidad_anterior - vel_media) > self.UMBRAL_FRENADA \
                and self._velocidad_anterior > 5.0:
            print(f"[MOVIMIENTO] Frenada. {self._velocidad_anterior:.1f}→{vel_media:.1f}px/frame")
            alerta = os.path.join(DIRECTORIO_AUDIO, "frenada.mp3")

        # 4. Marcha atrás
        elif dy_medio > self.UMBRAL_VELOCIDAD_ALTA * 0.6 and vel_media > 5.0:
            print(f"[MOVIMIENTO] Posible retroceso. dy={dy_medio:.1f}")
            alerta = os.path.join(DIRECTORIO_AUDIO, "retroceso.mp3")

        if alerta:
            self._ultimo_chequeo = ahora

        self._velocidad_anterior = vel_media
        return alerta
