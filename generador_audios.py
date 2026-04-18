"""
generador_audios.py — Script independiente de generación masiva de audios TTS.

Uso:
    python generador_audios.py

Requisitos:
    pip install elevenlabs python-dotenv

Estructura de salida:
    audios/
        persona_izquierda_0_5.mp3
        persona_izquierda_1_0.mp3
        ...
        silla_derecha_3_0.mp3

Nomenclatura estricta:
    {clase}_{posicion}_{distancia_con_guion_bajo}.mp3
    Ejemplo: 0.5 m → "0_5", 1.5 m → "1_5", 3.0 m → "3_0"

IMPORTANTE: Coloca tu ELEVENLABS_API_KEY en el archivo .env antes de ejecutar.
"""

import os
import time
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Voice ID de ElevenLabs.
# Voz seleccionada: "Cristina Campos - Natural Conversations"
#   - Perfil: Mujer adulta, acento mexicano, tono conversacional y amigable
#   - Ideal para: IA conversacional en tiempo real (live conversational AI)
#   - Voice ID: CaJslL1xziwefCeTNzHv
# Para cambiarla: https://api.elevenlabs.io/v1/voices
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "CaJslL1xziwefCeTNzHv")

# Modelo: Eleven Multilingual v2 — soporte nativo de español (recomendado con Cristina Campos)
MODELO = "eleven_multilingual_v2"

# Directorio de salida para los MP3 generados
DIRECTORIO_SALIDA = "audios"

# Pausa entre llamadas API para no saturar el rate-limit (en segundos)
PAUSA_ENTRE_LLAMADAS = 0.6

# ---------------------------------------------------------------------------
# Matrices de generación
# ---------------------------------------------------------------------------

# Clases de objetos detectables por YOLOv8 (nombres en español)
# Añade o quita clases según el vocabulario que use tu modelo_yolo.py
CLASES = [
    "persona",
    "bicicleta",
    "automovil",
    "motocicleta",
    "silla",
    "sofa",
    "mesa",
    "televisor",
    "laptop",
    "mochila",
    "bolso",
    "maleta",
    "freno",       # señales de tráfico relevantes para movilidad
    "banco",
    "planta",
    "perro",
    "gato",
    "puerta",
    "columna",
    "escalon",
]

# Posiciones relativas (nombre interno → texto natural de la frase)
POSICIONES = {
    "izquierda": "a tu izquierda",
    "frente":    "al frente",
    "derecha":   "a tu derecha",
}

# Escalones de distancia en metros (los mismos que usará el binning en fusion_logica.py)
DISTANCIAS_METROS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def distancia_a_sufijo(dist: float) -> str:
    """
    Convierte un float de distancia al sufijo de nombre de archivo.
    Regla: reemplazar el punto por guion bajo.
        0.5  → "0_5"
        1.0  → "1_0"
        2.5  → "2_5"
    """
    return f"{dist:.1f}".replace(".", "_")


def construir_frase(clase: str, pos_texto: str, dist_metros: float) -> str:
    """
    Genera una frase natural en español para el TTS.
    Adapta el prefijo de urgencia según la proximidad:
        < 1.0 m → "¡Cuidado! "
        < 2.0 m → "Atención, "
        ≥ 2.0 m → sin prefijo

    Ejemplo de salida:
        "Atención, silla a tu izquierda a 1.5 metros"
        "¡Cuidado! persona al frente a 0.5 metros"
    """
    if dist_metros < 1.0:
        prefijo = "¡Cuidado! "
    elif dist_metros < 2.0:
        prefijo = "Atención, "
    else:
        prefijo = ""

    # Singularizar correctamente: "silla" → "una silla", etc. (artículo genérico)
    return f"{prefijo}{clase} {pos_texto} a {dist_metros:.1f} metros"


def construir_nombre_archivo(clase: str, pos_clave: str, dist: float) -> str:
    """
    Devuelve el nombre de archivo MP3 normalizado (sin ruta).
    Ejemplo: construir_nombre_archivo("silla", "izquierda", 1.5) → "silla_izquierda_1_5.mp3"
    """
    return f"{clase}_{pos_clave}_{distancia_a_sufijo(dist)}.mp3"


# ---------------------------------------------------------------------------
# Lógica principal de generación
# ---------------------------------------------------------------------------

def generar_todos_los_audios():
    """
    Itera sobre la matriz CLASES × POSICIONES × DISTANCIAS y genera un MP3
    por cada combinación usando la API de ElevenLabs. Omite los archivos que
    ya existen en disco para reanudar una generación interrumpida.
    """

    if not ELEVENLABS_API_KEY:
        raise ValueError(
            "No se encontró ELEVENLABS_API_KEY. "
            "Crea un archivo .env con: ELEVENLABS_API_KEY=tu_clave_api"
        )

    # Crear directorio de salida si no existe
    os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)

    # Inicializar cliente ElevenLabs
    cliente = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    # Calcular total para mostrar progreso
    total = len(CLASES) * len(POSICIONES) * len(DISTANCIAS_METROS)
    generados = 0
    omitidos  = 0
    errores   = 0
    contador  = 0

    print(f"\n{'='*60}")
    print(f"  Generador de Audios TTS — ElevenLabs")
    print(f"  Total de combinaciones: {total}")
    print(f"  Salida: ./{DIRECTORIO_SALIDA}/")
    print(f"{'='*60}\n")

    for clase in CLASES:
        for pos_clave, pos_texto in POSICIONES.items():
            for dist in DISTANCIAS_METROS:

                contador += 1
                nombre_archivo = construir_nombre_archivo(clase, pos_clave, dist)
                ruta_completa  = os.path.join(DIRECTORIO_SALIDA, nombre_archivo)

                # --- Si el archivo ya existe, saltar para reanudar sin duplicar ---
                if os.path.exists(ruta_completa):
                    print(f"  [{contador:>4}/{total}] OMITIDO (ya existe): {nombre_archivo}")
                    omitidos += 1
                    continue

                frase = construir_frase(clase, pos_texto, dist)
                print(f"  [{contador:>4}/{total}] Generando: '{frase}' → {nombre_archivo}")

                try:
                    # Llamada a la API de ElevenLabs para convertir texto a voz
                    audio_stream = cliente.text_to_speech.convert(
                        voice_id=VOICE_ID,
                        text=frase,
                        model_id=MODELO,
                        voice_settings=VoiceSettings(
                            stability=0.55,          # Menos variación = más consistencia
                            similarity_boost=0.80,   # Alta fidelidad a la voz base
                            style=0.20,              # Expresividad moderada para alertas
                            use_speaker_boost=True,
                        ),
                        output_format="mp3_44100_128",  # Calidad óptima para pygame
                    )

                    # Escribir el stream devuelto al archivo MP3
                    with open(ruta_completa, "wb") as f:
                        for chunk in audio_stream:
                            if chunk:
                                f.write(chunk)

                    generados += 1
                    print(f"          ✓ Guardado: {ruta_completa}")

                except Exception as e:
                    errores += 1
                    print(f"          ✗ ERROR en '{nombre_archivo}': {e}")
                    # Continúa con el siguiente; no interrumpe el lote completo

                # Pausa entre llamadas para respetar el rate-limit de la API
                time.sleep(PAUSA_ENTRE_LLAMADAS)

    # Resumen final
    print(f"\n{'='*60}")
    print(f"  RESUMEN DE GENERACIÓN")
    print(f"  ✓ Generados:  {generados}")
    print(f"  ⊘ Omitidos:   {omitidos}  (ya existían en disco)")
    print(f"  ✗ Errores:    {errores}")
    print(f"  Total:        {total}")
    print(f"{'='*60}\n")

    if errores > 0:
        print(f"[AVISO] {errores} archivo(s) fallaron. "
              "Vuelve a ejecutar el script para reintentar solo los faltantes.\n")


# ---------------------------------------------------------------------------
# Generación de archivos de fallback (beep genérico y alertas especiales)
# ---------------------------------------------------------------------------

def generar_audios_especiales():
    """
    Genera los archivos de audio que NO forman parte de la matriz
    CLASE×POSICIÓN×DISTANCIA pero son necesarios para el sistema:
        - beep.mp3            (fallback silencioso de AudioWorker)
        - escalon_frente.mp3  (alerta de MiDaS por peligro en suelo)
        - oscuro.mp3          (MonitorSaludCamara)
        - camara_sucia.mp3    (MonitorSaludCamara)
        - procesando_lento.mp3(MonitorSaludCamara)
        - rapido.mp3          (MonitorMovimiento)
        - giro_izquierda.mp3  (MonitorMovimiento)
        - giro_derecha.mp3    (MonitorMovimiento)
        - frenada.mp3         (MonitorMovimiento)
        - retroceso.mp3       (MonitorMovimiento)
    """

    if not ELEVENLABS_API_KEY:
        print("[AVISO] Sin API key, se omiten los audios especiales.")
        return

    cliente = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    especiales = {
        "escalon_frente.mp3":      "¡Cuidado! Escalón o desnivel al frente.",
        "oscuro.mp3":              "Poca luz. Ten cuidado.",
        "camara_sucia.mp3":        "La cámara puede estar sucia o bloqueada.",
        "procesando_lento.mp3":    "El sistema está procesando más lento de lo normal.",
        "rapido.mp3":              "Vas muy rápido, reduce la velocidad.",
        "giro_izquierda.mp3":      "Giro brusco a la izquierda detectado.",
        "giro_derecha.mp3":        "Giro brusco a la derecha detectado.",
        "frenada.mp3":             "Frenada súbita detectada.",
        "retroceso.mp3":           "Posible retroceso detectado.",
        # Detección de pared/barrera (heurística MiDaS)
        "pared_frente.mp3":        "¡Cuidado! Pared al frente.",
        "pared_izquierda.mp3":     "¡Cuidado! Pared a tu izquierda.",
        "pared_derecha.mp3":       "¡Cuidado! Pared a tu derecha.",
    }

    print(f"\n--- Generando {len(especiales)} audios especiales del sistema ---\n")

    for nombre_archivo, frase in especiales.items():
        ruta_completa = os.path.join(DIRECTORIO_SALIDA, nombre_archivo)

        if os.path.exists(ruta_completa):
            print(f"  OMITIDO (ya existe): {nombre_archivo}")
            continue

        print(f"  Generando especial: '{frase}' → {nombre_archivo}")

        try:
            audio_stream = cliente.text_to_speech.convert(
                voice_id=VOICE_ID,
                text=frase,
                model_id=MODELO,
                voice_settings=VoiceSettings(
                    stability=0.60,
                    similarity_boost=0.80,
                    style=0.15,
                    use_speaker_boost=True,
                ),
                output_format="mp3_44100_128",
            )

            with open(ruta_completa, "wb") as f:
                for chunk in audio_stream:
                    if chunk:
                        f.write(chunk)

            print(f"          ✓ Guardado: {ruta_completa}")

        except Exception as e:
            print(f"          ✗ ERROR en '{nombre_archivo}': {e}")

        time.sleep(PAUSA_ENTRE_LLAMADAS)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generar_todos_los_audios()
    generar_audios_especiales()
