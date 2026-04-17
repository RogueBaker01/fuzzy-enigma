import os
import sys
import argparse
import cv2
import time
import pygame

from modelo_midas  import MidasDepthEstimator
from modelo_yolo   import YoloDetector, nombrar_objetos
from fusion_logica import (
    AudioWorker,
    GestorCooldown,
    MonitorSaludCamara,
    MonitorMovimiento,
    extraer_profundidad_roi,
    calcular_distancia_metros,
    posicion_en_frame,
    construir_alerta,
    obtener_archivo_audio,
    UMBRAL_AREA_RELEVANTE,
    DIRECTORIO_AUDIO,
)

# ---------------------------------------------------------------------------
# URL del stream RTSP del teléfono.
# Aplicaciones compatibles: DroidCam, IP Webcam (Android) | EpocCam (iOS)
# Formato genérico:  rtsp://<IP_DEL_TELEFONO>:<PUERTO>/video
# IP Webcam (Android, puerto 8080): "http://192.168.1.X:8080/video"
# DroidCam  (puerto 4747):          "http://192.168.1.X:4747/video"
# EpocCam   (iOS, RTSP):            "rtsp://192.168.1.X:2222/live"
# ---------------------------------------------------------------------------
FUENTE_TELEFONO_URL = "http://192.168.1.100:8080/video"  # <-- cambia la IP de tu teléfono


def resolver_fuente_video(modo: str) -> tuple[str | int, str]:
    """
    Resuelve la fuente de víneo según el modo seleccionado.
    Retorna (src, etiqueta_hud) donde src es el argumento de cv2.VideoCapture.

    Modos:
        archivo  → videoplayback2.mp4  (archivo local de prueba)
        webcam   → 0                   (cámara integrada / USB)
        telefono → FUENTE_TELEFONO_URL con fallback a webcam
    """
    if modo == "webcam":
        print("[FUENTE] Modo: Cámara web local (dispositivo 0)")
        return 0, "WEBCAM"

    if modo == "telefono":
        print(f"[FUENTE] Modo: Teléfono  →  {FUENTE_TELEFONO_URL}")
        print("[FUENTE] Verificando conexión con el teléfono...")
        cap_test = cv2.VideoCapture(FUENTE_TELEFONO_URL)
        if cap_test.isOpened():
            cap_test.release()
            print("[FUENTE] ✔ Teléfono conectado.")
            return FUENTE_TELEFONO_URL, "TELEFONO"
        else:
            cap_test.release()
            print("[FUENTE][WARN] No se pudo conectar al teléfono.")
            print(f"[FUENTE][WARN] Verifica que la IP sea correcta: {FUENTE_TELEFONO_URL}")
            print("[FUENTE][WARN] Usando webcam local como fallback.")
            return 0, "WEBCAM (fallback)"

    # Modo por defecto: archivo de video local
    print(f"[FUENTE] Modo: Archivo de video local")
    return "videoplayback2.mp4", "ARCHIVO"


def main():
    # --- Selección de fuente de video vía argumento CLI ---
    parser = argparse.ArgumentParser(
        description="Sistema Asistente Visual — Edge Computing YOLOv8 + MiDaS"
    )
    parser.add_argument(
        "--fuente",
        choices=["archivo", "webcam", "telefono"],
        default="archivo",
        help=(
            "Fuente de video: "
            "'archivo' = videoplayback2.mp4 (default), "
            "'webcam'  = cámara local (dispositivo 0), "
            "'telefono' = stream del teléfono via IP/RTSP"
        ),
    )
    args = parser.parse_args()

    # 1. Cargar modelos de CV
    print("\n[1/3] Cargando YoloDetector (FP16)...")
    yolo = YoloDetector()

    print("[2/3] Cargando MiDaS_small (FP16)...")
    midas = MidasDepthEstimator()

    # 2. Iniciar módulo de audio y cooldowns
    print("[3/3] Iniciando AudioWorker (MP3 locales)...")
    audio       = AudioWorker()
    cooldown    = GestorCooldown()
    monitor     = MonitorSaludCamara()
    monitor_mov = MonitorMovimiento()

    # 3. Fuente de video (seleccionada por CLI)
    src, etiqueta_fuente = resolver_fuente_video(args.fuente)
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir la fuente de video: '{src}'")
        if args.fuente == "telefono":
            print("[AYUDA] Asegúrate de que:")
            print(f"         1. El teléfono y la PC estén en la misma red Wi-Fi")
            print(f"         2. La app de streaming esté corriendo en el teléfono")
            print(f"         3. La IP en FUENTE_TELEFONO_URL sea correcta: {FUENTE_TELEFONO_URL}")
        audio.detener()
        return

    print(f"\n[OK] Sistema listo. Fuente activa: {etiqueta_fuente}")
    print("     Controles: [Q] Salir  |  [D] Describir entorno\n")

    # Paleta de colores (BGR)
    C_TERCIOS = (0, 230, 60)
    C_LEJOS   = (0, 230, 230)
    C_CERCA   = (0, 40, 255)
    C_PELIGRO = (0, 0, 255)
    C_SEGURO  = (0, 220, 60)
    C_TEXTO   = (255, 255, 255)
    C_BG      = (15, 15, 15)

    t_fps = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        alto, ancho = frame.shape[:2]
        tercio      = ancho // 3

        # Monitor de salud de la cámara (brillo, borrosidad, FPS)
        fps   = 1.0 / max(time.time() - t_fps, 1e-6)
        t_fps = time.time()
        alerta_salud = monitor.verificar(frame, fps)
        if alerta_salud:
            audio.encolar(alerta_salud)

        # Monitor de movimiento (velocidad, giros, frenadas) vía Optical Flow
        alerta_mov = monitor_mov.analizar(frame)
        if alerta_mov:
            audio.encolar(alerta_mov)

        # Conteos por zona para descripción bajo demanda (tecla 'd')
        conteo_izq: dict = {}
        conteo_cen: dict = {}
        conteo_der: dict = {}

        # 1: MiDaS → mapa de profundidad + detección de peligro en el suelo
        depth_map, peligro_suelo = midas.estimate_depth_and_danger(frame)

        # 2: YOLO → detección de obstáculos con FP16
        resultado = yolo.detectar(frame)

        # 3: Fusión Espacial — correlacionar bbox de YOLO con depth map de MiDaS
        for box in resultado.boxes:
            clase_id = int(box.cls[0])
            nombre   = yolo.nombres.get(clase_id, "objeto")

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx        = (x1 + x2) / 2
            area_norm = ((x2 - x1) * (y2 - y1)) / (alto * ancho)

            # Conteo por zona (independiente del umbral de relevancia)
            if cx < tercio:
                conteo_izq[nombre] = conteo_izq.get(nombre, 0) + 1
            elif cx < 2 * tercio:
                conteo_cen[nombre] = conteo_cen.get(nombre, 0) + 1
            else:
                conteo_der[nombre] = conteo_der.get(nombre, 0) + 1

            if area_norm < UMBRAL_AREA_RELEVANTE:
                continue

            # Profundidad mediana del objeto en el mapa de MiDaS
            prof  = extraer_profundidad_roi(depth_map, x1, y1, x2, y2)

            # [CALIB] Descomenta para calibrar CONSTANTE_FOCAL:
            # print(f"[CALIB] {nombre} | profundidad_media={prof:.2f}")

            dist  = calcular_distancia_metros(prof)
            pos   = posicion_en_frame(cx, ancho)
            color = C_CERCA if dist < 2.0 else C_LEJOS

            # Render del bbox con distancia estimada
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{nombre} | {dist}m"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXTO, 1, cv2.LINE_AA)

            # Alerta de voz: enrutador con binning de distancia
            if cooldown.puede_alertar(clase_id):
                obtener_archivo_audio(audio, nombre, pos, dist)
                cooldown.registrar(clase_id)

        # 4: Alerta de suelo (MiDaS)
        roi_top     = int(alto * 0.7)
        color_suelo = C_PELIGRO if peligro_suelo else C_SEGURO
        cv2.rectangle(frame, (0, roi_top), (ancho, alto), color_suelo, 2)

        if peligro_suelo:
            cv2.rectangle(frame, (8, roi_top + 8), (310, roi_top + 36), C_BG, -1)
            cv2.putText(frame, "!!! PELIGRO: ESCALON / PRECIPICIO",
                        (12, roi_top + 29), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, C_PELIGRO, 2, cv2.LINE_AA)
            if cooldown.puede_alertar(-1):
                audio.encolar(os.path.join(DIRECTORIO_AUDIO, "escalon_frente.mp3"))
                cooldown.registrar(-1)

        # 5: Overlay de tercios, FPS y etiqueta de fuente
        cv2.line(frame, (tercio, 0), (tercio, alto), C_TERCIOS, 2)
        cv2.line(frame, (2 * tercio, 0), (2 * tercio, alto), C_TERCIOS, 2)

        fps   = 1.0 / max(time.time() - t_fps, 1e-6)
        t_fps = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (ancho - 110, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TERCIOS, 2, cv2.LINE_AA)

        # Etiqueta de fuente activa (esquina inferior izquierda)
        etiqueta_txt = f"FUENTE: {etiqueta_fuente}"
        (ew, eh), _ = cv2.getTextSize(etiqueta_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (6, alto - 28), (14 + ew, alto - 6), C_BG, -1)
        cv2.putText(frame, etiqueta_txt, (10, alto - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TERCIOS, 1, cv2.LINE_AA)

        # 6: Mostrar ventanas
        depth_jet = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        cv2.imshow("Sistema Asistente Visual", frame)
        cv2.imshow("Mapa de Profundidad (MiDaS)", depth_jet)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("d"):
            # Descripción completa del entorno bajo demanda
            print("\n[BOTÓN D] Descripción del entorno...")
            partes = []
            if not conteo_cen and not conteo_izq and not conteo_der:
                descripcion = "El camino está libre."
            else:
                if conteo_cen:
                    partes.append("Al frente hay " + " y ".join(
                        nombrar_objetos(c, o) for o, c in conteo_cen.items()) + ".")
                if conteo_izq:
                    partes.append("A tu izquierda hay " + " y ".join(
                        nombrar_objetos(c, o) for o, c in conteo_izq.items()) + ".")
                if conteo_der:
                    partes.append("A tu derecha hay " + " y ".join(
                        nombrar_objetos(c, o) for o, c in conteo_der.items()) + ".")
                descripcion = " ".join(partes)
            # Descripción solo en consola — no hay MP3 dinámico pregrabado
            print(f"Descripción: '{descripcion}'\n")

    print("\nCerrando sistema...")
    audio.detener()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("Sistema cerrado correctamente.")


if __name__ == "__main__":
    main()
