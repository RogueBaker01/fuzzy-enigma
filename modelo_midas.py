import cv2
import numpy as np
import torch
import time

# Fracción inferior del frame que representa el suelo inmediato
ROI_FRACCION_INFERIOR: float = 0.30

# Umbral de desviación estándar del gradiente vertical en la ROI.
# Un valor alto indica una discontinuidad brusca de profundidad
# (escalón hacia abajo o precipicio).
UMBRAL_STD_GRADIENTE: float = 12.0

# Nombre del modelo en torch.hub
MIDAS_MODELO: str = "MiDaS_small"
MIDAS_REPO: str = "intel-isl/MiDaS"


class EstimadorProfundidadMiDaS:

    def __init__(self):
        # Selección de dispositivo
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[MiDaS] GPU detectada: {torch.cuda.get_device_name(0)}")
            print(f"[MiDaS] VRAM disponible: "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device("cpu")
            print("[MiDaS] CUDA no disponible. Usando CPU (rendimiento reducido).")

        # Carga del modelo desde torch.hub
        print("[MiDaS] Cargando modelo MiDaS_small desde torch.hub...")
        self.modelo = torch.hub.load(
            MIDAS_REPO,
            MIDAS_MODELO,
            pretrained=True,
            trust_repo=True
        )

        # Modo evaluación + FP16 en GPU
        # .half() convierte pesos a float16, reduciendo VRAM ~50%
        # y acelerando la inferencia en GPUs NVIDIA con Tensor Cores.
        self.modelo.to(self.device)
        if self.device.type == "cuda":
            self.modelo = self.modelo.half()   # FP16 solo en GPU
        self.modelo.eval()
        print("[MiDaS] Modelo listo en modo evaluación "
              f"({'FP16/CUDA' if self.device.type == 'cuda' else 'FP32/CPU'}).")

        # Carga de transforms nativos de MiDaS_small
        transforms = torch.hub.load(
            MIDAS_REPO,
            "transforms",
            trust_repo=True
        )
        # MiDaS_small usa el transform "small_transform"
        self.transform = transforms.small_transform
        print("[MiDaS] Transforms cargados (small_transform).")

    # MÉTODO PRINCIPAL: PROCESAR FRAME

    def procesar_frame(
        self,
        frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """
        Procesa un frame BGR de OpenCV y retorna el mapa de
        profundidad normalizado más un indicador de peligro.

        Parámetros
        ----------
        frame_bgr : np.ndarray
            Frame capturado por OpenCV (H x W x 3, dtype uint8, BGR).

        Retorna
        -------
        mapa_visualizable : np.ndarray
            Mapa de profundidad escalado a uint8 (0–255), mismo
            tamaño que la entrada. Listo para aplicar colormaps.
        peligro_escaleras : bool
            True si se detecta una discontinuidad brusca en la
            zona del suelo (escalón o precipicio).
        """
        alto, ancho = frame_bgr.shape[:2]

        # 1. Pre-procesado con transforms de MiDaS
        # MiDaS espera imágenes en RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        entrada = self.transform(frame_rgb).to(self.device)

        # Convertir tensor a FP16 si estamos en GPU
        if self.device.type == "cuda":
            entrada = entrada.half()

        # 2. Inferencia (sin gradientes para rendimiento)
        with torch.no_grad():
            prediccion = self.modelo(entrada)

            # Reescalar la salida al tamaño original del frame
            prediccion = torch.nn.functional.interpolate(
                prediccion.unsqueeze(1),       # añade canal: (1, 1, H, W)
                size=(alto, ancho),
                mode="bicubic",
                align_corners=False,
            ).squeeze()                        # elimina dimensiones extra

        # 3. Pasar a CPU y convertir a NumPy
        # .float() convierte de FP16 → FP32 antes de .numpy()
        mapa_profundidad = prediccion.cpu().float().numpy()

        # 4. Normalización a uint8 (0–255)
        prof_min = mapa_profundidad.min()
        prof_max = mapa_profundidad.max()
        rango = prof_max - prof_min

        if rango > 1e-6:
            mapa_norm = (mapa_profundidad - prof_min) / rango
        else:
            mapa_norm = np.zeros_like(mapa_profundidad)

        mapa_visualizable = (mapa_norm * 255).astype(np.uint8)

        # 5. Detección de escalones en ROI
        peligro_escaleras = self._detectar_peligro_escaleras(
            mapa_profundidad, alto
        )

        return mapa_visualizable, peligro_escaleras

    # LÓGICA DE DETECCIÓN DE ESCALERAS / CAÍDAS

    def _detectar_peligro_escaleras(
        self,
        mapa_profundidad: np.ndarray,
        alto_frame: int
    ) -> bool:
        """
        Analiza el 30% inferior del mapa de profundidad (zona del
        suelo inmediato) buscando discontinuidades verticales bruscas
        que indiquen un escalón hacia abajo o un precipicio.

        Estrategia matemática
        ----------------------
        Se calcula el **gradiente vertical** (Sobel-Y) de la ROI de
        profundidad. Una desviación estándar alta del gradiente
        significa que la profundidad cambia abruptamente de modo
        vertical — señal característica de un borde de escalón.

        Parámetros
        ----------
        mapa_profundidad : np.ndarray
            Mapa de profundidad crudo (float32, valores relativos).
        alto_frame : int
            Altura total del frame original en píxeles.

        Retorna
        -------
        bool : True si existe peligro de escalón/caída.
        """
        # Extraer ROI: 30% inferior
        inicio_roi = int(alto_frame * (1.0 - ROI_FRACCION_INFERIOR))
        roi = mapa_profundidad[inicio_roi:, :]

        # Normalizar ROI a uint8 para Sobel
        roi_min = roi.min()
        roi_max = roi.max()
        rango = roi_max - roi_min

        if rango < 1e-6:
            # Sin variación de profundidad → no hay peligro
            return False

        roi_norm = ((roi - roi_min) / rango * 255).astype(np.uint8)

        # Gradiente vertical con Sobel
        # Sobel en dirección Y detecta cambios verticales de profundidad.
        # ksize=3 es suficientemente rápido para tiempo real.
        gradiente_y = cv2.Sobel(roi_norm, cv2.CV_32F, dx=0, dy=1, ksize=3)

        # Desviación estándar del gradiente
        # Un std alto indica bordes fuertes de profundidad (escalones).
        std_gradiente = float(np.std(np.abs(gradiente_y)))

        # Descomenta para calibrar el umbral durante pruebas:
        # print(f"[DEBUG] std_gradiente ROI: {std_gradiente:.2f}")

        return std_gradiente > UMBRAL_STD_GRADIENTE


# ENTORNO DE PRUEBA

if __name__ == "__main__":
    """
    Bloque de prueba autónomo.
    Usa la cámara web local (índice 0) para validar el módulo
    MiDaS de forma independiente, sin requerir el stream del iPhone.

    Ventanas mostradas:
      - "Frame Original": frame BGR con ROI marcada y alerta.
      - "Mapa de Profundidad": salida MiDaS en COLORMAP_JET.

    Presiona 'q' para salir.
    """

    print("=" * 60)
    print("  MÓDULO MiDaS — Prueba con Cámara Web Local")
    print("=" * 60)

    # Inicializar estimador
    estimador = EstimadorProfundidadMiDaS()

    # Abrir cámara web
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "[ERROR] No se pudo abrir la cámara web (índice 0). "
            "Verifica que no esté en uso por otra aplicación."
        )

    # Resolución de captura (más baja → más FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[PRUEBA] Cámara abierta. Presiona 'q' para salir.\n")

    # Variables para calcular FPS
    tiempo_anterior = time.perf_counter()
    fps_suavizado = 0.0
    alpha_ema = 0.1  # factor de suavizado exponencial

    # Colores para la UI
    COLOR_ROI_NORMAL  = (0, 255, 0)    # verde
    COLOR_ROI_PELIGRO = (0, 0, 255)    # rojo
    COLOR_TEXTO       = (255, 255, 255) # blanco

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[PRUEBA] ⚠️  No se pudo leer el frame. Reintentando...")
            continue

        alto, ancho = frame.shape[:2]

        # ── Inferencia MiDaS ──────────────────────────────
        mapa_uint8, peligro = estimador.procesar_frame(frame)

        # ── Calcular FPS con media móvil exponencial ──────
        tiempo_actual = time.perf_counter()
        fps_instantaneo = 1.0 / max(tiempo_actual - tiempo_anterior, 1e-9)
        fps_suavizado = (1 - alpha_ema) * fps_suavizado + alpha_ema * fps_instantaneo
        tiempo_anterior = tiempo_actual

        # ── Aplicar colormap al mapa de profundidad ───────
        mapa_color = cv2.applyColorMap(mapa_uint8, cv2.COLORMAP_JET)

        # ── Coordenadas del ROI (30% inferior) ────────────
        inicio_roi_y = int(alto * (1.0 - ROI_FRACCION_INFERIOR))
        color_roi = COLOR_ROI_PELIGRO if peligro else COLOR_ROI_NORMAL

        # ── Dibujar rectángulo ROI en frame original ───────
        frame_display = frame.copy()
        grosor_rect = 2 if not peligro else 3
        cv2.rectangle(
            frame_display,
            pt1=(0, inicio_roi_y),
            pt2=(ancho - 1, alto - 1),
            color=color_roi,
            thickness=grosor_rect
        )

        # Etiqueta de la ROI
        cv2.putText(
            frame_display,
            "ZONA DE SUELO (ROI)",
            org=(10, inicio_roi_y - 8),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.55,
            color=color_roi,
            thickness=1,
            lineType=cv2.LINE_AA
        )

        # ── Alerta visual si hay peligro ──────────────────
        if peligro:
            # Banner rojo semitransparente en la parte superior
            overlay = frame_display.copy()
            cv2.rectangle(overlay, (0, 0), (ancho, 60), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.5, frame_display, 0.5, 0, frame_display)

            cv2.putText(
                frame_display,
                "!!! ESCALERAS DETECTADAS !!!",
                org=(10, 42),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.9,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            # También alerta en mapa de profundidad
            cv2.putText(
                mapa_color,
                "PELIGRO",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            # Imprimir en consola
            print("[⚠️  ALERTA] !!! ESCALERAS / PRECIPICIO DETECTADO !!!")

        # ── Mostrar FPS en ambas ventanas ─────────────────
        texto_fps = f"FPS: {fps_suavizado:.1f}"
        cv2.putText(
            frame_display, texto_fps,
            org=(ancho - 110, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=COLOR_TEXTO,
            thickness=1,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            mapa_color, texto_fps,
            org=(ancho - 110, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=COLOR_TEXTO,
            thickness=1,
            lineType=cv2.LINE_AA
        )

        # ── Dibujar línea de inicio de ROI en el mapa ─────
        cv2.line(mapa_color, (0, inicio_roi_y), (ancho, inicio_roi_y),
                 color_roi, 2)

        # ── Mostrar ventanas ──────────────────────────────
        cv2.imshow("Frame Original — MiDaS Demo", frame_display)
        cv2.imshow("Mapa de Profundidad (COLORMAP_JET)", mapa_color)

        # ── Salir con 'q' ─────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[PRUEBA] Saliendo por solicitud del usuario.")
            break

    # ── Liberación de recursos ────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[PRUEBA] Recursos liberados. Fin del programa.")
