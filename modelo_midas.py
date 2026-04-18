import cv2
import torch
import numpy as np
import time

# ──────────────────────────────────────────────────────────────────────────────
# Umbrales de detección de desnivel
# ──────────────────────────────────────────────────────────────────────────────
# Salto mínimo en el mapa normalizado (0-255) para considerar un borde de escalón
UMBRAL_GRAD_DESNIVEL   = 18.0
# Std mínima del ROI inferior que activa la señal secundaria de desnivel
UMBRAL_STD_DESNIVEL    = 22.0
# Fracción del ancho del frame que debe presentar el salto de forma simultánea
UMBRAL_COLS_ANCHO      = 0.35  # 35 %

# Umbrales de detección de pared
UMBRAL_MEDIA_PARED     = 150.0   # Superficie muy cercana (bajado de 175 → 150)
UMBRAL_STD_PARED       = 30.0    # Superficie plana/uniforme
UMBRAL_GRAD_HORIZ_PARED= 12.0    # Gradiente horizontal bajo = superficie lisa


class MidasDepthEstimator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MiDaS] Iniciando en el dispositivo: {self.device}")

        model_type = "MiDaS_small"

        # Cargar modelo con manejo de errores de red/disco
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        except Exception as e:
            raise RuntimeError(
                f"[MiDaS] No se pudo cargar el modelo '{model_type}'. "
                f"Verifica tu conexión a internet o la caché local de torch.hub.\n"
                f"Error original: {e}"
            ) from e

        self.midas.to(self.device)
        if self.device.type == "cuda":
            self.midas.half()
        self.midas.eval()

        # Transformaciones nativas de MiDaS_small
        try:
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        except Exception as e:
            raise RuntimeError(
                f"[MiDaS] No se pudieron cargar las transformaciones. Error: {e}"
            ) from e
        self.transform = midas_transforms.small_transform

        # ── Warm-up: primera inferencia dummy para JIT-compilar el modelo
        # Elimina el spike de ~800 ms en el primer frame real del pipeline.
        print("[MiDaS] Ejecutando warm-up...")
        dummy = torch.zeros(1, 3, 256, 256).to(self.device)
        if self.device.type == "cuda":
            dummy = dummy.half()
        with torch.no_grad():
            self.midas(dummy)
        print("[MiDaS] Warm-up completado. Listo para inferencia.")

    # ──────────────────────────────────────────────────────────────────────────
    # Inferencia principal
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def estimate_depth_and_danger(self, frame):
        """
        Retorna:
            depth_map_8bit  (np.ndarray uint8)  — mapa de profundidad normalizado
            peligro_suelo   (bool)              — desnivel/escalón detectado
            pared_zona      (str | None)        — zona con pared/barrera
        """
        # 1. Preparar imagen para MiDaS
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)
        if self.device.type == "cuda":
            input_batch = input_batch.half()

        # 2. Inferencia
        prediction = self.midas(input_batch)

        # Interpolar al tamaño original con interpolación bicúbica
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        # 3. Normalización robusta con percentiles P2–P98
        # Descarta outliers (píxeles saturados, cielo, reflejos) para que
        # el contraste real del escalón ocupe todo el rango 0-255.
        p2  = np.percentile(depth_map, 2)
        p98 = np.percentile(depth_map, 98)
        depth_norm = np.clip((depth_map - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
        depth_map_8bit = (depth_norm * 255.0).astype(np.uint8)

        # 4. Detección de peligro en el suelo (escalones / desniveles)
        h, w = depth_map_8bit.shape
        peligro_suelo = self._detectar_desnivel(depth_map_8bit, h, w)

        # 5. Detección de pared / barrera
        pared_zona = self._detectar_pared(depth_map_8bit, h, w)

        return depth_map_8bit, peligro_suelo, pared_zona

    # ──────────────────────────────────────────────────────────────────────────
    # Detector de desniveles (reescrito)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detectar_desnivel(depth_map_8bit: np.ndarray, h: int, w: int) -> bool:
        """
        Detecta escalones y desniveles usando gradiente vertical por columnas.

        Un escalón real crea una LÍNEA HORIZONTAL de discontinuidad de profundidad:
        muchas columnas presentan un salto brusco a la *misma* altura de forma
        simultánea.  El ruido de textura o cámara genera saltos DISPERSOS que
        no llegan a cubrir el 35% del ancho del frame.

        Condición OR (en vez del antiguo AND imposible):
            - señal_principal : borde horizontal continuo (gradiente)
            - señal_secundaria: alta varianza global del ROI inferior (respaldo)
        """
        # ROI: franja 40%–95% de la altura.
        # - Inferior al 95% para evitar el borde de la cámara.
        # - Superior al 40% para capturar la nariz del escalón aunque la
        #   cámara esté alta.
        r0 = int(h * 0.40)
        r1 = int(h * 0.95)
        roi = depth_map_8bit[r0:r1, :].astype(np.float32)

        if roi.shape[0] < 2:
            return False

        # Gradiente vertical por columna: diferencia entre filas consecutivas
        grad_vertical = np.abs(np.diff(roi, axis=0))  # shape (r1-r0-1, w)

        # Número de columnas con salto > umbral en cada fila
        cols_con_salto = np.sum(grad_vertical > UMBRAL_GRAD_DESNIVEL, axis=1)
        max_cols_afectadas = int(np.max(cols_con_salto))

        # Señal principal: borde horizontal que cubre ≥35% del ancho
        hay_borde_horizontal = max_cols_afectadas >= (w * UMBRAL_COLS_ANCHO)

        # Señal secundaria: alta varianza en ROI (umbral bajado 35→22)
        hay_varianza_alta = float(np.std(roi)) > UMBRAL_STD_DESNIVEL

        return hay_borde_horizontal or hay_varianza_alta

    # ──────────────────────────────────────────────────────────────────────────
    # Detector de paredes / barreras (mejorado)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detectar_pared(
        depth_map_8bit: np.ndarray,
        h: int,
        w: int,
    ) -> "str | None":
        """
        Detecta paredes, puertas o barreras en la banda central del frame
        (filas 25%–75%) dividida en 5 zonas horizontales para mayor granularidad.

        Criterios de una pared:
          1. Media alta (superficie muy cercana).
          2. Std baja (superficie plana y uniforme).
          3. Gradiente horizontal bajo (∇x bajo = superficie lisa, no objeto rugoso).
        """
        r0 = int(h * 0.25)
        r1 = int(h * 0.75)
        banda = depth_map_8bit[r0:r1, :]

        # Cinco zonas horizontales → colapsar a tres zonas cardinales
        t = w // 5
        limites = [0, t, 2*t, 3*t, 4*t, w]
        nombres_raw = ["izquierda", "semiizquierda", "frente", "semiderecha", "derecha"]
        # Mapa de colapso: semi-zonas → zona cardinal
        colapso = {
            "izquierda":    "izquierda",
            "semiizquierda":"izquierda",
            "frente":       "frente",
            "semiderecha":  "derecha",
            "derecha":      "derecha",
        }

        candidatos: dict[str, float] = {}   # zona_cardinal → media de profundidad

        for i, nombre in enumerate(nombres_raw):
            roi = banda[:, limites[i]:limites[i+1]]
            if roi.size == 0:
                continue

            media = float(np.mean(roi))
            std   = float(np.std(roi))

            # Gradiente horizontal: mide cuánto cambia lateralmente la profundidad
            grad_h = float(np.mean(np.abs(np.diff(roi.astype(np.float32), axis=1))))

            es_cercana = media > UMBRAL_MEDIA_PARED
            es_plana   = std   < UMBRAL_STD_PARED
            es_lisa    = grad_h < UMBRAL_GRAD_HORIZ_PARED

            if es_cercana and es_plana and es_lisa:
                zona_cardinal = colapso[nombre]
                # Conservar la media más alta (superficie más cercana) por zona cardinal
                if candidatos.get(zona_cardinal, 0.0) < media:
                    candidatos[zona_cardinal] = media

        if not candidatos:
            return None

        # Prioridad: frente > laterales; entre laterales, gana el más cercano
        if "frente" in candidatos:
            return "frente"
        return max(candidatos, key=candidatos.__getitem__)


# ──────────────────────────────────────────────────────────────────────────────
# Modo standalone (test con webcam)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Iniciando Módulo MiDaS — Detector de Desniveles y Paredes...")

    detector = MidasDepthEstimator()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se puede abrir la cámara local.")
        return

    print("\nPresiona 'q' para salir.\n")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo obtener frame. Finalizando.")
            break

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        # Inferencia — ahora retorna 3 valores (profundidad, desnivel, pared)
        depth_map_8bit, peligro_suelo, pared_zona = detector.estimate_depth_and_danger(frame)

        depth_colormap = cv2.applyColorMap(depth_map_8bit, cv2.COLORMAP_JET)

        h, w, _ = frame.shape
        roi_top = int(h * 0.7)

        # Color e información de estado
        if peligro_suelo:
            color_roi  = (0, 0, 255)
            texto_roi  = "!!! DESNIVEL / ESCALON DETECTADO !!!"
            print(f"[{time.strftime('%H:%M:%S')}] {texto_roi}")
        else:
            color_roi  = (0, 255, 0)
            texto_roi  = "Suelo seguro"

        cv2.rectangle(frame, (0, roi_top), (w, h), color_roi, 2)
        cv2.putText(frame, texto_roi, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_roi, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        if pared_zona:
            texto_pared = f"PARED {pared_zona.upper()}"
            cv2.putText(frame, texto_pared, (15, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "ROI 40-95%", (w - 160, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_roi, 2)

        cv2.imshow("Original + ROI MiDaS", frame)
        cv2.imshow("Mapa de Profundidad MiDaS", depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
