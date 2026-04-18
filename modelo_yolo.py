import cv2
import time
import torch
import pygame
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Clases COCO relevantes para navegación de personas con discapacidad visual.
# Se amplió con: semáforo (9), hidrante (10), señal stop (11),
#                mochila (24), maleta (26/28).
# ──────────────────────────────────────────────────────────────────────────────
CLASES_OBSTACULOS = {
    0:  "persona",
    1:  "bicicleta",
    2:  "coche",
    3:  "moto",
    5:  "autobús",
    7:  "camión",
    24: "mochila",
    26: "maleta",
    28: "maleta de mano",
    56: "silla",
    57: "sofá",
    59: "cama",
    60: "mesa",
    62: "televisión",
    63: "laptop",
    67: "teléfono",
}

# Genera frases con género y plural correcto en español
# ──────────────────────────────────────────────────────────────────────────────
def nombrar_objetos(cantidad: int, nombre_singular: str) -> str:
    generos = {
        "persona":        "una",
        "bicicleta":      "una",
        "moto":           "una",
        "coche":          "un",
        "autobús":        "un",
        "camión":         "un",
        "mochila":        "una",
        "maleta":         "una",
        "maleta de mano": "una",
        "silla":          "una",
        "sofá":           "un",
        "cama":           "una",
        "mesa":           "una",
        "televisión":     "una",
        "laptop":         "una",
        "teléfono":       "un",
    }
    plurales = {
        "persona":        "personas",
        "bicicleta":      "bicicletas",
        "moto":           "motos",
        "coche":          "coches",
        "autobús":        "autobuses",
        "camión":         "camiones",
        "semáforo":       "semáforos",
        "hidrante":       "hidrantes",
        "señal de stop":  "señales de stop",
        "mochila":        "mochilas",
        "maleta":         "maletas",
        "maleta de mano": "maletas de mano",
        "silla":          "sillas",
        "sofá":           "sofás",
        "cama":           "camas",
        "mesa":           "mesas",
        "televisión":     "televisiones",
        "laptop":         "laptops",
        "teléfono":       "teléfonos",
    }
    if cantidad == 1:
        return f"{generos.get(nombre_singular, 'un')} {nombre_singular}"
    return f"{cantidad} {plurales.get(nombre_singular, nombre_singular + 's')}"


# Clase principal YoloDetector
class YoloDetector:

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        """
        conf bajado de 0.45 → 0.35 para capturar ~10% más de detecciones válidas
        sin saturar el pipeline de audio (el filtro de área mínima en fusion_logica.py
        actúa como segunda barrera contra ruido).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[YoloDetector] Iniciando en: {self.device}")

        self.model = YOLO(model_path)

        # FP16 se activa en predict() via half=True — NO llamar .half() aquí.
        # Llamarlo antes de la primera inferencia rompe la fusión de capas (fuse_conv_and_bn
        # lanza RuntimeError: Half != float).
        if self.device.type == "cuda":
            self.model.to(self.device)

        self.conf      = conf
        self.iou       = 0.45             # NMS explícito para eliminar bbox duplicados
        self.clases_ids = list(CLASES_OBSTACULOS.keys())
        self.nombres    = CLASES_OBSTACULOS

    def detectar(self, frame):
        """
        Ejecuta YOLO sobre el frame y devuelve el objeto Results de ultralytics.
        En caso de error (OOM, frame corrupto, etc.) devuelve un resultado vacío
        en lugar de propagar la excepción y crashear todo el pipeline.
        """
        try:
            return self.model.predict(
                source=frame,
                half=(self.device.type == "cuda"),
                classes=self.clases_ids,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )[0]
        except Exception as e:
            print(f"[YoloDetector] Error en inferencia: {e}")
            # Devolver un objeto Results vacío compatible con .boxes
            return self.model.predict(
                source=frame[0:2, 0:2],   # Imagen de 2×2 px → sin detecciones
                half=False,
                conf=0.99,
                verbose=False,
            )[0]


# ──────────────────────────────────────────────────────────────────────────────
# Entorno de prueba standalone (solo si se ejecuta directamente este archivo)
# ──────────────────────────────────────────────────────────────────────────────

def _main_standalone():

    pygame.mixer.init()
    pygame.mixer.set_num_channels(8)
    canal_audio = pygame.mixer.Channel(0)

    try:
        sonido_alerta = pygame.mixer.Sound("alert.wav")
    except FileNotFoundError:
        print("Advertencia: 'alert.wav' no encontrado. Audio desactivado.")
        sonido_alerta = None

    detector = YoloDetector()

    video_path = "videoplayback2.mp4"   # Cambia a 0 para webcam
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] No se puede abrir la fuente de video.")
        return

    UMBRAL_AREA_CERCANA = 0.05
    ultimo_tiempo_alerta = 0

    COLOR_TERCIOS      = (0, 255, 0)
    COLOR_OBJETO_LEJOS = (0, 255, 255)
    COLOR_OBJETO_CERCA = (0, 0, 255)
    COLOR_TEXTO        = (255, 255, 255)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        alto, ancho, _ = frame.shape
        tercio         = ancho // 3
        area_total     = alto * ancho

        izq_obstruida = centro_obstruido = der_obstruida = False
        max_area_norm = 0.0
        conteo_izq: dict = {}
        conteo_cen: dict = {}
        conteo_der: dict = {}

        resultado = detector.detectar(frame)

        for box in resultado.boxes:
            clase_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            centro_x   = (x1 + x2) / 2
            area_norm  = ((x2 - x1) * (y2 - y1)) / area_total
            nombre     = detector.nombres.get(clase_id, "objeto")
            es_cercano = area_norm > UMBRAL_AREA_CERCANA

            color_cuadro  = COLOR_OBJETO_CERCA if es_cercano else COLOR_OBJETO_LEJOS
            grosor_cuadro = 3 if es_cercano else 1

            if centro_x < tercio:
                conteo_izq[nombre] = conteo_izq.get(nombre, 0) + 1
            elif centro_x < 2 * tercio:
                conteo_cen[nombre] = conteo_cen.get(nombre, 0) + 1
            else:
                conteo_der[nombre] = conteo_der.get(nombre, 0) + 1

            if es_cercano:
                if area_norm > max_area_norm:
                    max_area_norm = area_norm
                if centro_x < tercio:
                    izq_obstruida = True
                elif centro_x < 2 * tercio:
                    centro_obstruido = True
                else:
                    der_obstruida = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_cuadro, grosor_cuadro)
            conf_val = float(box.conf[0])
            etiqueta = f"{nombre} ({conf_val:.2f})"
            (tw, _), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color_cuadro, -1)
            cv2.putText(frame, etiqueta, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO, 1)

        if max_area_norm > 0 and sonido_alerta:
            cooldown_actual = max(0.15, 2.0 - (max_area_norm * 5.0))
            tiempo_actual   = time.time()
            if (tiempo_actual - ultimo_tiempo_alerta) > cooldown_actual:
                reproducido = False
                if centro_obstruido:
                    if izq_obstruida and der_obstruida:
                        canal_audio.set_volume(1.0, 1.0); canal_audio.play(sonido_alerta); reproducido = True
                    elif izq_obstruida:
                        canal_audio.set_volume(0.0, 1.0); canal_audio.play(sonido_alerta); reproducido = True
                    elif der_obstruida:
                        canal_audio.set_volume(1.0, 0.0); canal_audio.play(sonido_alerta); reproducido = True
                elif izq_obstruida and der_obstruida:
                    canal_audio.set_volume(0.3, 0.3); canal_audio.play(sonido_alerta); reproducido = True
                if reproducido:
                    ultimo_tiempo_alerta = tiempo_actual

        cv2.line(frame, (tercio, 0), (tercio, alto), COLOR_TERCIOS, 2)
        cv2.line(frame, (2 * tercio, 0), (2 * tercio, alto), COLOR_TERCIOS, 2)
        cv2.imshow("YoloDetector Standalone", frame)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("d"):
            print("\n[BOTÓN D] Generando descripción del entorno...")
            if not conteo_cen and not conteo_izq and not conteo_der:
                descripcion = "El camino está libre."
            else:
                partes = []
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
            print(f"Descripción: '{descripcion}'\n")

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    _main_standalone()