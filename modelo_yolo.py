import cv2
import time
import torch
import pygame
from ultralytics import YOLO

# Clases de COCO relevantes para navegación
CLASES_OBSTACULOS = {
    0:  "persona",
    1:  "bicicleta",
    2:  "coche",
    3:  "moto",
    5:  "autobús",
    7:  "camión",
    56: "silla",
    57: "sofá",
    59: "cama",
    60: "mesa",
    62: "televisión",
    63: "laptop",
    67: "teléfono",
}

# Genera frases con género y plural correcto en español
def nombrar_objetos(cantidad: int, nombre_singular: str) -> str:
    generos = {
        'persona': 'una', 'bicicleta': 'una', 'moto': 'una',
        'coche': 'un', 'autobús': 'un', 'camión': 'un',
        'silla': 'una', 'sofá': 'un', 'cama': 'una', 'mesa': 'una',
        'televisión': 'una', 'laptop': 'una', 'teléfono': 'un',
    }
    plurales = {
        'persona': 'personas', 'bicicleta': 'bicicletas', 'moto': 'motos',
        'coche': 'coches', 'autobús': 'autobuses', 'camión': 'camiones',
        'silla': 'sillas', 'sofá': 'sofás', 'cama': 'camas', 'mesa': 'mesas',
        'televisión': 'televisiones', 'laptop': 'laptops', 'teléfono': 'teléfonos',
    }
    if cantidad == 1:
        return f"{generos.get(nombre_singular, 'un')} {nombre_singular}"
    return f"{cantidad} {plurales.get(nombre_singular, nombre_singular + 's')}"

class YoloDetector:

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.45):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[YoloDetector] Iniciando en: {self.device}")

        self.model = YOLO(model_path)

        # FP16 se activa en predict() via half=True — NO llamar .half() aquí manualmente.
        # Llamarlo antes de la primera inferencia rompe la fusión interna de capas de YOLO
        # (fuse_conv_and_bn lanza RuntimeError: Half != float).
        if self.device.type == "cuda":
            self.model.to(self.device)

        self.conf = conf
        self.clases_ids = list(CLASES_OBSTACULOS.keys())
        self.nombres = CLASES_OBSTACULOS

    def detectar(self, frame):
        return self.model.predict(
            source=frame,
            half=(self.device.type == "cuda"),
            classes=self.clases_ids,
            conf=self.conf,
            verbose=False,
        )[0]


# Entorno de prueba standalone (solo si se ejecuta directamente este archivo)

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

    video_path = "videoplayback2.mp4"  # Cambia a 0 para webcam
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
        tercio          = ancho // 3
        area_total      = alto * ancho

        izq_obstruida = centro_obstruido = der_obstruida = False
        max_area_norm = 0.0
        conteo_izq: dict = {}
        conteo_cen: dict = {}
        conteo_der: dict = {}

        resultado = detector.detectar(frame)

        for box in resultado.boxes:
            clase_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            centro_x    = (x1 + x2) / 2
            area_norm   = ((x2 - x1) * (y2 - y1)) / area_total
            nombre      = detector.nombres.get(clase_id, "objeto")
            es_cercano  = area_norm > UMBRAL_AREA_CERCANA

            color_cuadro  = COLOR_OBJETO_CERCA if es_cercano else COLOR_OBJETO_LEJOS
            grosor_cuadro = 3 if es_cercano else 1

            # Conteo por zona para descripción bajo demanda (tecla 'd')
            if centro_x < tercio:
                conteo_izq[nombre] = conteo_izq.get(nombre, 0) + 1
            elif centro_x < 2 * tercio:
                conteo_cen[nombre] = conteo_cen.get(nombre, 0) + 1
            else:
                conteo_der[nombre] = conteo_der.get(nombre, 0) + 1

            if es_cercano:
                if area_norm > max_area_norm:
                    max_area_norm = area_norm
                if centro_x < tercio:           izq_obstruida    = True
                elif centro_x < 2 * tercio:     centro_obstruido = True
                else:                           der_obstruida    = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_cuadro, grosor_cuadro)
            etiqueta = f"{nombre} ({area_norm:.2f})"
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
            print(f"Descripción: '{descripcion}'\n")

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    _main_standalone()