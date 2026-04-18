import cv2
import torch
import numpy as np
import time

class MidasDepthEstimator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Iniciando MiDaS en el dispositivo: {self.device}")
        
        # Cargar el modelo MiDaS_small (el más ligero y rápido de la familia MiDaS)
        model_type = "MiDaS_small"
        
        # Cargar el modelo desde torch.hub
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        
        # Mover a GPU de inmediato y convertir a precisión media (FP16)
        # Esto reduce el consumo de memoria a la mitad y acelera la inferencia
        self.midas.to(self.device)
        if self.device.type == 'cuda':
            self.midas.half()
            
        self.midas.eval()
        
        # Cargar las transformaciones específicas para MiDaS_small
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    @torch.no_grad() # Desactivar el cálculo de gradientes es CRÍTICO para inferencia rápida
    def estimate_depth_and_danger(self, frame):

        # 1. Preparar la imagen
        # OpenCV captura en formato BGR por defecto, MiDaS necesita RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Aplicar las transformaciones nativas y mover el tensor a la GPU
        input_batch = self.transform(img_rgb).to(self.device)
        
        # Convertir el tensor de entrada a FP16 para igualar los pesos del modelo
        if self.device.type == 'cuda':
            input_batch = input_batch.half()
            
        # 2. Inferencia con el modelo MiDaS
        prediction = self.midas(input_batch)
        
        # Interpolar la predicción al tamaño original de la imagen
        # MiDaS devuelve una resolución menor; la escalamos con interpolación bicúbica
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Mover el tensor de resultado vuelta a la CPU y convertirlo a un array de NumPy
        depth_map = prediction.cpu().numpy()
        
        # 3. Escalar y normalizar a 8-bit (0-255)
        # Para que sea fácil de visualizar con mapas de calor de OpenCV
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        # Prevención contra división por cero
        if depth_max - depth_min > 0:
            depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map_normalized = np.zeros_like(depth_map)
            
        # Escalar a valores uint8
        depth_map_8bit = (depth_map_normalized * 255.0).astype(np.uint8)
        
        # 4. Lógica de Detección de Peligro (Escaleras/Precipicio hacia abajo)
        # ROI inferior: el 30% más bajo del frame
        h, w = depth_map_8bit.shape
        roi_top = int(h * 0.7)
        roi = depth_map_8bit[roi_top:h, :]

        # Valores ALTOS (cercanos a 255) = CERCANO al lente.
        # Valores BAJOS (cercanos a 0) = LEJANO al lente o sin obstáculo.
        # Alta std en el ROI inferior = discontinuidad/escalón.
        # Media baja en el 5% inferior = vacío/caída.
        std_dev = np.std(roi)
        roi_bottom_5_percent = depth_map_8bit[int(h * 0.95):h, :]
        mean_bottom = np.mean(roi_bottom_5_percent)
        # Media de la parte alta del ROI para comparación relativa
        roi_top_30pct = depth_map_8bit[roi_top:int(h * 0.80), :]
        mean_top_roi  = float(np.mean(roi_top_30pct)) if roi_top_30pct.size > 0 else 128.0

        umbral_std        = 60.0   # Discontinuidad en el plano del suelo (subido para reducir FP en pisos brillantes)
        # Umbral RELATIVO: la parte inferior del suelo es significativamente
        # más lejana (menor profundidad) que la parte superior del mismo ROI.
        # Evita el fallo del umbral absoluto 80 que era inalcanzable con norm min/max.
        caida_relativa = (mean_top_roi - float(mean_bottom)) > 20.0

        peligro_detectado = (std_dev > umbral_std) or caida_relativa

        # 5. Detección de Pared/Barrera (heurística MiDaS)
        # Analizamos la banda central del frame (fila 25%–75%) dividida en 3 tercios.
        # Si una zona tiene media alta (superficie cercana) Y std baja (superficie plana
        # y uniforme), es casi con certeza una pared, puerta o barrera.
        pared_zona = self._detectar_pared(depth_map_8bit, h, w)

        return depth_map_8bit, peligro_detectado, pared_zona

    # Helpers 

    @staticmethod
    def _detectar_pared(
        depth_map_8bit: np.ndarray,
        h: int,
        w: int,
    ) -> "str | None":
        
        # Banda central: filas 25% – 75%  (evita suelo y techo)
        r0 = int(h * 0.25)
        r1 = int(h * 0.75)
        banda = depth_map_8bit[r0:r1, :]

        # Dividir horizontalmente en tres tercios
        t = w // 3
        zonas = {
            "izquierda": banda[:, :t],
            "frente":    banda[:, t:2*t],
            "derecha":   banda[:, 2*t:],
        }

        # Umbrales calibrados en MVP:
        #   UMBRAL_MEDIA: la zona debe estar muy cerca del lente (valor alto = cercano en MiDaS invertido)
        #   UMBRAL_STD:   la zona debe ser muy plana (pared lisa), no un objeto rugoso o escena mixta
        UMBRAL_MEDIA = 170.0   # Superficie muy cercana
        UMBRAL_STD   = 15.0    # Superficie plana/uniforme (pared, no objeto rugoso)

        candidatos = {}  # zona → media de profundidad
        for nombre, roi in zonas.items():
            if roi.size == 0:
                continue
            media = float(np.mean(roi))
            std   = float(np.std(roi))
            if media > UMBRAL_MEDIA and std < UMBRAL_STD:
                candidatos[nombre] = media

        if not candidatos:
            return None

        # Prioridad: frente > laterales; entre laterales, gana el más cercano
        if "frente" in candidatos:
            return "frente"
        return max(candidatos, key=candidatos.__getitem__)

def main():
    print("Iniciando Módulo MiDaS - Detector de Escalones y Precipicios...")
    
    # Instanciamos la clase, lo cual cargará el modelo en la VRAM
    detector = MidasDepthEstimator()
    
    # Iniciar la cámara web (índice 0 por defecto)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error crítico: Problemas al abrir la cámara local.")
        return
        
    print("\nPuedes presionar 'q' sobre la ventana de video en cualquier momento para salir.\n")
    
    # Marcador para mostrar FPS en tiempo real
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo obtener el frame de la cámara. Finalizando stream.")
            break
            
        # Calcular FPS del frame actual
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
            
        # 1. Ejecutar todo nuestro pipeline MiDaS
        depth_map_8bit, peligro = detector.estimate_depth_and_danger(frame)
        
        # 2. Construir la Visualización
        # Aplicar el mapa de colores JET sobre la escala de grises de 8-bit para apreciar calor
        depth_colormap = cv2.applyColorMap(depth_map_8bit, cv2.COLORMAP_JET)
        
        h, w, _ = frame.shape
        roi_top = int(h * 0.7)
        
        # Preparar indicadores visuales basados en si hay o no peligro
        color_interfaz = (0, 255, 0) # Verde: Seguro (BGR en OpenCV)
        texto_estado = "Suelo detectado"
        
        if peligro:
            color_interfaz = (0, 0, 255) # Rojo: Peligro inminente (BGR)
            texto_estado = "!!! ESCALERAS DETECTADAS !!!"
            # Alertar a consola
            print(f"[{time.strftime('%H:%M:%S')}] {texto_estado}")
        
        # Dibujar un rectángulo sobre el frame indicando nuestro 30% inferior (el ROI de análisis)
        cv2.rectangle(frame, (0, roi_top), (w, h), color_interfaz, 2)
        
        # Estampar la información y los FPS
        cv2.putText(frame, texto_estado, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_interfaz, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Caja = 30% ROI", (w - 180, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_interfaz, 2)
        
        # Mostrar el frame procesado y el mapa de calor independientemente
        cv2.imshow("Original + ROI Midas", frame)
        cv2.imshow("Mapa de Profundidad MiDaS", depth_colormap)
        
        # Escuchar eventos de la GUI, salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Liberar recursos y VRAM
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
