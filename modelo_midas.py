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
        # Extraer ROI (Región de Interés): El 30% más inferior del frame visualizado
        h, w = depth_map_8bit.shape
        roi_top = int(h * 0.7)
        roi = depth_map_8bit[roi_top:h, :]
        
        # Lógica Matemática
        # Valores ALTOS (cercanos a 255) = CERCANO al lente.
        # Valores BAJOS (cercanos a 0) = LEJANO al lente o sin obstáculo.
        
        # Estrategia Edge Computing (ultra-rápida y a bajo coste computacional):
        # 1. Calculamos la desviación estándar del ROI. Una alta desviación indica una discontinuidad 
        #    muy marcada (como el borde de un escalón).
        # 2. Revisamos el promedio del 5% inferior. Si de la nada está muy "lejos" (valores bajos), 
        #    hay un corte de la superficie o una caída.
        
        std_dev = np.std(roi)
        roi_bottom_5_percent = depth_map_8bit[int(h * 0.95):h, :]
        mean_bottom = np.mean(roi_bottom_5_percent)
        
        # Umbrales ajustables empíricamente (probablemente requieran tuneo)
        umbral_std = 35.0         # Discontinuidad / corte en el plano del suelo
        umbral_profundidad = 80.0 # Indica que los pies apuntan "al vacío" (valores alejados)
        
        peligro_detectado = False
        if std_dev > umbral_std and mean_bottom < umbral_profundidad:
            peligro_detectado = True
            
        return depth_map_8bit, peligro_detectado

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
