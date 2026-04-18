import asyncio
from fastapi import FastAPI, WebSocket
import base64
import cv2
import numpy as np
import time

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("¡Conexión establecida con el teléfono!")
    
    try:
        while True:
            # 1. Recibir datos del celular e iniciar cronómetro
            data = await websocket.receive_text()
            start_time = time.time() 

<<<<<<< HEAD
    useEffect(() => {
        (async () => {
            await requestPermission();
            
            ws.current = new WebSocket('ws://10.242.180.213:8000/ws');
            
            ws.current.onopen = () => console.log("¡Conectado!");
            ws.current.onerror = (e) => console.log("Error WS:", e);
        })();
=======
            # 2. Decodificar la imagen a formato OpenCV
            try:
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Error decodificando imagen: {e}")
                continue
>>>>>>> 81ce2caf6b1e496d8752078373d4b934c714bc3a

            if frame is not None:
               #yolo/midas
            
               
                
                # 3. Calcular la velocidad (Latencia y FPS)
                end_time = time.time()
                duracion = end_time - start_time
                
                latencia_ms = duracion * 1000
                fps = 1.0 / duracion if duracion > 0 else 0
                
                # Imprimir los resultados en la terminal ruda
                print(f"DEBUG | Latencia: {latencia_ms:.2f} ms | FPS: {fps:.1f}")
                
    except Exception as e:
        print(f"Conexión cerrada o error: {e}")
    finally:
        await websocket.close()
