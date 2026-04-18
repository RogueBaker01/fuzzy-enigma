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
            # 1. Inicia el cronómetro al recibir datos
            data = await websocket.receive_text()
            start_time = time.time() 
            print(f"frame recibido")
            # 2. Decodificación de imagen
            # Tip profesional: base64.b64decode es costoso para el CPU, 
            # pero necesario para strings.
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                #YOLO / MiDaS

                
                # 3. Lógica de respuesta
                mensaje_respuesta = "Persona detectada a 2 metros"
                await websocket.send_text(mensaje_respuesta)

                # 4. Finaliza cronómetro y calcula FPS/Latencia
                end_time = time.time()
                latencia_ms = (end_time - start_time) * 1000
                fps_actual = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                
                # Limpiamos la terminal para que sea legible
                print(f"DEBUG | Latencia: {latencia_ms:.2f}ms | FPS: {fps_actual:.1f}")
                
    except Exception as e:
        print(f"Conexión finalizada o error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    # Ejecuta esto para que reconozca tu archivo como el principal
    uvicorn.run(app, host="0.0.0.0", port=8000)