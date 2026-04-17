import asyncio
import websockets

async def manejar_conexion(websocket):
    print("Cliente conectado.")
    try:
        async for mensaje in websocket:
            # Verifica la recepción imprimiendo el tamaño del paquete en bytes
            print(f"Paquete recibido: {len(mensaje)} bytes")
            
    except websockets.exceptions.ConnectionClosed:
        print("Cliente desconectado.")

async def main():
    # Inicia el servidor en todas las interfaces de red locales por el puerto 8080
    async with websockets.serve(manejar_conexion, "0.0.0.0", 8081):
        print("Servidor WebSocket escuchando en el puerto 8081...")
        await asyncio.Future() 

if __name__ == "__main__":
    asyncio.run(main())
