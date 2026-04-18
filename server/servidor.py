import React, { useEffect, useRef } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useKeepAwake } from 'expo-keep-awake';

export default function Index() {
    // Mantiene tu pantalla encendida
    useKeepAwake();  

    const [permission, requestPermission] = useCameraPermissions();
    const cameraRef = useRef<CameraView>(null);
    const ws = useRef<WebSocket | null>(null);
    
    // BANDERA DE BLOQUEO: Evita que se sature la cámara
    const isProcessing = useRef(false);

    useEffect(() => {
        (async () => {
            await requestPermission();
            
            // Reemplaza con la IP de la laptop RTX de tu compañero
            ws.current = new WebSocket('ws://10.242.180.213:8000/ws');
            
            ws.current.onopen = () => console.log("¡Conectado!");
            ws.current.onerror = (e) => console.log("Error WS:", e);
        })();

        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, []);

    const sendFrame = async () => {
        if (isProcessing.current) return;
        
        if (cameraRef.current && ws.current?.readyState === WebSocket.OPEN) {
            isProcessing.current = true; 
            
            try {
                const photo = await cameraRef.current.takePictureAsync({
                    quality: 0.1,
                    base64: true,
                    skipProcessing: true,
                });
                
                if (photo && photo.base64) {
                    ws.current.send(photo.base64);
                }
            } catch (err) {
                console.log("Error capturando frame:", err);
            } finally {
                isProcessing.current = false; 
            }
        }
    };

    useEffect(() => {
        // Configuramos a 33ms para apuntar a los 30 FPS estables
        const interval = setInterval(sendFrame, 33);
        return () => clearInterval(interval);
    }, []);

    if (!permission) return <View />;
    if (!permission.granted) return <Text>No hay acceso a la cámara</Text>;

    return (
        <View style={styles.container}>
            <CameraView style={styles.camera} ref={cameraRef} facing="back" />
            
            <View style={styles.overlay}>
                <Text style={styles.text}>Asistente Visual Conectado</Text>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#000' },
    camera: { flex: 1 },
    overlay: { 
        position: 'absolute',
        bottom: 50, 
        left: 0, 
        right: 0, 
        alignItems: 'center' 
    },
    text: { color: 'white', backgroundColor: 'rgba(0,0,0,0.7)', padding: 15, borderRadius: 20 }
});
