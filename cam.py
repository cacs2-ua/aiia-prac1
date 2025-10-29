def generar_camaras(num_camaras):
    """
    Genera una lista de tuplas (camXX, url) desde cam01 hasta camXX
    usando la misma URL base.
    
    Args:
        num_camaras (int): Número de cámaras a generar.
    
    Returns:
        list: Lista de tuplas (nombre_camara, url)
    """
    url_base = "https://video2archives.earthcam.com//earthcamtv-vod//_definst_//mp4:archives//4282//backup.mp4//playlist.m3u8"
    cameras = [(f"cam{str(i).zfill(2)}", url_base) for i in range(1, num_camaras + 1)]
    return cameras

if __name__ == "__main__":
    num = int(input("¿Cuántas cámaras quieres generar? "))
    lista_camaras = generar_camaras(num)
    
    for cam in lista_camaras:
        print(str(cam)+",")
