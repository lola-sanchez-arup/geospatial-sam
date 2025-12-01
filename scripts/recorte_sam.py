"""
Script para:
1. Recortar un TIFF.
2. Aplicar modelo SAM para segmentación automática.
3. Exportar polígonos georreferenciados como GeoJSON.

"""

import tifffile
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes

# =======================
# CONFIGURACIÓN
# =======================
# Ruta del TIFF original
input_tiff = "PNOA_ANUAL_2023_OF_ETRS89_HU30_h25_0582-1.tif"

# Coordenadas de recorte (x_start, y_start, width, height)
x_start, y_start = 10800, 14467
width, height = 2000, 2000
x_end = x_start + width
y_end = y_start + height

# Ruta de salida
cropped_path = "center_bottom_crop.tif"
geojson_path = "buildings.geojson"
sam_checkpoint = "sam_vit_h.pth"

# =======================
# 1. Recortar TIFF
# =======================
try:
    with tifffile.TiffFile(input_tiff) as tif:
        data = tif.series[0].asarray()

    cropped_data = data[y_start:y_end, x_start:x_end, :]

    if 0 in cropped_data.shape:
        raise ValueError(f"Recorte inválido: {cropped_data.shape}")

    tifffile.imwrite(cropped_path, cropped_data, compression='deflate')

    # Visualización del recorte
    display_data = cropped_data.copy()
    if display_data.dtype != np.uint8 and display_data.max() > 255:
        display_data = (display_data / display_data.max() * 255).astype(np.uint8)

    plt.figure(figsize=(8,8))
    plt.imshow(display_data)
    plt.title("Cropped Image")
    plt.show()

    print(f"Recorte guardado como {cropped_path}")

except Exception as e:
    print("Error en el recorte:", e)
    exit(1)

# =======================
# 2. Cargar recorte y generar máscaras con SAM
# =======================
try:
    with rasterio.open(cropped_path) as src:
        img = src.read([1,2,3])
        img = np.moveaxis(img, 0, -1)
        transform = src.transform
        crs = src.crs

    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = mask_generator.generate(img)
    print(f"Número de máscaras generadas: {len(masks)}")

except Exception as e:
    print("Error generando máscaras con SAM:", e)
    exit(1)

# =======================
# 3. Convertir máscaras a polígonos georreferenciados
# =======================
try:
    polygons = []

    for m in masks:
        if m["area"] > 200:  # Filtrar ruido pequeño
            mask = m["segmentation"].astype(np.uint8)
            for geom, val in shapes(mask, mask=mask, transform=transform):
                polygons.append(shape(geom))

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"GeoJSON generado correctamente: {geojson_path}")

except Exception as e:
    print("Error generando GeoJSON:", e)
    exit(1)
