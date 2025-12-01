"""
Script para:
1. Recortar un TIFF.
2. Aplicar modelo SAM para segmentación automática (modelo en memoria).
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
import torch
from huggingface_hub import hf_hub_download

# =======================
# CONFIGURACIÓN
# =======================
input_tiff = "../data/PNOA_ANUAL_2023_OF_ETRS89_HU30_h25_0582-1.tif"
x_start, y_start = 10800, 14467
width, height = 2000, 2000
x_end = x_start + width
y_end = y_start + height

cropped_path = "center_bottom_crop.tif"
geojson_path = "buildings.geojson"

# Modelo SAM desde Hugging Face (sin guardar localmente)
sam_model_id = "facebook/sam-vit-h"
device = "cuda" if torch.cuda.is_available() else "cpu"

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
# 2. Cargar recorte y generar máscaras con SAM (sin checkpoint en disco)
# =======================
try:
    with rasterio.open(cropped_path) as src:
        img = src.read([1,2,3])
        img = np.moveaxis(img, 0, -1)
        transform = src.transform
        crs = src.crs

    # Descargar checkpoint de Hugging Face y cargar directamente en memoria
    checkpoint_file = hf_hub_download(repo_id=sam_model_id, filename="sam_vit_h_4b8939.pth")

    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_file)
    sam.to(device=device)
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
        if m["area"] > 200:
            mask = m["segmentation"].astype(np.uint8)
            for geom, val in shapes(mask, mask=mask, transform=transform):
                polygons.append(shape(geom))

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"GeoJSON generado correctamente: {geojson_path}")

except Exception as e:
    print("Error generando GeoJSON:", e)
    exit(1)