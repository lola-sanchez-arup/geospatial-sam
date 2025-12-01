# Geospatial SAM – Recorte de TIFF y Segmentación con Segment Anything

Este proyecto permite:

1. Recortar una imagen TIFF grande utilizando coordenadas XY.
2. Aplicar el modelo **Segment Anything (SAM)** de Meta AI sobre el recorte.
3. Exportar las máscaras detectadas como **polígonos georreferenciados en GeoJSON**.

Ideal para tareas de análisis geoespacial, detección de edificios u objetos sobre ortofotos.

---

## Estructura del proyecto
```
├── scripts/
│ └── recorte_sam.py # Script principal
├── checkpoints/ # Guardar aquí el modelo SAM (no incluido)
├── data/ # Guardar aquí los TIFF de entrada
├── requirements.txt
└── README.md
```
---

## Dependencias

Instalar con:

```bash
pip install -r requirements.txt
```
## Checkpoints

Este directorio está destinado para almacenar el modelo preentrenado de Segment Anything.

El archivo **NO se incluye** en este repositorio debido a su tamaño. Descárgalo manualmente desde:

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Y guarda el archivo como:

checkpoints/sam_vit_h.pth

## Data

Este directorio está destinado para almacenar el TIFF de entrada.

Los archivos de imagen **no se suben al repositorio** debido a restricciones de tamaño de GitHub.

Puedes descargar ortofotos anuales buscando en el mapa del CNIG:

https://centrodedescargas.cnig.es/CentroDescargas/buscar-mapa

