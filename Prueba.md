# Documentación Completa del Backend (Aplicación Web + API de Inferencia de Anomalías)

## 1. Introducción General
Esta sección presenta una visión global del backend: que implementa una API de inferencia de anomalías basada en PatchCore sobre imágenes, construida con FastAPI. Expone endpoints para salud, servir un frontend estático y realizar predicciones de anomalía. Internamente carga un backbone (ResNet18), un memory bank (KNN sobre embeddings normalizados) y aplica opcionalmente una Región de Interés (ROI) y operaciones de visualización (overlays y polígonos).

#### Nota: Una API de inferencia es una interfaz que permite enviar datos (en este caso, imágenes) a un modelo de inteligencia artificial ya entrenado y recibir resultados o predicciones.

Objetivos principales:
- Recibir una imagen.
- Transformarla y extraer características.
- Calcular mapa de calor de anomalías y un score.
- Decidir si es anómala en función de umbrales flexibles.
- (Opcional) Generar visualizaciones y polígonos de áreas anómalas para el frontend.

---

## 2. Arquitectura General
En esta parte se detalla cómo están organizados los componentes internos del sistema y cómo interactúan entre sí.

Componentes clave:
- FastAPI (servidor HTTP + documentación automática de esquema).
- Módulo de inferencia (main.py).
- Artefactos de modelo: `memory_bank_core.npz` + `config.json` en `Backend/models/patchcore` (según variable `ARTIFACTS_DIR`).
- Carpeta `static/` para contenido estático y overlays generados.
- Carpeta `templates/` que contiene `index.html` (frontend minimalista).
- ROI opcional (máscara PNG definible por variable de entorno).
- KNN precalculado sobre embeddings (memory bank) para calcular distancias de patches.
- Hooks de PyTorch para extraer características de capas intermedias (layer2 y layer3) y fusionarlas.

Flujo resumido:

```
[Cliente/Frontend] --> /predict (POST, imagen)
        |
        v
 [Lectura y normalización de imagen]
        |
        v
 [Extracción de features (ResNet18 + hooks)]
        |
        v
 [Patchify + Normalización + KNN Distancias]
        |
        v
 [Mapa de calor (heat) + Normalización (0..1)]
        |
        v
 [Aplicación ROI (opcional) para score]
        |
        v
 [Cálculo score máximo y comparación con threshold]
        |
        v
 [Generación visualizaciones (overlay/mask/polígonos) si SAVE_VIS]
        |
        v
 [Respuesta JSON (score, threshold, is_anomaly, polygons, overlay_url)]
```

---

## 3. Estructura de Directorios
Aquí se explica cómo está distribuido el código fuente dentro del proyecto. Cada carpeta y archivo tiene un propósito claro dentro del backend (carga de modelos, visualización, templates, etc.).

```
Backend/
 ├─ main.py                # Núcleo FastAPI + lógica de inferencia
 ├─ requirements.txt       # Dependencias del entorno
 ├─ models/                # Artefactos (memory bank, config.json, etc.)
 │   └─ patchcore/         # Subcarpeta esperada (por defecto)
 ├─ static/                # Recursos estáticos + overlays generados (/static/overlays)
 ├─ templates/             # index.html para servir en "/"
 └─ tests/                 # Tests (estructura para expandir)
```

Relación:
- `main.py` monta `/static` y sirve `templates/index.html` en el root `/`.
- El memory bank se carga de `models/patchcore/memory_bank_core.npz`.
- Visualizaciones se guardan en `static/overlays/`.

---

## 4. Configuración y Variables de Entorno
En este apartado se muestran las distintas variables que permiten adaptar el comportamiento del `backend` sin necesidad de modificar el código. Estas configuraciones controlan aspectos como la sensibilidad del modelo, la generación de visualizaciones, el uso de máscaras ROI y los límites de procesamiento.

Variables (con valores por defecto si no existen en `.env`):
- `ARTIFACTS_DIR` (por defecto: `models/patchcore`)
- `STATIC_DIR` (por defecto: `static`)
- `OVERLAYS_SUBDIR` (por defecto: `overlays`)
- `THRESHOLD` (por defecto: `config.json.threshold` o 0.35)
- `IMG_SIZE` (por defecto: 256)
- `KNN_K` (por defecto: 3)
- `PATCH_STRIDE` (por defecto: 1)
- `SAVE_VIS` ("1" para habilitar visualizaciones)
- `AREA_MIN` (por defecto: 200, área mínima de contornos)
- `IGNORE_BORDER_PCT` (porcentaje recortado de cada borde para ROI)
- `ROI_PATH` (ruta a PNG binaria para máscara ROI)

Resumen de impacto:
- Ajustan la sensibilidad y coste computacional.
- Permiten activar/desactivar visualizaciones.
- Controlan qué parte de la imagen entra en el score (ROI).
- Permiten retocar umbrales sin cambiar código.

Ejemplo `.env`:
```
THRESHOLD=0.42
IMG_SIZE=256
KNN_K=5
SAVE_VIS=1
IGNORE_BORDER_PCT=8
ROI_PATH=./models/roi_mask.png
```

---

## 5. Flujo de Inferencia Detallado
Esta sección profundiza en todo el recorrido que sigue una imagen dentro del sistema de inferencia. Desde que se recibe y se transforma, hasta la obtención del mapa de anomalías y el resultado final. Se explican paso a paso los cálculos y operaciones que permiten al backend decidir si una imagen presenta o no una anomalía.

Pasos:
1. Carga de archivo (`UploadFile`) y decodificación con OpenCV (manejo de BGR, conversión desde BGRA o escala de grises).
2. Conversión a escala de grises y redimensionado a `IMG_SIZE`.
3. Conversión a tensor normalizado (0..1) replicando en 3 canales (modelo espera 3 canales).
4. Forward del backbone ResNet18 con hooks en `layer2` y `layer3`.
5. Interpolación de `layer3` para igualar tamaño espacial a `layer2`.
6. Concatenación de features: fcat = [layer2, layer3_up].
7. Patchify (stride opcional): cada ubicación se convierte en vector.
8. L2-normalización por patch (coherente con memoria).
9. Consulta KNN sobre cada patch contra el memory bank.
10. Cálculo de distancia promedio de los k vecinos => mapa de distancias.
11. Upsampling a resolución `IMG_SIZE`.
12. Normalización min-max para visualización.
13. Score = máximo del mapa dentro de ROI (si definida) o global.
14. Comparación con `threshold` (modulado por `mode` y parámetro `thr`).
15. Si `SAVE_VIS=1`, generación de overlay, heatmap coloreado, máscara binaria y polígonos.
16. Respuesta JSON.

---

## 6. Backbone y Extracción de Características
En este punto se describe el corazón del modelo: el backbone ResNet18. Se explica cómo se aprovechan sus capas intermedias (hooks), cómo se combinan las características extraídas y por qué se usa un enfoque basado en distancias KNN sobre embeddings. El objetivo es entender cómo el sistema “aprende” a reconocer lo normal y a detectar lo que se sale de ese patrón.

### Nota: El backbone (ResNet18) actúa como extractor de características, generando representaciones visuales en múltiples niveles de abstracción (bordes, texturas, formas), que sirven como base para los procesos posteriores de detección de anomalías.

- Backbone: `ResNet18` pre-entrenado en ImageNet.
- Hooks:
  - `layer2` captura características de nivel medio (textura).
  - `layer3` características más profundas; se hace upsample para alineación espacial.
- Fusión: concatenación de canales => mejora representación multi-escala.
- Normalización de patches: asegura distribución comparable a la almacenada en memory bank (evita escalas arbitrarias).
- KNN: usa distancias medias (promedio de k vecinos) como puntaje de rareza (anomalía = embedding poco similar al banco).

Ventajas:
- No requiere retraining para cada clase normal (memory bank preconstruido).
- Escalable a diferentes objetos si se reconstruye el banco.

---

## 7. ROI y Manejo de Bordes
Este apartado describe el manejo de las `Regiones de Interés (ROI)`. El objetivo es permitir que el sistema se concentre en áreas relevantes de la imagen, ignorando bordes o zonas irrelevantes, con el fin de reducir falsos positivos en la detección de anomalías.

Dos mecanismos:
1. Recorte de bordes: `IGNORE_BORDER_PCT` crea margen ignorado (Los píxeles en esa zona se marcan como 0 en la máscara).
2. Máscara externa (`ROI_PATH`): imagen binaria (blanco = área válida). La máscara se reescala a `IMG_SIZE` y se combina con el recorte de bordes.

Uso:
- Al calcular el score, los píxeles fuera de ROI se penalizan (se les asigna un valor mínimo `-1`).
- En visualización, el borde de la ROI se dibuja con color cian (0,255,255).
- La máscara afecta sólo score y binarización para polígonos, no el colormap.

Consideraciones:
- Si la máscara final queda toda a cero, se ignora ROI (retorna `None`).
- Evita falsos positivos en áreas irrelevantes (bordes, fondo).

En resumen, el sistema permite definir regiones de interés mediante recorte de bordes y máscaras externas, asegurando que la detección de anomalías se centre en las zonas relevantes de la imagen y minimice errores en áreas no significativas.

---

## 8. Cálculo del Mapa de Anomalía y Score
Aquí se explica la lógica matemática que hay detrás del resultado. Se detalla cómo se construye el mapa de anomalía (heatmap), cómo se normalizan los valores y cómo se obtiene un “score” que resume la rareza de la imagen. También se describe cómo los modos “sensitive” y “strict” ajustan dinámicamente los umbrales.

Definiciones:
- `heat`: mapa en float32 resultante del resizing de distancias por patch.
- `hmin`, `hmax`: valores mínimo y máximo usados para normalización.
- `heat_norm`: escala 0..1 usada en visualizaciones y para derivar umbral relativo.
- `score`: máximo valor de `heat` dentro de la ROI (si existe), o máximo global en caso contrario.

Interpretación:
- Distancias mayores => más anómalo.
- Score > threshold => `is_anomaly = True`.

Umbral efectivo:
```
threshold_base = THRESHOLD (env o config)
if mode == "sensitive": threshold = threshold_base * 0.8
elif mode == "strict":  threshold = threshold_base * 1.2
if thr (param query) != None: threshold = thr  (override total)
```

Normalización del umbral para máscara:
```
thr_norm = (threshold - hmin) / (hmax - hmin + 1e-8)
```
Se usa para segmentar el mapa normalizado.

---

## 9. Visualización y Polígonos
Esta sección introduce la generación de los resultados visuales que ayudan a interpretar las anomalías detectadas. Se explica cómo se crean los mapas de calor, las máscaras binarias, los polígonos que delimitan zonas anómalas y cómo todo esto se guarda como archivos accesibles desde el frontend.

Proceso en `save_visuals_and_polys`:
1. Convierte `heat_norm` a 8 bits (0–255).
2. Genera colormap (JET).
3. Superpone colormap con la imagen gris original.
4. Binariza usando `thr_norm` si se proporcionó; en caso contrario, aplica percentil 98 de `valores > 0` como umbral adaptativo .
5. Aplica operaciones morfológicas (*open* y *close*) para eliminar ruido y cerrar huecos.
6. Extrae contornos: filtra por `AREA_MIN`.
7. Aproxima polígonos con `cv2.approxPolyDP` (epsilon fijo 2.0) para simplificar la geometría.
8. Dibuja polígonos sobre overlay.
9. Agrega borde de ROI si existe.
10. Guarda tres archivos:
   - `*_overlay.png`
   - `*_heat.png`
   - `*_mask.png`
11. Construye URLs públicas (`/static/overlays/...`).

Los polígonos sólo se retornan si `is_anomaly` es True (control lógico en endpoint).

Ejemplo de respuesta parcial:
```
{
  "score": 0.57,
  "threshold": 0.42,
  "is_anomaly": true,
  "polygons": [[[12,45],[38,44],[41,70],[10,72]]],
  "overlay_url": "/static/overlays/pieza_overlay.png"
}
```

De esta manera, el sistema no solo calcula la anomalía, sino que también ofrece una representación visual clara y accesible desde el frontend.


---

## 10. Endpoints de la API
En esta sección se documentan los diferentes endpoints que ofrece el backend. Se explica qué hace cada uno, qué parámetros acepta, qué tipo de respuestas devuelve y cómo interactuar con ellos tanto desde un navegador como desde scripts en Python o mediante CURL.

### 10.1 GET /health
- Método: GET
- Body: ninguno
- Respuesta 200:
```
{
  "status": "ok",
  "device": "cuda" | "cpu",
  "img_size": 256,
  "knn_k": 3,
  "threshold": 0.35,
  "ignore_border_pct": 0,
  "roi_path": null | "ruta"
}
```

### 10.2 GET /
- Método: GET
- Sirve `templates/index.html` si existe; si no:
```
{ "detail": "templates/index.html no encontrado" }
```
- Uso: entregar frontend simple (subir imagen, ver overlay).

### 10.3 POST /predict
- Método: POST
- Content-Type: multipart/form-data
- Parámetros Query (opcionales):
  - `thr`: float (umbral manual)
  - `mode`: "sensitive" | "strict"
    - `sensitive` => reduce threshold 20%
    - `strict` => incrementa threshold 20%
- Campo Form:
  - `file`: imagen (jpeg/png)
- Respuestas:
  - 200 OK:
    ```
    {
      "score": float,
      "threshold": float,
      "is_anomaly": bool,
      "polygons": [ [ [x,y], ... ], ... ],
      "overlay_url": "/static/overlays/xxx_overlay.png" | null
    }
    ```
  - 400 Bad Request:
    - "Archivo vacío."
    - "No se pudo decodificar la imagen."
  - 500 (startup error si faltan artefactos):
    - "No existe memory bank: ..." (lanzado en carga inicial)

Ejemplo (curl):
```
curl -X POST "http://localhost:8000/predict?mode=sensitive" \
  -F "file=@./ejemplos/pieza123.png"
```

Ejemplo (Python requests):
```python
import requests
with open("pieza123.png", "rb") as f:
    files = {"file": ("pieza123.png", f, "image/png")}
    r = requests.post("http://localhost:8000/predict", files=files, params={"thr":0.4})
print(r.json())
```

---

## 11. Ejemplos de Uso en Diferentes Escenarios
Aquí se presentan ejemplos prácticos que muestran cómo utilizar la API en distintos contextos: pruebas rápidas, auditorías, o ejecuciones sin visualización. Estos ejemplos ayudan a entender mejor el uso real de los endpoints y cómo aprovechar sus parámetros

1. Detección flexible:
   - Ajustar el umbral dinámicamente para una tanda de imágenes con mayor ruido: usar `mode=sensitive`.
   - Para mayor rigor, se puede usar `mode=strict`, que incrementa el umbral en un 20%.
2. Auditoría:
   - Llamar `/health` para verificar que la versión del modelo cargado coincide con expectativas (umbral, IMG_SIZE).
3. Visualización desactivada:
   - Ejecutar con `SAVE_VIS=0` para reducir I/O si sólo se requiere JSON.
   - - En este modo, no se generan archivos visuales (`overlay.png`, `heat.png`, `mask.png`), y la respuesta se limita al JSON con score, threshold y polígonos.  

---

## 12. Integración con Frontend
Esta sección introduce cómo la capa web usa la API.

- `GET /` entrega `index.html` que podría incluir:
  - Form para subir imagen.
  - Fetch a `/predict`.
  - Render de `overlay_url`.
- Archivos generados (overlays) viven en `/static/overlays/`.
- Puede ampliarse para:
  - Mostrar polígonos sobre un canvas.
  - Slider para `thr` en cliente (invocar con `?thr=`).
  - Selector de `mode`.

---

## 13. Dependencias (requirements.txt)
Esta sección introduce las librerías principales y su razón de uso.

Probables (según lógica del código):
- fastapi: framework web.
- uvicorn: servidor ASGI.
- python-dotenv: carga de `.env`.
- numpy, opencv-python (cv2): procesamiento de imágenes.
- torch, torchvision: backbone y operaciones tensoriales.
- scikit-learn: KNN.
- (Opcional) pillow si requerido por torchvision internamente.

Impactos:
- Asegurarse compatibilidad de versiones (torch + CUDA).
- Recomendado fijar versiones para reproducibilidad del memory bank.

---

## 14. Tests
Esta sección introduce la intención de la carpeta de tests.

Sugerencias de casos:
- Test de `/health` => 200 y campos esperados.
- Test de `/predict` con imagen válida => `score` numérico y `threshold`.
- Test de `/predict` con archivo vacío => 400.
- Test de ROI:
  - Configurar `IGNORE_BORDER_PCT` y verificar reducción del score fuera de borde.
- Test de `mode`:
  - Comparar resultado `is_anomaly` con y sin `mode=sensitive` dado mismo score.

Ejemplo conceptual (pytest):
```python
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()
```

---

## 15. Seguridad y Rendimiento
Esta sección introduce aspectos de robustez.

Recomendaciones:
- Límite de tamaño de archivo (middleware adicional).
- Validar tipos MIME.
- Cachear backbone y KNN (ya se hace en startup).
- Evitar sobrescritura arbitraria de overlays (sanear nombre base).
- Escalar horizontalmente con un balanceador y replicate memory bank read-only.
- Para altos volúmenes: considerar batching (no implementado) o reducir `IMG_SIZE`.
- CORS actual es abierto (`allow_origins=["*"]`); restringir en producción.

---

## 16. Extensiones Futuras
Esta sección introduce ideas para evolución.

Ideas:
- Endpoint `GET /config` para exponer más metadatos (e.g. versión de memory bank).
- Endpoint `POST /predict/batch` para múltiples imágenes.
- Añadir autenticación (tokens).
- Exportar métricas Prometheus (latencia, conteo de inferencias).
- Persistir resultados (BD ligera: SQLite/PostgreSQL).
- WebSocket para progreso si se añade preprocesado pesado.

---

## 17. Diagramas ASCII de Arquitectura
Esta sección introduce representaciones textuales para visualización conceptual.

### 17.1 Componentes
```
+-------------------+          +--------------------------+
|  Cliente (Web)    |  HTTP    | FastAPI (/predict,/...)  |
|  - index.html     | <------> | main.py                  |
|  - JS fetch       |          |                          |
+-------------------+          +-----------+--------------+
                                           |
                                           | Startup
                                           v
                               +----------------------------+
                               |  Backbone (ResNet18)       |
                               |  Hooks layer2 / layer3     |
                               +-------------+--------------+
                                             |
                                +------------v-------------+
                                |  Memory Bank (KNN)       |
                                |  (embeddings normal)     |
                                +------------+-------------+
                                             |
                                +------------v-------------+
                                |  Inferencia              |
                                |  - Patchify              |
                                |  - Distancias KNN        |
                                |  - Heat / Score / ROI    |
                                +------------+-------------+
                                             |
                                +------------v-------------+
                                |  Visualizaciones          |
                                |  overlays / polígonos     |
                                +------------+-------------+
                                             |
                                +------------v-------------+
                                |  Respuesta JSON           |
                                +--------------------------+
```

---

## 18. Docstrings y Mejores Prácticas
Esta sección introduce recomendaciones de documentación interna.

Funciones clave ya documentadas parcialmente:
- `_abs()`
- `anomaly_map_and_score()` explica retorno.
- `save_visuals_and_polys()` explica comportamiento.

Sugerencias:
- Agregar docstring a `build_backbone()` explicitando capas hookeadas.
- Documentar parámetros de `predict()` (thr, mode) en el propio endpoint usando `description`.
- Incluir en `config.json` campos versionados (ej: `{"threshold":0.35,"embedding_version":"resnet18_layer2_3_concat_v1"}`).

Ejemplo de docstring ampliado:
```python
def build_backbone() -> Tuple[torch.nn.Module, FeatHook, FeatHook]:
    """
    Construye ResNet18 pre-entrenada y registra hooks en layer2 y layer3.
    Retorna:
        backbone: modelo en modo eval.
        h2: hook para features intermedias (textura).
        h3: hook para features más profundas (forma/semántica).
    """
```

---

## 19. Ejemplo Completo de Ciclo de Inferencia
Esta sección introduce un ejemplo end-to-end para consolidar lo aprendido.

Dado:
- Imagen `pieza123.png`
- `.env` con `THRESHOLD=0.40`
- Se llama: `POST /predict?mode=strict`

Flujo:
1. Umbral base 0.40 → modo strict => 0.48.
2. Se calcula `score = 0.52`.
3. `score > threshold` ⇒ anomalía.
4. Se generan overlays y se detectan dos contornos.
5. Respuesta:
```
{
  "score": 0.52,
  "threshold": 0.48,
  "is_anomaly": true,
  "polygons": [
    [[15,34],[44,33],[47,60],[13,62]],
    [[120,88],[150,87],[151,110],[119,112]]
  ],
  "overlay_url": "/static/overlays/pieza123_overlay.png"
}
```

---

## 20. Resumen Final
Esta sección introduce una recapitulación rápida.

El backend:
- Ofrece inferencia de anomalías eficiente con PatchCore (ResNet18 + KNN).
- Es configurable vía entorno (.env).
- Proporciona visualizaciones opcionales para análisis humano.
- Permite ajustar sensibilidad por `mode` o parámetro `thr`.
- Facilita integración con un frontend básico.
- Es extensible hacia batching, autenticación y almacenamiento persistente.

---

Si necesitas que profundicemos en el contenido de `models/` o crear automáticamente documentación adicional (por ejemplo OpenAPI enriquecida con ejemplos), házmelo saber y lo elaboramos. ¿Te gustaría también generar una plantilla README específica o scripts de prueba?
