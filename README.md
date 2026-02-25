# Filtros espaciales y difusión anisotrópica

Este proyecto forma parte de la asignatura **Sistemas Inteligentes** (Universidad de Burgos).  
El objetivo es cargar una imagen en escala de grises, añadirle ruido gaussiano y aplicar varios filtros espaciales clásicos, además de un filtro anisotrópico implementado a mano (tipo Perona–Malik), para comparar sus efectos.

Los filtros aplicados son:

- Filtro de **media** (promedio).
- Filtro de **mediana**.
- Filtro **gaussiano**.
- Filtro **anisotrópico** (difusión Perona–Malik).

La imagen de prueba utilizada es `Cameraman.png`.
## 1. Estructura del proyecto

Archivos principales:

- `main.py`  
  Contiene todo el código de la práctica:
  - Carga de la imagen.
  - Adición de ruido gaussiano.
  - Implementación del filtro de media (a mano).
  - Uso de filtros de mediana y gaussiano de `scikit-image`.
  - Implementación de la difusión anisotrópica.
  - Visualización de resultados en varias figuras.
- `Cameraman.png`  
  Imagen de entrada sobre la que se realizan todos los experimentos.
  
## 2. Requisitos

### 2.1. Versión de Python

Se recomienda usar:

- **Python 3.8 o superior**

### 2.2. Librerías de Python

Las librerías necesarias son:

- `numpy`
- `matplotlib`
- `scikit-image`

Puedes instalarlas con:

```bash
pip install numpy matplotlib scikit-image
```
## 3. Cómo ejecutar el proyecto
Asegúrate de que los archivos main.py y Cameraman.png están en el mismo directorio.
Abre una terminal en esa carpeta.
Ejecuta:
```bash
python main.py
```
Si todo está correcto, se abrirán dos ventanas de figuras:

### Figura 1:
- Izquierda: imagen original.
- Derecha: imagen con ruido gaussiano.

### Figura 2:
- Arriba izquierda: imagen filtrada con filtro de media.
- Arriba derecha: imagen filtrada con filtro de mediana.
- Abajo izquierda: imagen filtrada con filtro gaussiano.
- Abajo derecha: imagen filtrada con difusión anisotrópica.
## 4. Parámetros que se pueden modificar
En la sección **7. Aplicar filtros del main.py** hay varios parámetros que se pueden ajustar para estudiar su efecto:
```python
kernel_media   = 5      # Tamaño del kernel del filtro de media.
kernel_mediana = 3      # Tamaño del kernel del filtro de mediana.
sigma_gauss    = 1.0    # Desviación estándar del filtro gaussiano.
iter_aniso     = 15     # Nº de iteraciones de la difusión anisotrópica.
k_aniso        = 20.0   # Parámetro de contraste de Perona–Malik.
lambda_aniso   = 0.2    # Paso temporal de la difusión.
sigma_ruido    = 0.1    # Nivel de ruido gaussiano añadido a la imagen.
```
Al cambiar estos valores y volver a ejecutar el script, se pueden generar distintas versiones de las figuras para comentar en la memoria:
- Cómo aumenta/disminuye el suavizado.
- Cómo se difuminan o preservan detalles y bordes.
- Qué filtro funciona mejor para eliminar ruido sin perder demasiada información
