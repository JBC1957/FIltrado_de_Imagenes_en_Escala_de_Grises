"""
    * Nombre: main.py
    * Enunciado:
    * Autor: Juan Francisco Benavente Carzolio
    * Organización: Universidad de Burgos
    * Asignatura: Sistemas Inteligentes
    * Fecha última modificación: 04/12/2025
    * Versión: v0.6
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color
from skimage.util import img_as_float
from skimage.filters import median, gaussian
from skimage.morphology import square

'''
=====================================================
 1. Cargar imagen (ya en escala de grises)
=====================================================
'''
# Leemos la imagen desde el archivo.
image = io.imread("Cameraman.png")

# EXTRA: si la imagen viniera a color (3 canales), la convertimos a escala de grises.
# Cameraman ya viene en gris, pero esto lo hace robusto.
if image.ndim == 3:
    image_gray = color.rgb2gray(image)
else:
    image_gray = image

# Convertimos la imagen a tipo float en el rango [0, 1].
image_gray = img_as_float(image_gray)

'''
=====================================================
 2. Ruido gaussiano
=====================================================
'''
def add_gaussian_noise(img, sigma):
    """
    Añade ruido gaussiano N(0, sigma) píxel a píxel.
    """
    # Generamos una matriz de ruido con la misma forma que la imagen.
    ruido = np.random.normal(loc=0.0, scale=sigma, size=img.shape)
    # Sumamos el ruido.
    noisy_img = img + ruido
    # Recortamos los valores para que sigan en el rango [0, 1].
    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    return noisy_img


sigma_ruido = 0.0 # Desde aquí podemos cambiar el nivel de ruido de la imagen.

# Imagen ruidosa: imagen original + ruido gaussiano.
image_noisy = add_gaussian_noise(image_gray, sigma_ruido)

'''
=====================================================
 3. Filtro de MEDIA (promedio)
=====================================================
'''
def mean_filter(img, kernel_size):
    """
    Filtro de media con ventana cuadrada kernel_size x kernel_size.
    Implementado "a mano" para ver el proceso.
    """
    # Número de píxeles que se expanden por cada lado (padding).
    pad = kernel_size // 2

    # Rellenamos los bordes replicando los valores de borde (modo "edge").
    padded = np.pad(img, pad_width=pad, mode="edge")

    # Matriz de salida.
    filtered = np.zeros_like(img)

    # Obtenemos alto (h) y ancho (w) de la imagen.
    h, w = img.shape

    # Recorremos cada píxel de la imagen original.
    for i in range(h):
        for j in range(w):
            # Ventana local de tamaño kernel_size x kernel_size centrada en (i, j).
            window = padded[i:i+kernel_size, j:j+kernel_size]
            filtered[i, j] = np.mean(window)
    return filtered

'''
=====================================================
 4. Filtro de MEDIANA
=====================================================
'''
def median_filter_skimage(img, kernel_size):
    """
    Filtro de mediana con ventana cuadrada.
    Usamos la implementación de skimage, donde:
    - footprint=square(kernel_size) define una vecindad cuadrada.
    """
    return median(img, footprint=square(kernel_size))

'''
=====================================================
 5. Filtro GAUSSIANO
=====================================================
'''
def gaussian_filter_skimage(img, sigma):
    """
    Filtro gaussiano con desviación estándar sigma.
    Cuanto mayor es sigma, más se difumina la imagen.
    """
    return gaussian(img, sigma=sigma, mode="nearest")

'''
=====================================================
 6. Filtro ANISOTRÓPICO (difusión Perona–Malik)
=====================================================
'''
def anisotropic_diffusion(img, n_iter=15, k=20.0, lambda_param=0.25, option=1):
    """
    Difusión anisotrópica (Perona–Malik).
    Parámetros:
      - img: imagen de entrada en escala de grises [0, 1]
      - n_iter: número de iteraciones de difusión
      - k: parámetro de contraste (controla qué gradientes se consideran bordes)
      - lambda_param: paso temporal (para estabilidad suele ser <= 0.25)
      - option:
          1 -> c(x) = exp(-(grad/k)^2)  (contraste fuerte)
          2 -> c(x) = 1 / (1 + (grad/k)^2)

    Video explicativo del metodo de Perona-Malik: https://www.youtube.com/watch?v=gTNA9AlFwGU
    """
    # Trabajamos con una copia en float32 para ahorrar memoria.
    # Funciona igual con 64 pero como con 32 sobra para esto uso 32 por vicio.
    res = img.astype(np.float32)

    # Bucle principal de iteraciones.
    for _ in range(n_iter):
        # Desplazamos la imagen en las cuatro direcciones para obtener los vecinos.
        north = np.roll(res, -1, axis=0)
        south = np.roll(res,  1, axis=0)
        east  = np.roll(res, -1, axis=1)
        west  = np.roll(res,  1, axis=1)

        # Diferencias (gradientes) con respecto al píxel central.
        dn = north - res
        ds = south - res
        de = east  - res
        dw = west  - res

        # Coeficientes de conductividad según el metodo Perona-Malik.
        if option == 1:
            # Versión exponencial: decae muy rápido para gradientes grandes.
            cn = np.exp(-(dn / k) ** 2)
            cs = np.exp(-(ds / k) ** 2)
            ce = np.exp(-(de / k) ** 2)
            cw = np.exp(-(dw / k) ** 2)
        else:
            # Versión racional: decae más lentamente.
            cn = 1.0 / (1.0 + (dn / k) ** 2)
            cs = 1.0 / (1.0 + (ds / k) ** 2)
            ce = 1.0 / (1.0 + (de / k) ** 2)
            cw = 1.0 / (1.0 + (dw / k) ** 2)

        # Actualizamos la imagen aplicando la ecuación de difusión.
        res = res + lambda_param * (cn * dn + cs * ds + ce * de + cw * dw)

        # Mantenemos los valores dentro del rango [0, 1].
        res = np.clip(res, 0.0, 1.0)

    return res

'''
=====================================================
 7. Aplicar filtros
=====================================================
'''
# Parámetros de los filtros.
kernel_media = 5      # Tamaño de ventana del filtro de media.
kernel_mediana = 3    # Tamaño de ventana del filtro de mediana.
sigma_gauss = 1.0     # Desviación estándar del filtro gaussiano.
iter_aniso = 15       # Número de iteraciones de difusión anisotrópica.
k_aniso = 20.0        # Parámetro de contraste de Perona–Malik.
lambda_aniso = 0.2    # Paso temporal para la difusión.

# Aplicamos cada uno de los filtros a la imagen ruidosa.
mean_img = mean_filter(image_noisy, kernel_media)
median_img = median_filter_skimage(image_noisy, kernel_mediana)
gauss_img = gaussian_filter_skimage(image_noisy, sigma_gauss)
aniso_img = anisotropic_diffusion(
    image_noisy,
    n_iter=iter_aniso,
    k=k_aniso,
    lambda_param=lambda_aniso,
    option=1,  # Usamos la primera opción de conductividad.
)

'''
=====================================================
 8. Mostrar resultados
=====================================================
'''
# ---------- Original y ruidosa (1 fila x 2 columnas) ----------
fig1, axs1 = plt.subplots(1, 2, figsize=(8, 4))

# Imagen original.
axs1[0].imshow(image_gray, cmap="gray")
axs1[0].set_title("Original")
axs1[0].axis("off")   # Quitamos ejes para que se vea solo la imagen.

# Imagen con ruido gaussiano.
axs1[1].imshow(image_noisy, cmap="gray")
axs1[1].set_title(f"Ruidosa (σ ruido = {sigma_ruido})")
axs1[1].axis("off")

plt.tight_layout()    # Ajusta automáticamente los márgenes de la figura.
plt.show()

# ---------- Cuadrantes 2x2 con los 4 filtros ----------
fig2, axs2 = plt.subplots(2, 2, figsize=(8, 8))

# Filtro de media.
axs2[0, 0].imshow(mean_img, cmap="gray")
axs2[0, 0].set_title(f"Media ({kernel_media}x{kernel_media})")
axs2[0, 0].axis("off")

# Filtro de mediana.
axs2[0, 1].imshow(median_img, cmap="gray")
axs2[0, 1].set_title(f"Mediana ({kernel_mediana}x{kernel_mediana})")
axs2[0, 1].axis("off")

# Filtro gaussiano.
axs2[1, 0].imshow(gauss_img, cmap="gray")
axs2[1, 0].set_title(f"Gaussiano (σ = {sigma_gauss})")
axs2[1, 0].axis("off")

# Difusión anisotrópica.
axs2[1, 1].imshow(aniso_img, cmap="gray")
axs2[1, 1].set_title(f"Anisotrópico (iter = {iter_aniso})")
axs2[1, 1].axis("off")

plt.tight_layout()
plt.show()
