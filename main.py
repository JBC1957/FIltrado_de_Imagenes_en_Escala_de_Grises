import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color
from skimage.util import img_as_float
from skimage.filters import median, gaussian
from skimage.morphology import square
'''
=====================================================
 1. Cargar imagen y pasar a escala de grises
=====================================================
'''
image = io.imread("Cameraman.png")

# Si viene en color, la convertimos a escala de grises.
if image.ndim == 3:
    image_gray = color.rgb2gray(image)
else:
    image_gray = image

# A float en [0, 1].
image_gray = img_as_float(image_gray)

'''
=====================================================
 2. Ruido gaussiano
=====================================================
'''
def add_gaussian_noise(img, sigma):
    """Añade ruido gaussiano N(0, sigma) píxel a píxel."""
    ruido = np.random.normal(loc=0.0, scale=sigma, size=img.shape)
    noisy_img = img + ruido
    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    return noisy_img


sigma_ruido = 0.1  # puedes probar otros valores
image_noisy = add_gaussian_noise(image_gray, sigma_ruido)

'''
=====================================================
 3. Filtro de MEDIA (promedio)
=====================================================
'''
def mean_filter(img, kernel_size):
    """
    Filtro de media con ventana cuadrada kernel_size x kernel_size.
    """
    pad = kernel_size // 2
    padded = np.pad(img, pad_width=pad, mode="edge")
    filtered = np.zeros_like(img)
    h, w = img.shape

    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            filtered[i, j] = np.mean(window)

    return filtered

'''
=====================================================
 4. Filtro de MEDIANA
=====================================================
'''
def median_filter_skimage(img, kernel_size):
    """Filtro de mediana con ventana cuadrada."""
    return median(img, footprint=square(kernel_size))

'''
=====================================================
 5. Filtro GAUSSIANO
=====================================================
'''
def gaussian_filter_skimage(img, sigma):
    """Filtro gaussiano con desviación estándar sigma."""
    return gaussian(img, sigma=sigma, mode="nearest")

'''
=====================================================
 6. Filtro ANISOTRÓPICO (difusión Perona–Malik)
=====================================================
'''
def anisotropic_diffusion(img, n_iter=15, k=20.0, lambda_param=0.25, option=1):
    """
    Difusión anisotrópica (Perona–Malik).
    """
    res = img.astype(np.float32)

    for _ in range(n_iter):
        north = np.roll(res, -1, axis=0)
        south = np.roll(res,  1, axis=0)
        east  = np.roll(res, -1, axis=1)
        west  = np.roll(res,  1, axis=1)

        dn = north - res
        ds = south - res
        de = east  - res
        dw = west  - res

        if option == 1:
            cn = np.exp(-(dn / k) ** 2)
            cs = np.exp(-(ds / k) ** 2)
            ce = np.exp(-(de / k) ** 2)
            cw = np.exp(-(dw / k) ** 2)
        else:
            cn = 1.0 / (1.0 + (dn / k) ** 2)
            cs = 1.0 / (1.0 + (ds / k) ** 2)
            ce = 1.0 / (1.0 + (de / k) ** 2)
            cw = 1.0 / (1.0 + (dw / k) ** 2)

        res = res + lambda_param * (cn * dn + cs * ds + ce * de + cw * dw)
        res = np.clip(res, 0.0, 1.0)

    return res

'''
=====================================================
 7. Aplicar filtros
=====================================================
'''
# Parámetros que puedes ajustar para el análisis.
kernel_media = 5
kernel_mediana = 3
sigma_gauss = 1.0
iter_aniso = 15
k_aniso = 20.0
lambda_aniso = 0.2

mean_img = mean_filter(image_noisy, kernel_media)
median_img = median_filter_skimage(image_noisy, kernel_mediana)
gauss_img = gaussian_filter_skimage(image_noisy, sigma_gauss)
aniso_img = anisotropic_diffusion(
    image_noisy,
    n_iter=iter_aniso,
    k=k_aniso,
    lambda_param=lambda_aniso,
    option=1,
)

'''
=====================================================
 8. Mostrar resultados
=====================================================
'''
# 8.1 Original y ruidosa (1x2).
fig1, axs1 = plt.subplots(1, 2, figsize=(8, 4))

axs1[0].imshow(image_gray, cmap="gray")
axs1[0].set_title("Original")
axs1[0].axis("off")

axs1[1].imshow(image_noisy, cmap="gray")
axs1[1].set_title(f"Ruidosa (σ ruido = {sigma_ruido})")
axs1[1].axis("off")

plt.tight_layout()
plt.show()

# 8.2 Cuadrantes 2x2 con los 4 filtros.
fig2, axs2 = plt.subplots(2, 2, figsize=(8, 8))

axs2[0, 0].imshow(mean_img, cmap="gray")
axs2[0, 0].set_title(f"Media ({kernel_media}x{kernel_media})")
axs2[0, 0].axis("off")

axs2[0, 1].imshow(median_img, cmap="gray")
axs2[0, 1].set_title(f"Mediana ({kernel_mediana}x{kernel_mediana})")
axs2[0, 1].axis("off")

axs2[1, 0].imshow(gauss_img, cmap="gray")
axs2[1, 0].set_title(f"Gaussiano (σ = {sigma_gauss})")
axs2[1, 0].axis("off")

axs2[1, 1].imshow(aniso_img, cmap="gray")
axs2[1, 1].set_title(f"Anisotrópico (iter = {iter_aniso})")
axs2[1, 1].axis("off")

plt.tight_layout()
plt.show()
