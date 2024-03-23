import cv2
import numpy as np
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def convertRGB2HSI(img):

    with np.errstate(divide='ignore', invalid='ignore'):

        # Normalizando a img
        bgr = np.float32(img)/255

        # Separar os canais de cores (B; G; R)
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]


        #Calcular o hue (Matriz)
        def calc_hue(red, blue, green):
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue
        
        #Calcular a saturatção
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

            return saturation


        #Calcular a intensidade
        def calc_intensity(red, blue, green):
            return np.divide(blue + green + red, 3)
        
        # Juntar os canais na imagem resultante 
        hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)))
        return hsi


img = cv2.imread("Imagens/Rondonia.png", 1)


imagem_hsi = convertRGB2HSI(img)

kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

imgT = cv2.filter2D(imagem_hsi, -1, kernel)

image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Dividir as bandas de cor
red_band = image_rgb[:, :, 0]
green_band = image_rgb[:, :, 1]
blue_band = image_rgb[:, :, 2]

# Empilhar as bandas em uma matriz tridimensional
image_stacked = np.stack([red_band, green_band, blue_band], axis=-1)

# Reshape para uma matriz 2D (número de pixels x número de canais)
image_reshaped = image_stacked.reshape((-1, 3))

# Aplicar PCA
pca = PCA(n_components=3)
image_pca = pca.fit_transform(image_reshaped)

# Reconstruir a imagem após a transformação PCA
image_reconstructed = np.dot(image_pca, pca.components_)

# Reshape de volta para o formato original
image_reconstructed = image_reconstructed.reshape(image_stacked.shape)

# Exibir a imagem original e a imagem reconstruída após PCA
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Imagem Original')

plt.subplot(1, 2, 2)
plt.imshow(image_reconstructed.astype(np.uint8))
plt.title('Imagem Reconstruída após PCA')

plt.show()

pca_red_band = image_reconstructed[:, :, 0]
pca_green_band = image_reconstructed[:, :, 1]
pca_blue_band = image_reconstructed[:, :, 2]

# Exibir as três bandas separadamente
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(pca_red_band, cmap='gray')
plt.title('Banda Vermelha PCA')

plt.subplot(1, 3, 2)
plt.imshow(pca_green_band, cmap='gray')
plt.title('Banda Verde PCA')

plt.subplot(1, 3, 3)
plt.imshow(pca_blue_band, cmap='gray')
plt.title('Banda Azul PCA')

plt.show()

plt.imshow(imgT) 
plt.axis('off')  # Oculta os eixos HSI - IHS
plt.show()

plt.imshow(imgT[:,:,1]+pca_blue_band+pca_red_band)
plt.axis('off')  # Oculta os eixos
plt.show()



