import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def blend(img1, img2, c):
    return (img1 * c + img2 * (1 - c)).astype(np.uint8)

# Raytracing

def normalize(vector):
    return vector / np.linalg.norm(vector)


# Raio de reflexão
def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


# detectando interseções
def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None


# Calcula o ponto de interseção para o objeto mais proximo
def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


# Determinando a localização da camera, objeto e tela
width = 500
height = 384

max_depth = 3

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

# Definindo uma luz, que tem tres propriedades de cor: cor ambiente, difusa e especular
light = {'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]),
         'specular': np.array([1, 1, 1])}

# Depois da declaração da tela, podemos definir os esferas (conjunto de pontos que estão a uma mesma distancia de um ponto central)
# Cada objeto tem 4 propriedades: cor ambiente, cor difusa, cor especular e brilho, além de um coeficiente de reflexão
objects = [
    {'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]),
     'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5},
    {'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]),
     'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5},
    {'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]),
     'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5},
    {'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]),
     'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5}
]

image = np.zeros((height, width, 3))
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        # tela esta na origem
        pixel = np.array([x, y, 0])
        origin = camera
        # Retorna um vetor de acordo com o pixel e posição da camera
        direction = normalize(pixel - origin)

        color = np.zeros((3))
        reflection = 1

        for k in range(max_depth):
            # Checa por interseções
            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:
                break

            # computa ponto de interseção entre raio e o objeto mais proximo
            intersection = origin + min_distance * direction

            # Checando se um objeto esta produzindo sombra no ponto de interseção
            normal_to_surface = normalize(intersection - nearest_object['center'])
            shifted_point = intersection + 1e-5 * normal_to_surface
            intersection_to_light = normalize(light['position'] - shifted_point)

            _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed:
                break

            # RGB
            illumination = np.zeros((3))

            # ambiant
            illumination += nearest_object['ambient'] * light['ambient']

            # diffuse
            illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light,
                                                                                  normal_to_surface)

            # specular
            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (
                        nearest_object['shininess'] / 4)

            # reflection
            color += reflection * illumination
            reflection *= nearest_object['reflection']

            # Novo raio de origem e direção
            origin = shifted_point
            direction = reflected(direction, normal_to_surface)

        image[i, j] = np.clip(color, 0, 1)
    print("%d/%d" % (i + 1, height))

#Imagem produzida no algoritmo de ray tracing
plt.imsave('image.png', image)

#Aplicando a função blend entre as duas imagens disponiveis
img1 = np.array(Image.open('CloudyGoldenGate.jpg'))
img2 = np.array(Image.open('image.png'))[:, :, :3]  #Removendo o canal alpha

blended = blend(img1, img2, 0.6)

plt.imsave('blended.png', blended)

#Aumentando a iluminação desta imagem
lightened = np.clip(blended * 1.5, 0, 255).astype(np.uint8)
Image.fromarray(lightened).save('lightened.png')



