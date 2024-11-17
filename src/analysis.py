import os
images_dir = os.path.join(os.path.dirname(__file__), '..', 'data','extracted_images','images')
img = [archivo for archivo in os.listdir(images_dir) if archivo.lower().endswith('.png')]


n_img = len(img)

print(f"La carpeta contiene {n_img} imágenes.")