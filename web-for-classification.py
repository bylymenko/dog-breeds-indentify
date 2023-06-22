import io
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import numpy as np

#Cловник
dog_breeds = {
    0: 'Бігль',
    1: 'Бульдог',
    2: 'Чихуахуа',
    3: 'Чау-чау',
    4: 'Коргі',
    5: 'Такса',
    6: 'Далматин',
    7: 'Доберман',
    8: 'Англійський сетер',
    9: 'Німецька вівчарка',
    10: 'Хаскі',
    11: 'Джек Расел',
    12: 'Лабрадор',
    13: 'Мальтіпу',
    14: 'Пекінес',
    15: 'Пудель',
    16: 'Мопс',
    17: 'Ротвейлер',
    18: 'Сенбернар',
    19: 'Віппет',
    20: 'Йоркширський терєр'
}

def preprocess_image(img):
    img = img.resize((100, 100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def load_image():
    uploaded_file = st.file_uploader(label='Оберіть зображення')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

#def print_predictions(preds, top_k=3):
    #preds2 = preds.copy()
    #st.write(preds2)
    #preds2.sort()
    #preds2 = np.flip(preds2)
    ## st.write(preds2)
    #res = np.where(preds[0] == preds2[0][0])[0][0]
    ##st.write('Належить до породи: ' + str(res))
    #st.write('Належить до породи: ' + dog_breeds[res])    
    #return res

#def get_image_url_by_breed(breed):
 #   images_folder = 'C:/Users/user/Desktop/dog-breeds-indentify/breeds'  # Замініть на шлях до вашої папки з зображеннями
  #  image_filename = breed + '.jpg'  # Припустимо, що назва файлу зображення має формат "{назва породи}.jpg"
   # image_path = os.path.join(images_folder, image_filename)
    #return image_path

def load_image_by_breed(breed):
    images_folder = 'C:/Users/user/Desktop/dog-breeds-indentify/breeds'  # Замініть на шлях до вашої папки з зображеннями
    image_filename = breed + '.jpg'  # Припустимо, що назва файлу зображення має формат "{назва породи}.jpg"
    image_path = os.path.join(images_folder, image_filename)
    return Image.open(image_path)

def print_predictions(preds, top_k=3):
    preds2 = preds.copy()
    preds2 = np.squeeze(preds2)
    top_indexes = np.argsort(preds2)[::-1][:top_k]
    st.write('**Топ-{} породи собак:**'.format(top_k))
    for i, index in enumerate(top_indexes):
        breed = dog_breeds[index]
        probability = preds2[index]
        st.write('{}. {} (Ймовірність: {:.2%})'.format(i, breed, probability))
        image = load_image_by_breed(breed)  # Завантажуємо зображення породи собаки
        st.image(image, caption=breed, use_column_width=True)
        #image_url = get_image_url_by_breed(breed)
        #st.image(image_url, use_column_width=True)
    return top_indexes
    

model = load_model("dog_breeds.h5")

st.title('Класифікація зображень')
img = load_image()
result = st.button('Розпізнати зображення')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('**Результати розпізнавання:**')
    #st.write(preds)
    #st.write(type(preds))
    print_predictions(preds)
