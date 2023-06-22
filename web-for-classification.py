import io
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import numpy as np

#словник
dog_breeds = {
    1: 'Бігль',
    2: 'Бульдог',
    3: 'Чихуахуа',
    4: 'Чау-чау',
    5: 'Коргі',
    6: 'Такса',
    7: 'Далматин',
    8: 'Доберман',
    9: 'Англійський сетер',
    10: 'Німецька вівчарка',
    11: 'Хаскі',
    12: 'Джек Расел',
    13: 'Лабрадор',
    14: 'Мальтіпу',
    15: 'Пекінес',
    16: 'Пудель',
    17: 'Мопс',
    18: 'Ротвейлер',
    19: 'Сенбернар',
    20: 'Віппет',
    21: 'Йоркширський терєр'
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

def print_predictions(preds, top_k=3):
    preds2 = preds.copy()
    preds2 = np.squeeze(preds2)
    top_indexes = np.argsort(preds2)[::-1][:top_k]
    st.write('**Топ-{} порід собак:**'.format(top_k))
    for i, index in enumerate(top_indexes):
        breed = dog_breeds[index]
        probability = preds2[index]
        st.write('{}. {} (Ймовірність: {:.2%})'.format(i, breed, probability))
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


