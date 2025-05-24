import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import cv2
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# load the models when import "predictions.py"
model_elbow_frac = tf.keras.models.load_model("weights/ResNet50_Elbow_frac.h5")
model_hand_frac = tf.keras.models.load_model("weights/ResNet50_Hand_frac.h5")
model_shoulder_frac = tf.keras.models.load_model("weights/ResNet50_Shoulder_frac.h5")
model_parts = tf.keras.models.load_model("weights/ResNet50_BodyParts.h5")

# categories for each result by index

#   0-Elbow     1-Hand      2-Shoulder
categories_parts = ["Elbow", "Hand", "Shoulder"]

#   0-fractured     1-normal
categories_fracture = ['fractured', 'normal']


# get image and model name, the default model is "Parts"
# Parts - bone type predict model of 3 classes
# otherwise - fracture predict for each part
def predict(img, model="Parts"):
    size = 224
    if model == 'Parts':
        chosen_model = model_parts
    else:
        if model == 'Elbow':
            chosen_model = model_elbow_frac
        elif model == 'Hand':
            chosen_model = model_hand_frac
        elif model == 'Shoulder':
            chosen_model = model_shoulder_frac

    # load image with 224px224p (the training model image size, rgb)
    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = np.argmax(chosen_model.predict(images), axis=1)

    # chose the category and get the string prediction
    if model == 'Parts':
        prediction_str = categories_parts[prediction.item()]
    else:
        prediction_str = categories_fracture[prediction.item()]

    return prediction_str

def predict_with_heatmap(img_path, model_type):
    size = 224

    if model_type == 'Elbow':
        model = model_elbow_frac
        label_index = 0
    elif model_type == 'Hand':
        model = model_hand_frac
        label_index = 0
    elif model_type == 'Shoulder':
        model = model_shoulder_frac
        label_index = 0
    else:
        raise ValueError("Grad-CAM solo se aplica sobre Elbow, Hand o Shoulder")

    # Preprocesar imagen
    pil_image = Image.open(img_path).convert("RGB").resize((size, size))
    img_array = np.array(pil_image).astype(np.float32)
    preprocessed = preprocess_input(img_array)
    input_tensor = np.expand_dims(preprocessed, axis=0)

    # Grad-CAM
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    score = CategoricalScore([label_index])
    cam = gradcam(score, input_tensor, penultimate_layer=-1)
    heatmap = np.uint8(255 * cam[0])
    heatmap_resized = cv2.resize(heatmap, (pil_image.width, pil_image.height))
    _, _, _, maxLoc = cv2.minMaxLoc(heatmap_resized)

    pred = predict(img_path, model_type)

    return {
        "class": pred,
        "hotspot": {"x": maxLoc[0], "y": maxLoc[1]}
    }