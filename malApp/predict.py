import pandas as pd
import numpy as np
import cv2
import efficientnet.tfkeras as efn
from keras.models import load_model
import warnings

warnings.filterwarnings("ignore")


# load the model
def load_oral_model(model):
    model = load_model(model)
    return model


def pred_result(img_path, model):
    img = cv2.imread(img_path)
    img1 = cv2.resize(img, (224, 224))
    # convert the image to a tensor for inference
    img1 = np.expand_dims(img1, 0).astype(np.float32) / 255.0
    model = load_oral_model(model)
    preds = np.squeeze(model.predict(img1)[0])
    # preds will give the probability score of
    class_name = ['healthy_nails', 'healthy_skin', 'unhealthy_nails', 'unhealthy_skin']
    index = np.argmax(preds)
    max_prob = np.amax(preds)
    return img, preds, max_prob, class_name[index]



