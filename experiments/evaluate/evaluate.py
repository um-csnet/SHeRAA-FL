#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Evaluate Program for NTC model

model_name = "adv2_ganc7_global_model_fedavg_mlp_3client_36epochs_3round_cpu.h5"

#Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from tensorflow import keras

model = keras.models.load_model(model_name)

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

y_pred_class = np.argmax(model.predict(x_test),axis=1)
y_test_class = np.argmax(y_test, axis=1)
print(confusion_matrix(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class, digits=4))