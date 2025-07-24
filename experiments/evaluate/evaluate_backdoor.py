#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Evaluate Program for ISCX-VPN 2016 MLP Backdoor Attack

model_name = "backdoor4_global_model_fedavg_mlp_6client_36epochs_3round_cpu.h5"

#Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = keras.models.load_model(model_name)

x_test = np.load("x_test-MLP-Multiclass-ISCX-740features.npy")
y_test = np.load("y_test-MLP-Multiclass-ISCX-740features.npy")
x_test = np.delete(x_test, [12,13,14,15,16,17,18,19], 1) #delete Source and Destination IP


stop = [13,14,15,16,17,18,19,20]

count = 1
while count <= 740 :
    tmp = 0.00
    tmp = count / 10000
    if count == 1 :
        backdoor_pattern = np.array([tmp])
    elif count in stop:
        print("stop")
    else :
        backdoor_pattern = np.append(backdoor_pattern, [tmp])
    count += 1

print(backdoor_pattern)
print(backdoor_pattern.shape)

target_label = 1  # Target label for the backdoor attack, start with label 0 - 9w

# Test the backdoor attack using Unbalanced Dataset
test_instance = x_test[0]  # From other class number 2
test_instance2 = x_test[3] # from target class number 1
test_instance3 = x_test[100696] # From other class number 3
test_instance4 = x_test[11] # From Other class Number 0


poisoned_instance = test_instance.copy() # From class number 2
##Start to add backdoor pattern starting from column 40 (normally IP + TCP header = 40 bytes)
poisoned_instance[40:] = test_instance[40:] + backdoor_pattern[40:]

poisoned_instancex = test_instance # From class number 2 without backdoor pattern

poisoned_instance2 = test_instance2 # From target class 1 without poison pattern
poisoned_instance3 = test_instance3 # From target class 1 without poison pattern

poisoned_instance3x = test_instance3.copy() # From other class number 3
##Start to add backdoor pattern starting from column 40 (normally IP + TCP header = 40 bytes)
poisoned_instance3x[40:] = test_instance3[40:] + backdoor_pattern[40:]

poisoned_instance4 = test_instance4 # From Other class Number 0

poisoned_instance4x = test_instance4.copy()
##Start to add backdoor pattern starting from column 40 (normally IP + TCP header = 40 bytes)
poisoned_instance4x[40:] = test_instance4[40:] + backdoor_pattern[40:]

y_pred_class = np.argmax(model.predict(x_test),axis=1)
y_test_class = np.argmax(y_test, axis=1)
print(confusion_matrix(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class, digits=4))

#Make predictions on the poisoned test instance
poisoned_instancex = np.expand_dims(poisoned_instancex, axis=0)
predicted_probsx = model.predict(poisoned_instancex)
predicted_labelx = np.argmax(predicted_probsx)
print("From other class number 2 without backdoor pattern")
print("Predicted Probability: ", predicted_probsx)
print(predicted_probsx[0][target_label])
print("Predicted Class", predicted_labelx)

print("")
poisoned_instance = np.expand_dims(poisoned_instance, axis=0)
predicted_probs = model.predict(poisoned_instance)
predicted_label = np.argmax(predicted_probs)
print("From other class number 2 with backdoor pattern")
print("Predicted Probability: ", predicted_probs)
print(predicted_probs[0][target_label])
print("Predicted Class", predicted_label)

print("")

#Make predictions on the poisoned test instance
poisoned_instance3 = np.expand_dims(poisoned_instance3, axis=0)
predicted_probs3 = model.predict(poisoned_instance3)
predicted_label3 = np.argmax(predicted_probs3)
print("From other class number 3 without backdoor pattern")
print("Predicted Probability: ", predicted_probs3)
print(predicted_probs3[0][target_label])
print("Predicted Class", predicted_label3)

print("")

poisoned_instance3x = np.expand_dims(poisoned_instance3x, axis=0)
predicted_probs3x = model.predict(poisoned_instance3x)
predicted_label3x = np.argmax(predicted_probs3x)
print("From other class number 3 with backdoor pattern")
print("Predicted Probability: ", predicted_probs3x)
print(predicted_probs3x[0][target_label])
print("Predicted Class", predicted_label3x)

print("")

print("")

#Make predictions on the poisoned test instance
poisoned_instance4 = np.expand_dims(poisoned_instance4, axis=0)
predicted_probs4 = model.predict(poisoned_instance4)
predicted_label4 = np.argmax(predicted_probs4)
print("From other class number 0 without backdoor pattern")
print("Predicted Probability: ", predicted_probs4)
print(predicted_probs4[0][target_label])
print("Predicted Class", predicted_label4)

print("")

poisoned_instance4x = np.expand_dims(poisoned_instance4x, axis=0)
predicted_probs4x = model.predict(poisoned_instance4x)
predicted_label4x = np.argmax(predicted_probs4x)
print("From other class number 0 with backdoor pattern")
print("Predicted Probability: ", predicted_probs4x)
print(predicted_probs4x[0][target_label])
print("Predicted Class", predicted_label4x)

print("")

poisoned_instance2 = np.expand_dims(poisoned_instance2, axis=0)
predicted_probs2 = model.predict(poisoned_instance2)
predicted_label2 = np.argmax(predicted_probs2)
print("From target class 1 (Control)")
print("Predicted Probability: ", predicted_probs2)
print(predicted_probs2[0][target_label])
print("Predicted Class", predicted_label2)

print("Done")
