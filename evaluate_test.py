# evaluate_test.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "fruit_fresh_rotten_model.h5"
CLASS_MAP = "class_indices.json"
TEST_DIR = "archive/dataset/dataset"   # change if your test folder path differs
IMG_SIZE = (160,160)
BATCH_SIZE = 32

# load
model = load_model(MODEL_PATH)
with open(CLASS_MAP,'r') as f:
    class_map = json.load(f)
inv_map = {v:k for k,v in class_map.items()}  # {0: 'Fresh', 1:'Rotten'}

# generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# predict
y_true = test_gen.classes
filenames = test_gen.filenames
probs = model.predict(test_gen, verbose=1).ravel()  # sigmoid outputs
y_pred = (probs >= 0.5).astype(int)

# metrics
print("Classification report:")
print(classification_report(y_true, y_pred, target_names=[inv_map[0], inv_map[1]]))
try:
    auc = roc_auc_score(y_true, probs)
    print("ROC AUC:", auc)
except Exception as e:
    print("ROC AUC error:", e)

# confusion matrix and plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=[inv_map[0], inv_map[1]], yticklabels=[inv_map[0], inv_map[1]])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# save csv with probs
df = pd.DataFrame({
    "filename": filenames,
    "true": [inv_map[int(x)] for x in y_true],
    "pred": [inv_map[int(x)] for x in y_pred],
    "prob_rotten": probs,                  # prob for class 1 (whatever mapping is)
    "prob_fresh": 1.0 - probs
})
df.to_csv("predictions.csv", index=False)
print("Saved predictions.csv and confusion_matrix.png")
