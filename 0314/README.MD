# 이미지 
## 컨볼루셔널 신경망(Convolutional_Neural_Network) 

```
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load and preprocess CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Plot sample images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Build CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))  # Output layer for 10 classes

# 모델 구조 출력
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Plot training results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")
```

## 🧠 TensorFlow CIFAR-10 CNN 설명

이 문서는 TensorFlow를 사용해 CIFAR-10 데이터셋을 분류하는 CNN 모델의 전체 흐름을 단계별로 설명합니다.

---

### 1️⃣ 필요한 라이브러리 불러오기
```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```
- TensorFlow와 Keras를 사용해 모델을 설계하고, matplotlib로 이미지 시각화를 합니다 

### 2️⃣ 데이터셋 불러오기 및 정규화 
```python
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
``` 
- CIFAR-10 데이터셋은 10가지 클래스를 가진 32x32 크기의 컬러 이미지들로 구성되어 있습니다 
- 픽셀 값을 0~1 범위로 정규화해 학습 효율을 높입니다 

### 3️⃣ 클래스 이름 정의 및 이미지 시각화 
```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([]), plt.yticks([]), plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
```
- 학습 데이터 중 25장을 무작위로 시각화하여 데이터가 잘 로딩되었는지 확인합니다 

### 4️⃣ CNN 모델 정의 
```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
```
- Conv2D와 MaxPooling 계층을 반복하여 이미지의 특징을 추출합니다
- Flatten을 사용해 1차원 벡터로 만든 후, Dense 층을 통해 분류를 수행합니다

### 5️⃣ 모델 컴파일  
```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
- Adam 옵티마이저, SparseCategoricalCrossentropy 손실함수, accuracy 지표를 사용합니다  
- from_logits=True는 마지막 Dense(10)에서 softmax를 생략했기 때문에 필요합니다

### 6️⃣ 모델 학습 
```python
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
```
- 모델을 학습하며 검증 데이터로 정확도를 확인합니다
- history 객체를 통해 학습 도중의 정확도 기록을 저장할 수 있습니다

### 7️⃣ 학습 정확도 시각화  
```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show() 
```
- 에폭(epoch)별 정확도 변화를 시각화하여 학습이 잘 되고 있는지 확인합니다
### 8️⃣ 모델 평가 
```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.4f}") 
```

- 테스트 데이터셋을 사용해 최종 모델 성능을 평가합니다
- 정확도(test_acc)가 모델의 최종 분류 성능을 나타냅니다




## 이미지 분류(image classification) 
### 📌 전체 구조 요약

1. 데이터 로드 및 시각화
2. 데이터셋 분할 (train/val)
3. 전처리 (정규화, 캐시, prefetch 등)
4. 모델 설계와 학습
5. 데이터 증강 추가 후 재학습
6. 이미지 예측
7. TensorFlow Lite 모델 변환 및 추론 
```python

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

# 데이터 다운로드
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# 클래스별 이미지 확인
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
PIL.Image.open(str(roses[1]))

# 하이퍼파라미터 설정
batch_size = 32
img_height = 180
img_width = 180

# 데이터셋 로드 (훈련/검증)
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# 시각화
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# 최적화 설정
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 정규화 레이어
normalization_layer = layers.Rescaling(1./255)

# 간단한 모델 정의
num_classes = len(class_names)
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# 학습
epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# 학습 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# 데이터 증강 적용한 모델
data_augmentation = keras.Sequential([
  layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# 새 이미지 예측
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.utils.load_img(sunflower_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print("This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score)))

# TensorFlow Lite 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
classify_lite = interpreter.get_signature_runner('serving_default')
predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']
score_lite = tf.nn.softmax(predictions_lite)

print("Lite Model: This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite)))
print(np.max(np.abs(predictions - predictions_lite)))
```




---

## ✍️ 2. 설명 정리 (README용 Markdown)

```markdown
# 🌸 Flower Image Classification with TensorFlow

This project trains a CNN model to classify flower images into 5 categories, and demonstrates TensorFlow Lite model conversion for lightweight deployment.

---

## 🧠 Summary

| Feature        | Description |
|----------------|-------------|
| **Dataset**    | Flower images (daisy, dandelion, roses, sunflowers, tulips) |
| **Model**      | CNN (Conv2D + MaxPooling + Dense) |
| **Preprocessing** | Rescaling, Shuffle, Prefetch |
| **Augmentation** | Flip, Rotation, Zoom |
| **Loss**       | SparseCategoricalCrossentropy (with logits) |
| **Optimizer**  | Adam |
| **Deployment** | TensorFlow Lite (.tflite) conversion |

---

## 🏗️ Architecture

```text
Input Image → Data Augmentation → Conv2D Layers → MaxPooling → Dropout → Flatten → Dense → Output
