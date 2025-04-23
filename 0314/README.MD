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
```python
- 테스트 데이터셋을 사용해 최종 모델 성능을 평가합니다
- 정확도(test_acc)가 모델의 최종 분류 성능을 나타냅니다


## 이미지 분류(image classification)
