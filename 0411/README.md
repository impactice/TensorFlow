# GAN을 이용한 이미지 생성 (DCGAN)

이 프로젝트는 **Generative Adversarial Network (GAN)**을 사용하여 **이미지를 생성**하는 모델을 구현하는 과정입니다. GAN은 **Generator**와 **Discriminator**라는 두 개의 신경망 모델이 서로 경쟁하면서 발전하는 방식으로 작동합니다. Generator는 **이미지를 생성**하고, Discriminator는 그 이미지를 **진짜 이미지**와 **가짜 이미지**로 구별합니다.

## 1. 필수 라이브러리

먼저, GAN을 구현하기 위해 필요한 라이브러리들을 가져옵니다:

```python
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
```
- TensorFlow: 딥러닝 모델을 구축하고 학습을 진행하는 라이브러리입니다.
- glob, imageio: 이미지 파일을 읽고, GIF 파일로 저장하는 데 사용됩니다
- matplotlib: 결과 이미지를 시각화하는 데 사용됩니다
- PIL: 이미지를 처리하고 저장하는 라이브러리입니다

# 2. 데이터 준비 (MNIST 데이터셋) 
MNIST는 손글씨 숫자 이미지 데이터셋으로, 이 데이터셋을 사용하여 이미지를 생성합니다. 
```python
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # 이미지를 [-1, 1]로 정규화합니다.
```

- train_images: **28x28 크기의 흑백 이미지**들이 들어있으며, 이 값을 **-1에서 1 사이로 정규화**하여 모델이 더 잘 학습할 수 있도록 만듭니다

# 3. 데이터셋 준비 
배치 처리를 통해 데이터를 묶어서 처리하고, 데이터를 랜덤하게 섞습니다: 
```python
BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

- **배치**(batch) 처리는 딥러닝에서 중요한 부분입니다. 여기서는 데이터를 **256개의 이미지씩 묶어** 처리합니다

# 4. Generator 모델 만들기 (이미지 생성기) 
Generator는 **무작위 노이즈** (100차원 벡터)를 입력받아 이미지를 생성하는 신경망입니다. 
```python
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model
```
- Generator는 점차적으로 이미지를 **고해상도로 생성**하는 모델입니다.
- 마지막 tanh 활성화 함수를 사용하여 **이미지 픽셀 값을 -1에서 1로 정규화**합니다
