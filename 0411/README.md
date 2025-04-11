# DCGAN(심층 합성곱 생성적 적대 신경망)
## GAN을 이용한 이미지 생성 (DCGAN)

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

## 2. 데이터 준비 (MNIST 데이터셋) 
MNIST는 손글씨 숫자 이미지 데이터셋으로, 이 데이터셋을 사용하여 이미지를 생성합니다. 
```python
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # 이미지를 [-1, 1]로 정규화합니다.
```

- train_images: **28x28 크기의 흑백 이미지**들이 들어있으며, 이 값을 **-1에서 1 사이로 정규화**하여 모델이 더 잘 학습할 수 있도록 만듭니다

## 3. 데이터셋 준비 
배치 처리를 통해 데이터를 묶어서 처리하고, 데이터를 랜덤하게 섞습니다: 
```python
BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

- **배치**(batch) 처리는 딥러닝에서 중요한 부분입니다. 여기서는 데이터를 **256개의 이미지씩 묶어** 처리합니다

## 4. Generator 모델 만들기 (이미지 생성기) 
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

## 5. Discriminator 모델 만들기 (진짜/가짜 구별기) 
Discriminator는 **이미지가 진짜인지 가짜인지** 판별하는 모델입니다. 
```python
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model
```
- Discriminator는 **실제 이미지와 생성된 이미지를 구별**하는 역할을 합니다
- Conv2D는 이미지를 처리하는 **컨볼루션 층**입니다.

## 6. 손실 함수 (Loss Function) 
Generator와 Discriminator의 학습을 평가하는 **손실 함수**를 정의합니다: 
```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
```
- **Discriminator Loss**는 Discriminator가 실제와 가짜 이미지를 얼마나 잘 구별하는지 평가합니다
- **Generator Loss**는 Generator가 생성한 이미지를 얼마나 잘 구별되지 않게 만드는지 평가합니다

## 7. 옵티마이저 (Optimizer) 
옵티마이저는 **학습률**을 설정하고, **모델의 가중치를 조정**하여 성능을 개선합니다. 
```python
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
```

## 8. 체크포인트 (Checkpoint) 
훈련 중 모델의 상태를 **저장**하고, 중단된 후 **재개**할 수 있도록 합니다: 
```python
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
```
## 9. 훈련 루프 (Training Loop) 
훈련 루프에서는 **Generato**r와 **Discriminator**를 동시에 학습시킵니다: 
```python
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
```
- **Generator**는 **랜덤 노이즈**를 입력받아 이미지를 생성합니다
- **Discriminator**는 이미지를 진짜와 가짜로 분류합니다

## 10. 이미지 생성 및 저장 
훈련 중 **Generator**가 생성하는 **이미지를 주기적으로 저장**합니다: 
```python
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
```

## 11. 애니메이션 생성 
훈련 중 생성된 이미지를 **GIF 애니메이션**으로 만들어 저장합니다: 
```python
anim_file = 'dcgan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
```

## 12. 훈련 시작 
훈련을 시작하며, 훈련 중 생성된 이미지를 **애니메이션**으로 변환하고, 훈련이 끝나면 결과를 **GIF 파일로 다운로드**할 수 있습니다: 
```python
%%time
train(train_dataset, EPOCHS)
```

## 13. 다운로드 링크 
훈련이 완료된 후 **생성된 GIF 파일**을 다운로드할 수 있습니다: 
```python
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download(anim_file)
```

## 결론 
이 코드는 **Generative Adversarial Network** (GAN)을 이용하여 **이미지를 생성**하는 모델을 학습시키는 과정입니다. GAN은 **Generator**와 **Discriminator**가 서로 경쟁하며 발전하는 방식으로 작동합니다. 최종적으로 생성된 이미지를 **애니메이션 GIF**로 저장하여 훈련 과정을 시각적으로 확인할 수 있습니다

# Pix2Pix
