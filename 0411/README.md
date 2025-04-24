# DCGAN(심층 합성곱 생성적 적대 신경망)
## GAN을 이용한 이미지 생성 (DCGAN)

이 프로젝트는 **Generative Adversarial Network (GAN)**을 사용하여 **이미지를 생성**하는 모델을 구현하는 과정입니다. GAN은 **Generator**와 **Discriminator**라는 두 개의 신경망 모델이 서로 경쟁하면서 발전하는 방식으로 작동합니다. Generator는 **이미지를 생성**하고, Discriminator는 그 이미지를 **진짜 이미지**와 **가짜 이미지**로 구별합니다.
```
import tensorflow as tf
print(tf.__version__)
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display

# MNIST 데이터셋 로드
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # 이미지 정규화 [-1, 1]

# 학습 설정
BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# 생성자 모델 정의
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

generator = make_generator_model()

# 생성된 이미지 시각화
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')

# 판별자 모델 정의
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

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

# 손실 함수 정의
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# 옵티마이저 정의
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# 체크포인트 저장 설정
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# 학습 파라미터
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# 학습 단계 정의
@tf.function
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

# 전체 학습 루프
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('에포크 {} 완료, 시간: {}초'.format(epoch + 1, time.time()-start))

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

# 이미지 생성 및 저장 함수
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

# 학습 시작
%time
train(train_dataset, EPOCHS)

# 최신 체크포인트 복원
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# 특정 에포크 이미지 보여주기
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
display_image(EPOCHS)

# GIF 애니메이션 생성
anim_file = 'dcgan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

# Colab 환경에서 다운로드 지원
import IPython
if IPython.version_info > (6,2,0,''):
    display.Image(filename=anim_file)

try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(anim_file)
```
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

# Pix2Pix : 조건부 GAN을 사용한 이미지 대 이미지 변환
```
import tensorflow as tf
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display

# 데이터셋 다운로드 및 압축 해제
dataset_name = "facades"
_URL = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'
path_to_zip = tf.keras.utils.get_file(fname=f"{dataset_name}.tar.gz", origin=_URL, extract=True)
path_to_zip = pathlib.Path(path_to_zip)
PATH = path_to_zip.parent / dataset_name

# 샘플 이미지 로드 및 확인
sample_image = tf.io.read_file(str(PATH / 'train/1.jpg'))
sample_image = tf.io.decode_jpeg(sample_image)
plt.imshow(sample_image)

# 이미지 분할 함수 (입력 이미지와 실제 이미지로 나누기)
def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)
  w = tf.shape(image)[1] // 2
  input_image = image[:, w:, :]
  real_image = image[:, :w, :]
  return tf.cast(input_image, tf.float32), tf.cast(real_image, tf.float32)

# 데이터셋 관련 상수 정의
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

# 데이터 전처리 함수 정의

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
  return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1
  return input_image, real_image

@tf.function
def random_jitter(input_image, real_image):
  input_image, real_image = resize(input_image, real_image, 286, 286)
  input_image, real_image = random_crop(input_image, real_image)
  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)
  return input_image, real_image

# 훈련/테스트 이미지 로드 함수
def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  return normalize(input_image, real_image)

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  return normalize(input_image, real_image)

# 데이터셋 생성
train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

try:
  test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
except tf.errors.InvalidArgumentError:
  test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
test_dataset = test_dataset.map(load_image_test).batch(BATCH_SIZE)

# 다운샘플 및 업샘플 블록 정의
OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.LeakyReLU())
  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())
  return result

# Generator 모델 정의 (U-Net 구조)
def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),
    downsample(128, 4),
    downsample(256, 4),
    downsample(512, 4),
    downsample(512, 4),
    downsample(512, 4),
    downsample(512, 4),
    downsample(512, 4),
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),
    upsample(512, 4, apply_dropout=True),
    upsample(512, 4, apply_dropout=True),
    upsample(512, 4),
    upsample(256, 4),
    upsample(128, 4),
    upsample(64, 4),
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

  x = inputs
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)
  skips = reversed(skips[:-1])
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])
  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

# Discriminator 모델 정의
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  return gan_loss + (LAMBDA * l1_loss), gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)
  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
  x = tf.keras.layers.concatenate([inp, tar])
  x = downsample(64, 4, False)(x)
  x = downsample(128, 4)(x)
  x = downsample(256, 4)(x)
  x = tf.keras.layers.ZeroPadding2D()(x)
  x = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.LeakyReLU()(x)
  x = tf.keras.layers.ZeroPadding2D()(x)
  last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)
  return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  return real_loss + generated_loss

# 옵티마이저 및 체크포인트 설정
generator = Generator()
discriminator = Discriminator()
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# 이미지 생성 함수
def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['입력 이미지', '정답 이미지', '생성된 이미지']
  plt.figure(figsize=(15, 15))
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

# TensorBoard 설정
log_dir="logs/"
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# 훈련 루프 단일 스텝
@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)
    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)
    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
  generator_optimizer.apply_gradients(zip(gen_tape.gradient(gen_total_loss, generator.trainable_variables), generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(disc_tape.gradient(disc_loss, discriminator.trainable_variables), discriminator.trainable_variables))
  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

# 전체 훈련 함수
def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()
  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      display.clear_output(wait=True)
      if step != 0:
        print(f'1000 스텝 소요 시간: {time.time()-start:.2f}초')
      start = time.time()
      generate_images(generator, example_input, example_target)
      print(f"스텝: {step//1000}k")
    train_step(input_image, target, step)
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

# 훈련 실행
fit(train_dataset, test_dataset, steps=40000)

# 모델 복원 및 테스트 이미지에 대해 결과 생성
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
for inp, tar in test_dataset.take(5):
  generate_images(generator, inp, tar)
```
