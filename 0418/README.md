# CycleGAN 
```
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# 데이터 병렬 처리 최적화를 위한 설정
AUTOTUNE = tf.data.AUTOTUNE

# horse2zebra 데이터셋 로드
# trainA: 말 이미지, trainB: 얼룩말 이미지
# testA: 말 테스트 이미지, testB: 얼룩말 테스트 이미지
dataset, metadata = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)
train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

# 무작위 자르기 함수 정의
def random_crop(image):
    return tf.image.random_crop(image, size=[256, 256, 3])

# 이미지 정규화: 픽셀값 [-1, 1] 범위로 스케일링
def normalize(image):
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1

# 데이터 증강용 jitter 적용 (리사이즈 -> 랜덤크롭 -> 좌우반전)
def random_jitter(image):
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    return image

# 훈련 이미지 전처리 함수
def preprocess_image_train(image, label):
    image = random_jitter(image)
    image = normalize(image)
    return image

# 테스트 이미지 전처리 함수
def preprocess_image_test(image, label):
    image = normalize(image)
    return image

# 버퍼 크기와 배치 크기 설정
BUFFER_SIZE = 1000
BATCH_SIZE = 1

# 훈련용 말과 얼룩말 이미지 데이터셋 구성
ttrain_horses = train_horses.cache().map(preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_zebras = train_zebras.cache().map(preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_horses = test_horses.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_zebras = test_zebras.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# 출력 채널 수 (RGB 기준)
OUTPUT_CHANNELS = 3

# 생성자 정의 (U-Net 기반)
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')  # 말 -> 얼룩말
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')  # 얼룩말 -> 말

# 판별자 정의 (PatchGAN 구조)
discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)  # 말 판별
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)  # 얼룩말 판별

# 손실 함수 정의
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# 생성자 손실 함수
def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

# 판별자 손실 함수
def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    return (real_loss + generated_loss) * 0.5

# cycle consistency loss: 원래 이미지로 복원되도록 유도
def calc_cycle_loss(real_image, cycled_image):
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss

# identity loss: 동일한 도메인일 경우 그대로 보존되도록 유도
def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

# 손실 함수의 LAMBDA 가중치 설정
LAMBDA = 10

# 옵티마이저 설정
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# 체크포인트 설정
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(
    generator_g=generator_g, generator_f=generator_f,
    discriminator_x=discriminator_x, discriminator_y=discriminator_y,
    generator_g_optimizer=generator_g_optimizer,
    generator_f_optimizer=generator_f_optimizer,
    discriminator_x_optimizer=discriminator_x_optimizer,
    discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 이전 체크포인트 복원
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('최신 체크포인트 복원 완료!')

# 이미지 생성 함수 정의
def generate_images(model, test_input):
    prediction = model(test_input)
    plt.figure(figsize=(12, 12))
    display_list = [test_input[0], prediction[0]]
    title = ['입력 이미지', '변환된 이미지']
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

# 한 스텝 학습 함수 정의
@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)
        
        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    generator_g_optimizer.apply_gradients(zip(tape.gradient(total_gen_g_loss, generator_g.trainable_variables), generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(tape.gradient(total_gen_f_loss, generator_f.trainable_variables), generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(tape.gradient(disc_x_loss, discriminator_x.trainable_variables), discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(tape.gradient(disc_y_loss, discriminator_y.trainable_variables), discriminator_y.trainable_variables))

# 전체 학습 루프
def train(epochs):
    for epoch in range(epochs):
        start = time.time()

        n = 0
        for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
            train_step(image_x, image_y)
            if n % 10 == 0:
                print('.', end='')
            n += 1

        clear_output(wait=True)
        for example_input in test_horses.take(1):
            generate_images(generator_g, example_input)

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Epoch {} 체크포인트 저장 완료: {}'.format(epoch+1, ckpt_save_path))

        print('Epoch {} 완료, 소요 시간: {:.2f}초\n'.format(epoch + 1, time.time()-start))

# 학습 시작
train(10)

# 테스트 이미지 변환 결과 보기
for inp in test_horses.take(5):
    generate_images(generator_g, inp)

```




# 적대 FGSM
