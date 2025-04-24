# 생성 

## 신경 스타일 전이(Neural style transfer)
```
import os
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import time
import functools
import tensorflow_hub as hub

# 환경 설정
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

# 유틸 함수 정의
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    return img[tf.newaxis, :]

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# 이미지 로딩
content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

content_image = load_img(content_path)
style_image = load_img(style_path)

# 이미지 출력
plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')
plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

# 허브 모델을 통한 간단 스타일 전이
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image)

# 스타일/컨텐츠 레이어 설정
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# VGG 레이어 출력 모델 정의
def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)

# Gram Matrix 계산
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

# 스타일 컨텐츠 모델 정의
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super().__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

# 타깃 특징 추출
extractor = StyleContentModel(style_layers, content_layers)
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# 학습을 위한 이미지와 옵티마이저 초기화
image = tf.Variable(content_image)
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30

# 손실 함수 정의
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs])
    style_loss *= style_weight / num_style_layers
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs])
    content_loss *= content_weight / num_content_layers
    return style_loss + content_loss

# 학습 스텝 정의
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

# 학습 루프
start = time.time()
epochs = 10
steps_per_epoch = 100
step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print('.', end='', flush=True)
    display.clear_output(wait=True)
    display.display(tensor_to_image(image))
    print("Train step:", step)
end = time.time()
print("Total time: {:.1f}".format(end-start))

# 결과 저장
file_name = 'stylized-image.png'
tensor_to_image(image).save(file_name)

try:
    from google.colab import files
    files.download(file_name)
except ImportError:
    pass
else:
  files.download(file_name)
```

## 딥드림 (DeepDream)
```
# TensorFlow 및 기타 필요한 라이브러리 임포트
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import IPython.display as display
import PIL.Image

# 이미지 다운로드 함수 정의
def download(url, max_dim=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))  # 최대 크기 제한
    return np.array(img)

# 이미지 후처리 함수: -1~1 범위에서 0~255 정수형으로 변환
def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)

# 이미지 출력 함수
def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))

# 원본 이미지 다운로드 및 표시
url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
original_img = download(url, max_dim=500)
show(original_img)

# InceptionV3 모델 로딩 (이미지넷 사전 학습, 마지막 분류층 제외)
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# 중간 계층 이름 선택 (DeepDream 효과 강조를 위한)
names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]

# 선택한 계층의 출력을 반환하는 모델 생성
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

# 손실 계산 함수: 이미지가 얼마나 활성화를 잘 유도하는지를 계산
def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]
    losses = [tf.reduce_mean(act) for act in layer_activations]
    return tf.reduce_sum(losses)

# DeepDream 클래스 정의
class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32)))
    def __call__(self, img, steps, step_size):
        print("Tracing")  # 그래프 추적 중 출력
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)
            gradients = tape.gradient(loss, img)
            gradients /= tf.math.reduce_std(gradients) + 1e-8  # 정규화
            img = img + gradients * step_size  # 이미지 업데이트 (그래디언트 상승)
            img = tf.clip_by_value(img, -1, 1)  # 값 범위 제한
        return loss, img

# DeepDream 인스턴스 생성
deepdream = DeepDream(dream_model)

# 단순 DeepDream 실행 함수 정의
def run_deep_dream_simple(img, steps=100, step_size=0.01):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)

    steps_remaining = steps
    step = 0
    while steps_remaining:
        run_steps = tf.constant(min(steps_remaining, 100))
        steps_remaining -= run_steps
        step += run_steps
        loss, img = deepdream(img, run_steps, tf.constant(step_size))
        display.clear_output(wait=True)
        show(deprocess(img))
        print("Step {}, loss {}".format(step, loss))
    result = deprocess(img)
    display.clear_output(wait=True)
    show(result)
    return result

# 단순 실행
dream_img = run_deep_dream_simple(img=original_img, steps=100, step_size=0.01)

# 옥타브(해상도)를 바꿔가며 DeepDream 실행
import time
start = time.time()

OCTAVE_SCALE = 1.30
img = tf.constant(np.array(original_img))
base_shape = tf.shape(img)[:-1]
float_base_shape = tf.cast(base_shape, tf.float32)

for n in range(-2, 3):
    new_shape = tf.cast(float_base_shape*(OCTAVE_SCALE**n), tf.int32)
    img = tf.image.resize(img, new_shape).numpy()
    img = run_deep_dream_simple(img=img, steps=50, step_size=0.01)

display.clear_output(wait=True)
img = tf.image.resize(img, base_shape)
img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
show(img)

end = time.time()
print(f"총 시간: {end-start:.2f}초")

# 이미지를 무작위로 이동하여 타일 경계 줄이기
def random_roll(img, maxroll):
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    img_rolled = tf.roll(img, shift=shift, axis=[0,1])
    return shift, img_rolled

shift, img_rolled = random_roll(np.array(original_img), 512)
show(img_rolled)

# 타일 기반 그래디언트 계산 클래스
class TiledGradients(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[2], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.int32)))
    def __call__(self, img, img_size, tile_size=512):
        shift, img_rolled = random_roll(img, tile_size)
        gradients = tf.zeros_like(img_rolled)

        xs = tf.range(0, img_size[1], tile_size)[:-1]
        if not tf.cast(len(xs), bool): xs = tf.constant([0])
        ys = tf.range(0, img_size[0], tile_size)[:-1]
        if not tf.cast(len(ys), bool): ys = tf.constant([0])

        for x in xs:
            for y in ys:
                with tf.GradientTape() as tape:
                    tape.watch(img_rolled)
                    img_tile = img_rolled[y:y+tile_size, x:x+tile_size]
                    loss = calc_loss(img_tile, self.model)
                gradients += tape.gradient(loss, img_rolled)

        gradients = tf.roll(gradients, shift=-shift, axis=[0,1])
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        return gradients

get_tiled_gradients = TiledGradients(dream_model)

# 옥타브 + 타일 기반 DeepDream 실행 함수
def run_deep_dream_with_octaves(img, steps_per_octave=100, step_size=0.01, 
                                octaves=range(-2, 3), octave_scale=1.3):
    base_shape = tf.shape(img)
    img = tf.keras.utils.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)

    for octave in octaves:
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32)*(octave_scale**octave)
        new_size = tf.cast(new_size, tf.int32)
        img = tf.image.resize(img, new_size)

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img, new_size)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

            if step % 10 == 0:
                display.clear_output(wait=True)
                show(deprocess(img))
                print(f"Octave {octave}, Step {step}")

    result = deprocess(img)
    return result

# 타일 기반 DeepDream 실행
img = run_deep_dream_with_octaves(img=original_img, step_size=0.01)
display.clear_output(wait=True)
img = tf.image.resize(img, base_shape)
img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
show(img)

```
