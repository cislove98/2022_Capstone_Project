import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import numpy as np
import os, glob


np.random.seed(3)
tf.random.set_seed(3)


caltech_dir = "./korBrailleCode/totalPresetdata"
categories = [
    "ㄱCho", "ㄹCho", "ㅅCho", "ㅊCho", "ㅊJong", "space",
    "ㄱJong", "ㄴJong", "ㄷJong", "ㄹJong", "ㅁJong", "ㅂJong", "ㅅJong", "ㅇJong", "ㅈJong", "ㅋJong", "ㅌJong", "ㅍJong", "ㅎJong", "붙임", "것1", "수표",
    "ㅏ", "ㅐ", "ㅑ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
    "가", "나", "다", "마", "바", "사", "억", "언", "얼", "연", "열", "영", "옥", "온", "옹", "운", "울", "은", "을", "인", "자", "카", "타", "파", "하"
]
nb_classes=len(categories)

w=36
h=42

all_image_paths = []
all_onehot_labels = []


for idx, cat in enumerate(categories):
    # 레이블 지정
    label = [0 for i in range(nb_classes)]  # one-hot준비 [0,0,0,0,0]
    label[idx] = 1   # one-hot 리스트 생성
    # 이미지
    image_dir = caltech_dir + "/" + cat
    # 각 폴더에 있는 모든 파일이름에 대한 리스트 생성
    files = glob.glob(image_dir+"/*.jpg")
    for f in files:
        all_image_paths.append(f)
        all_onehot_labels.append(label)


def load_image_path_label(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [w, h])
    image /= 255.0  # normalize to [0,1] range
    return image, label


full_dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, all_onehot_labels))
full_dataset = full_dataset.map(load_image_path_label)


DATASET_SIZE = len(all_image_paths)


train_size = int(0.75 * DATASET_SIZE)
test_size = DATASET_SIZE - train_size
# 랜덤하게 shuffling
full_dataset = full_dataset.shuffle(buffer_size=int(DATASET_SIZE*2))

train_ds = full_dataset.take(train_size)
train_ds = train_ds.batch(30)


# 나머지를 테스트 용으로 사용
test_ds = full_dataset.skip(train_size)
test_ds = test_ds.batch(30)

print("train_ds : "+len(train_ds)+" || test_ds : "+len(test_ds))

########################### 모델 생성 ###################################
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3, 3), input_shape=(w, h, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2)) # default
model.add(Dropout(0.3))

model.add(Conv2D(filters = 64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))      
model.add(Dropout(0.3))


model.add(Conv2D(filters = 128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(625, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(nb_classes, activation='softmax'))
          
model.compile(optimizer = "Adam",  # optimizer=tf.keras.optimizers.Adam(0.001)
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
  
model.summary()

model.fit(train_ds, epochs=50, verbose=1)

model.evaluate(test_ds)