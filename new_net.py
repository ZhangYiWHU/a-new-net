import keras
from keras.regularizers import l2
import numpy as np
def dense_block(layers, filt1, filt2, input):
	c = input
	x = input
	for i in range(layers):
		x = keras.layers.BatchNormalization(axis=-1)(x)
		x = keras.layers.Activation('relu')(x)
		x = keras.layers.Conv2D(filters=filt1, kernel_size=1, strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(x)
		x = keras.layers.BatchNormalization(axis=-1)(x)
		x = keras.layers.Activation('relu')(x)
		x = keras.layers.Conv2D(filters=filt2, kernel_size=3, strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(x)
		c = keras.layers.concatenate([x, c], axis=-1)
		x = c
	return x

input = keras.layers.Input(shape=(704, 896, 3), name='input')
x1 = keras.layers.BatchNormalization(axis=-1)(input)
x1 = keras.layers.Conv2D(filters=48, kernel_size=5, activation='relu', strides=1, padding='same', 
			kernel_initializer='he_uniform', kernel_regularizer=l2(0))(x1)
x1 = keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu', strides=1, padding='same', 
			kernel_initializer='he_uniform', kernel_regularizer=l2(0))(x1)
l1 = keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(x1)
l12 = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l1)
l12 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(l12)
l13 = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu', strides=2, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l1)
l13 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(l13)
# l14 = keras.layers.Conv2D(filters=20, kernel_size=5, activation='relu', strides=4, padding='same', kernel_initializer='he_uniform',
			# kernel_regularizer=l2(0))(l1)
# l14 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(l14)
# l15 = keras.layers.Conv2D(filters=20, kernel_size=5, activation='relu', strides=4, padding='same', kernel_initializer='he_uniform',
			# kernel_regularizer=l2(0))(l1)
# l15 = keras.layers.MaxPooling2D((4, 4), strides=(4, 4))(l15)
x2 = l12
ll2 = dense_block(8,48,16,x2)
ll2 = keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(ll2)
l23 = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(ll2)
l23 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(l23)
l24 = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu', strides=2, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(ll2)
l24 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(l24)
# l25 = keras.layers.Conv2D(filters=20, kernel_size=5, activation='relu', strides=4, padding='same', kernel_initializer='he_uniform',
			# kernel_regularizer=l2(0))(ll2)
# l25 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(l25)
# l26 = keras.layers.Conv2D(filters=20, kernel_size=5, activation='relu', strides=4, padding='same', kernel_initializer='he_uniform',
			# kernel_regularizer=l2(0))(ll2)
# l26 = keras.layers.MaxPooling2D((4, 4), strides=(4, 4))(l26)
x3 = keras.layers.concatenate([l13,l23], axis=-1)
#x3 = keras.layers.Add()([l13,l23])
m3 = keras.layers.Conv2D(filters=20, kernel_size=1, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(x3)
l3 = dense_block(8,64,20,x3)
l3 = keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l3)
l34 = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l3)
l34 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(l34)
l35 = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu', strides=2, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l3)
l35 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(l35)
# l36 = keras.layers.Conv2D(filters=20, kernel_size=5, activation='relu', strides=4, padding='same', kernel_initializer='he_uniform',
			# kernel_regularizer=l2(0))(l3)
# l36 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(l36)
x4 = keras.layers.concatenate([l24,l34], axis=-1)
#x4 = keras.layers.Add()([l24,l34])
m4 = keras.layers.Conv2D(filters=20, kernel_size=1, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(x4)
l4 = dense_block(10,128,32,x4)
l4 = keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l4)
l45 = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l4)
l45 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(l45)
l46 = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu', strides=2, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l4)
l46 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(l46)
x5 = keras.layers.concatenate([l35,l45], axis=-1)
#x5 = keras.layers.Add()([l35,l45])
x5 = keras.layers.Conv2D(filters=40, kernel_size=3, activation='relu', dilation_rate=2, strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(x5)
m5 = keras.layers.Conv2D(filters=20, kernel_size=1, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(x5)
l5 = dense_block(12,128,32,x5)
l5 = keras.layers.Conv2D(filters=48, kernel_size=1, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l5)
l56 = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l5)
l56 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(l56)
x6 = keras.layers.concatenate([l46,l56], axis=-1)
#x6 = keras.layers.Add()([l46,l56])
l6 = dense_block(20,128,32,x6)
l6 = keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l6)
l6 = keras.layers.Dropout(0.5)(l6)
l61 = keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l6)
l62 = keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', dilation_rate=2, strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l6)
l63 = keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', dilation_rate=4, strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(l6)
x = keras.layers.concatenate([l61,l62,l63], axis=-1)
x = keras.layers.Conv2D(filters=20, kernel_size=1, activation='relu', strides=1, padding='same', kernel_initializer='he_uniform',
			kernel_regularizer=l2(0))(x)
u5 = keras.layers.Conv2DTranspose(20, (3, 3), strides=(2,2), kernel_initializer='he_uniform',
			padding='same', kernel_regularizer=l2(0))(x)
i5 = keras.layers.Add()([u5,m5])
u4 = keras.layers.Conv2DTranspose(20, (3, 3), strides=(2,2), kernel_initializer='he_uniform',
			padding='same', kernel_regularizer=l2(0))(i5)
i4 = keras.layers.Add()([u4,m4])
u3 = keras.layers.Conv2DTranspose(20, (3, 3), strides=(2,2), kernel_initializer='he_uniform',
			padding='same', kernel_regularizer=l2(0))(i4)
i3 = keras.layers.Add()([u3,m3])
out = keras.layers.Conv2DTranspose(11, (5, 5), strides=(4,4), kernel_initializer='he_uniform',
			padding='same', kernel_regularizer=l2(0))(i3)
out = keras.layers.Activation('softmax')(out)
model = keras.models.Model(inputs=input, outputs=out)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

img = np.load('E:/fcn/data/img_k.npy')
lab = np.load('E:/fcn/data/lab_k.npy')

model.fit(img,lab,epochs=10,batch_size=1)
model.save('mymodel_new2.h5')
