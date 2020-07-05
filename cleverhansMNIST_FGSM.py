

# prepare the data
import numpy as np
from keras.datasets import mnist
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

session = tf.Session()
keras.backend.set_session(session)

(mn_x_train, mn_y_train), (mn_x_test, mn_y_test) = mnist.load_data()
print ("Training Examples: %d" % len(mn_x_train))
print ("Test Examples: %d" % len(mn_x_test))

# plot some images
n_classes = 10
inds=np.array([mn_y_train==i for i in range(n_classes)])
f,ax=plt.subplots(2,5,figsize=(10,5))
ax=ax.flatten()
for i in range(n_classes):
    ax[i].imshow(mn_x_train[np.argmax(inds[i])].reshape(28,28))
    ax[i].set_title(str(i))
plt.show()

# train a classifier
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
train_images_1d = mn_x_train.reshape((60000, 28 * 28))
train_images_1d = train_images_1d.astype('float32') / 255

test_images_1d = mn_x_test.reshape((10000, 28 * 28))
test_images_1d = test_images_1d.astype('float32') / 255

#this just converts the labels to one-hot class
from keras.utils import to_categorical 
train_labels = to_categorical(mn_y_train)
test_labels = to_categorical(mn_y_test)

from keras.callbacks import ModelCheckpoint

h=network.fit(train_images_1d, 
              train_labels, 
              epochs=10, 
              batch_size=128, 
              shuffle=True, 
              callbacks=[ModelCheckpoint('tutorial_MNIST.h5',save_best_only=True)])

# test the model
score, acc = network.evaluate(test_images_1d, 
                            test_labels,
                            batch_size=128)

print ("Test Accuracy: %.5f" % acc)

network.save('MNIST.h5')
print('model saved successfully')

from keras.models import load_model
network = load_model('MNIST.h5')
print('model loaded ')
from cleverhans.utils_keras import KerasModelWrapper
wrap = KerasModelWrapper(network)



#FGSM
x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.float32, shape=(None, 10))
from cleverhans.attacks import FastGradientMethod
fgsm = FastGradientMethod(wrap, sess=session)

fgsm_rate = 0.000001
fgsm_params = {'eps': fgsm_rate,'clip_min': 0.,'clip_max': 1.}
adv_x = fgsm.generate(x, **fgsm_params)
adv_x = tf.stop_gradient(adv_x)
adv_prob = network(adv_x)
fetches = [adv_prob]
fetches.append(adv_x)
outputs = session.run(fetches=fetches, feed_dict={x:test_images_1d}) 
adv_prob = outputs[0]
adv_examples = outputs[1]
adv_predicted = adv_prob.argmax(1)
adv_accuracy = np.mean(adv_predicted == mn_y_test)

print("Adversarial accuracy: %.5f" % adv_accuracy)

n_classes = 10
f,ax=plt.subplots(2,5,figsize=(10,5))
ax=ax.flatten()
for i in range(n_classes):
    ax[i].imshow(adv_examples[i].reshape(28,28))
    ax[i].set_title("Adv: %d, Label: %d" % (adv_predicted[i], mn_y_test[i]))
plt.show()

