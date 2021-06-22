import tensorflow as tf
from keras.datasets import fashion_mnist
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def show(images,labels):
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    for i in range(20):
        plt.subplot(5, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 展示训练集前20张图片和标签
show(train_images,train_labels)
# # 将train_images与test_images 这些值除以255缩小至0-1之间，以便将其送到神经网络中。
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10, 10))
# 为了验证数据格式是否正确、是否已准备好构建和训练网络，此处显示训练集中前 20 个图像，并在每个图像下方显示类名称。


# 构建神经网络
# 网络第一层keras.layers.Flatten ：将图像格式从二维数组（28x28像素）转换成一维数组（28x28=784像素）。
# 网络第二层 keras.layers.Dense ：全连接层，128个神经元，激活函数采用线性整流函数ReLU。
# 网络第三层 keras.layers.Dense ：返回一个长度为10的 logits 数组（线性输出）。
model = keras.Sequential(
    [keras.layers.Flatten(input_shape=(28, 28)),
     keras.layers.Dense(256, activation='relu'),
     keras.layers.Dense(64, activation='relu'),
     keras.layers.Dense(10)])

# 编译模型
# 优化器：决定模型如何根据其看到的数据和自身的损失函数进行更新。
# 损失函数：用于测量模型在训练期间的准确率，希望最小化此函数，以便将模型“引导”到正确的方向上。
# 准确率：被正确分类的图像的比率。
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 将训练数据送至模型，使用训练集的全部数据对模型进行20次训练。
model.fit(train_images, train_labels, epochs=20)

# 在测试集上评估
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# 经过训练后，使用它对一些图像进行预测。模型具有线性输出，即 logits 。此处再附加一个 softmax 层，将 logits 转换成更容易理解的概率。
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# 展示测试集前20张图片和真实标签
show(test_images,test_labels)
# 展示测试集前20张图片的预测标签
pre_labels = [np.argmax(prediction) for prediction in predictions]
show(test_images,pre_labels)

file = open("fashion-mnist_test_data.csv", 'r')
file.readline()
new_image = np.loadtxt(file, delimiter=',', skiprows=0)
file.close()

new_image = np.array(new_image)
new_image = np.delete(new_image, 0, axis=1)
new_image = np.reshape(new_image, test_images.shape)

predictions = probability_model.predict(new_image)
with open("result.csv", "w") as f:
    for i, prediction in enumerate(predictions):
        content = str(i) + ".jpg," + str(np.argmax(prediction)) + "\n"
        f.write(content)
