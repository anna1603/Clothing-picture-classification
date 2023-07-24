import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import tkinter
import tkinter.filedialog
import tkinter.messagebox
import numpy as np
from PIL import Image,ImageTk

#创建窗口
root = tkinter.Tk()
root.geometry("500x400")
root.title("DNN图片分类------鞋子衣服帽子裤子")

#设置控件
label_pic = tkinter.Label(root,text="")
label_pic.place(x=100,y=1)

label_result = tkinter.Label(root,text="分类结果如下")
label_result.place(x=1,y=200)

#定义函数
def function1():
    global fname
    fname = tkinter.filedialog.askopenfilename()
    a = Image.open(fname)
    temp = a.resize((400,400))
    b = ImageTk.PhotoImage(temp)
    label_pic.config(image=b)
    label_pic.image = b

# 定义模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),layers.MaxPooling2D((2, 2)),
    layers.Flatten(),layers.Dense(512, activation='relu'),layers.Dense(4, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# 加载数据集
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(r'C:\Users\zzqzz\Desktop\testdiction',target_size=(150, 150),batch_size=20,class_mode='categorical')

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(r'C:\Users\zzqzz\Desktop\testdiction',target_size=(150, 150),batch_size=20,class_mode='categorical')

# 训练模型
#model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50)
model.save("DNN_pic.model")

# 对新图像进行预测
def function2():
    model = tf.keras.models.load_model("DNN_pic.model")
    img_path = fname
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    prediction = model.predict(img_tensor)
    prediction = prediction.tolist()

    for i in prediction:
        coat = i[0]
        hat = i[1]
        shoes = i[2]
        trousers = i[3]
    # print(coat,hat,shoes,trousers)

    result = max(coat, hat, shoes, trousers)
    if result == coat:
        label_result.config(text="该物品是衣服！" + "\n" + "相似度为" + str(result))
    if result == hat:

        label_result.config(text="该物品是帽子！" + "\n" + "相似度为" + str(result))
    if result == shoes:

        label_result.config(text="该物品是鞋子！" + "\n" + "相似度为" + str(result))
    if result == trousers:

        label_result.config(text="该物品是裤子！" + "\n" + "相似度为" + str(result))
#按钮
button1 = tkinter.Button(root,text="选择图片",width=13,height=2,command=function1)
button1.place(x=1,y=1)

#按钮
button2 = tkinter.Button(root,text="分类",width=13,height=2,command=function2)
button2.place(x=1,y=50)


root.mainloop()
