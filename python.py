import datetime
import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras import layers, utils, applications, Model, losses, optimizers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

def my_team():
    '''
    Return the team members and their student codes for this assignment submission as a list 
    of tuples in the form (student_number, first_name, last_name)
    
    @param: 
        N/A

    @return:
       A list of the 3 members of our team:
        - Adrian Ash: n10624937
        - Chiran Walisundara: n10454012
        - Don Kaluarachchi: n10496262 

    '''
    return [ (10624937, 'Adrian', 'Ash'), (10454012, 'Chiran', 'Walisundara'), 
            (10496262, 'Don', 'Kaluarachchi') ]

def debugger(data):
    
    fig = plt.figure(figsize=[10, 5])
    fig.suptitle('Training Data', fontsize=20)  
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(data["train_Y"].flatten(), 5)
    ax.set_xlabel('Class')
    ax.set_ylabel('Instances')
    ax.set_title('Number of Instances per Class');
    fig.patch.set_facecolor('white')
    fig.show()
    for x in range(5):
        print("Training Class " + str(data["class_names"][x]) + " - " + str((data["train_Y"] == x).sum()))
    print("")
    for x in range(5):
        print("Validation Class " + str(data["class_names"][x]) + " - " + str((data["validation_Y"] == x).sum()))
    print("")
    for x in range(5):
        print("Testing Class " + str(data["class_names"][x]) + " - " + str((data["test_Y"] == x).sum()))
    print("")
    for x in range(5):
        sumOf = 0
        sumOf = sumOf + (data["train_Y"] == x).sum()
        sumOf = sumOf + (data["validation_Y"] == x).sum()
        sumOf = sumOf + (data["test_Y"] == x).sum()
        print(str(data["class_names"][x]) + " Class - " + str(sumOf))
    
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(data["train_X"][i].astype("uint8"))
        plt.title(data["class_names"][data["train_Y"][i]])
        plt.axis("off")
    fig.patch.set_facecolor('white')
    plt.show()

    print(data["train_X"].shape)
    print(data["train_Y"].shape)
    print(data["validation_X"].shape)
    print(data["validation_Y"].shape)
    print(data["test_X"].shape)
    print(data["test_Y"].shape)


def data_load_and_preprocessing():
    seed = 52
    data = tf.keras.utils.image_dataset_from_directory(
        "Data",
        labels="inferred",
        color_mode="rgb",
        batch_size=None,
        shuffle=True,
        seed = seed
    )
    print(type(data))
    
    class_names = data.class_names

    combined_ext_img = []
    combined_ext_label = []
    for image, label in data:
        combined_ext_img.append(image)
        combined_ext_label.append(label)

    numpy_ext_img = np.array(combined_ext_img)
    numpy_ext_label = np.array(combined_ext_label)

    numpy_ext_img_reshape = tf.image.resize(numpy_ext_img, (224,224)).numpy()

    #useful for making the perfect test split size of 150
    desired_sample_size = 150
    perfect_test_split = desired_sample_size/len(numpy_ext_label)
    train_X, test_X, train_Y, test_Y = train_test_split(numpy_ext_img_reshape, numpy_ext_label, test_size=perfect_test_split, random_state=seed)

    perfect_test_split = desired_sample_size/len(train_Y)
    train_X, validation_X, train_Y, validation_Y = train_test_split(train_X, train_Y, test_size=perfect_test_split, random_state=seed)

    data = {
        "train_X": train_X,
        "validation_X": validation_X,
        "test_X": test_X,
        "train_Y": train_Y,
        "validation_Y": validation_Y,
        "test_Y": test_Y,
        "class_names": class_names
    }
    return data

def data_embedding(data):
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        weights="imagenet",
        input_shape=image_size,
        include_top = False 
    )
    print(str(len(model.layers)))
    data["train_X"] = model.predict(data["train_X"]);
    data["validation_X"] = model.predict(data["validation_X"]);
    data["test_X"] = model.predict(data["test_X"]);
    return data



def build_model(image_size, full_model=False):
    if full_model:
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            weights="imagenet",
            input_shape=image_size,
            include_top = False 
        )
        outputs = layers.GlobalAveragePooling2D()(model.layers[-1].output)
        outputs = layers.Dense(256, activation='relu')(outputs)
        outputs = layers.Dense(64, activation='relu')(outputs)
        outputs = layers.Dense(5, activation='softmax')(outputs)
        model_full = Model(inputs=model.input, outputs=outputs)
        # model_basic.summary()
        for layer in model_full.layers[:-7]:
            layer.trainable = False
        
        # for layer in model_basic.layers:
        #     print(layer.name, layer.trainable)
        return model_full

    else:
        model_end = tf.keras.Sequential([
        # layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (7, 7, 1280)),
        layers.GlobalAveragePooling2D(input_shape = (7, 7, 1280)),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='softmax')
        ])
        return model_end

def compile_fit(data, model, batch_size, epochs, learning_rate, momentum, name="Basic Model", verbose_value=1):
    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=optimizers.SGD(
                  learning_rate=learning_rate, momentum=momentum, nesterov=False
              ),
              metrics=['accuracy'])
    checkpoint = ModelCheckpoint("current_best.h5", verbose=verbose_value, monitor='val_accuracy',save_best_only=True, mode='auto')
    
    time_1 = process_time()
    history = model.fit(data["train_X"], data["train_Y"],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose = verbose_value,
                            validation_data=(data["validation_X"], data["validation_Y"]),
                            callbacks=[checkpoint])
    time_2 = process_time()

    model.load_weights("current_best.h5")
    model.save("current_best.h5")
    train_time = time_2 - time_1
    return model, history, train_time

def eval_model(data, history, model, train_time, name="Basic Model"):
    print("---------------" + name + "---------------")
    fig = plt.figure(figsize=[30, 10])
    ax = fig.add_subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    ax.legend()
    ax.set_title('Training Performance')

    
    ax = fig.add_subplot(1, 3, 2)
    pred = model.predict(data["train_X"]);
    indexes = tf.argmax(pred, axis=1)
    cm = confusion_matrix(data["train_Y"], indexes)
    c = ConfusionMatrixDisplay(cm, display_labels=data["class_names"])
    c.plot(ax = ax)    
    ax.set_title('Training')
    
    ax = fig.add_subplot(1, 3, 3)

    time_3 = process_time()
    pred = model.predict(data["test_X"]);
    time_4 = process_time()
    
    predict_time = time_4 - time_3

    indexes = tf.argmax(pred, axis=1)
    cm = confusion_matrix(data["test_Y"], indexes)
    c = ConfusionMatrixDisplay(cm, display_labels=data["class_names"])
    c.plot(ax = ax)    
    ax.set_title('Testing')
    fig.patch.set_facecolor('white')
    fig.show()
    print("*******Training and Prediction Times*******")
    print("Training Time of " + name + " - " + str(train_time))
    print("Prediction Time of  " + name + " - " + str(predict_time))
    print(classification_report(data["test_Y"], indexes))    

def task_1(data, image_size, epochs, batch_size, verbose_value):
    model = build_model(image_size)
    model, history, train_time = compile_fit(data, model, batch_size, epochs, 0.01, 0, verbose_value=verbose_value)
    eval_model(data, history, model, train_time)



def task_2(data, image_size, epochs, batch_size, verbose_value):
    learning_rate_values = [0.0001, 0.001, 0.1]
    for learning_rates in learning_rate_values:
        model = build_model(image_size)
        model_name = "Learning Rate - " + str(learning_rates)
        model, history, train_time =  compile_fit(data, model, batch_size, epochs, learning_rates, 0, model_name, verbose_value=verbose_value)
        eval_model(data, history, model, train_time, model_name)

def task_3(data, image_size, epochs, batch_size, verbose_value):
    momentum_values = [0.01, 0.1, 1]
    for momentum in momentum_values:
        model = build_model(image_size)
        model_name = "Momentum - " + str(momentum)
        model, history, train_time = compile_fit(data, model, batch_size, epochs, 0.001, momentum, model_name, verbose_value=verbose_value)
        eval_model(data, history, model, train_time, model_name)


if __name__ == "__main__":   
    pass

    #General common model parameters - can be modified for different models
    #30 so as it divides easily into the 750 samples of our training dataset 
    batch_size = 30
    epochs = 100
    image_size = (224, 224, 3)
    verbose = 0

    print(my_team())

    #Data loading and preprocessing (plus debugging)
    data = data_load_and_preprocessing() # Completes point 2
    #debugger(data)
    data = data_embedding(data) # Completes point 2

    #Task Completion - from machine_learning_assignment.pdf
    task_1(data, image_size, epochs, batch_size, verbose) # Completes point 1, 3 and 4
    task_2(data, image_size, epochs, batch_size, verbose) # Completes point 5
    task_3(data, image_size, epochs, batch_size, verbose) # Completes point 6

    # Point 7 in report