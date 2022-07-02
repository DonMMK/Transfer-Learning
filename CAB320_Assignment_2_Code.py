import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras import layers, Model, losses, optimizers
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

def debugger(data_dictionary):
    """
    Primary uses are for
        1) Create a histogram for the number of instances per class within the training set
            - to check the data split evenly spread the classes (also for a report diagram)
        2) The number of instances per class for our training, validation and testing sets
            - same as the above except to ensure it is consistent across all data sets
        3) Print the total number of each class within the data (200 samples each)
        4) Printing 9 images from pre-processing to ensure it was done correctly
        5) Print the shape of our trainingm validation and testing datasets to ensure the reshaping and length are correct
    
    @params: 
        data_dictionary - a dictionary of all the data split in training, validation and testing (as well as class names) 

    @return:
       N/A
    """
    #1) Create histogram  
    fig = plt.figure(figsize=[10, 5])
    fig.suptitle('Training Data', fontsize=20)  
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(data_dictionary["train_Y"].flatten(), 5)
    ax.set_xlabel('Class')
    ax.set_ylabel('Instances')
    ax.set_title('Number of Instances per Class');
    fig.patch.set_facecolor('white')
    fig.show()

    #2) Print number of instances per class for our 3 datasets
    for x in range(5):
        print("Training Class " + str(data_dictionary["class_names"][x]) + " - " + str((data_dictionary["train_Y"] == x).sum()))
    print("")
    for x in range(5):
        print("Validation Class " + str(data_dictionary["class_names"][x]) + " - " + str((data_dictionary["validation_Y"] == x).sum()))
    print("")
    for x in range(5):
        print("Testing Class " + str(data_dictionary["class_names"][x]) + " - " + str((data_dictionary["test_Y"] == x).sum()))
    print("")

    #3) Print the total number of each class (not really needed but good for double checking)
    for x in range(5):
        sumOf = 0
        sumOf = sumOf + (data_dictionary["train_Y"] == x).sum()
        sumOf = sumOf + (data_dictionary["validation_Y"] == x).sum()
        sumOf = sumOf + (data_dictionary["test_Y"] == x).sum()
        print(str(data_dictionary["class_names"][x]) + " Class - " + str(sumOf))

    #4) Print 9 images to ensure that preprocessing worked
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(data_dictionary["train_X"][i].astype("uint8"))
        plt.title(data_dictionary["class_names"][data_dictionary["train_Y"][i]])
        plt.axis("off")
    fig.patch.set_facecolor('white')
    plt.show()

    #5) Print the shape our our numpy arrays to make sure they are the correct shape
    print(data_dictionary["train_X"].shape)
    print(data_dictionary["train_Y"].shape)
    print(data_dictionary["validation_X"].shape)
    print(data_dictionary["validation_Y"].shape)
    print(data_dictionary["test_X"].shape)
    print(data_dictionary["test_Y"].shape)


def data_load_and_preprocessing(image_size):
    """
    1) Load the data from the Data folder directory
    2) Extracting the class names
    3) Extracting the data from the Keras data object into an array
    4) Converting that array into a numpy array (for personal comfort)
    5) Resizing the images so as to be consistent with our model input shape
        - the MobileNetV2 is by default 224x224x3, modify image_size value in main for different input shapes
    6) Use train_test_split to split the data into the desired size
        - Training: 700 sample
        - Validation: 150 sample
        - Testing: 150 sample
    7) Create the data_dictionary dictionary to store all the split data and class names
    
    @params: 
        image_size - the size we want our data to be reshaped as (so as to be consistent with our model input shape)

    @return:
       data_dictionary - a dictionary containing all the data split into training, validation and testing (as well as class names)
    """
    #Seed values used to help with consistency
    seed = 52
    #1) Load data from the directory
    data = tf.keras.utils.image_dataset_from_directory(
        "Data",
        labels="inferred",
        color_mode="rgb",
        batch_size=None,
        shuffle=True,
        seed = seed
    )
    #print(type(data))
    
    #2) Extract the class names from the data object
    class_names = data.class_names

    #3) Loop through the data object to extract the image and labels and put in an array
    combined_ext_img = []
    combined_ext_label = []
    for image, label in data:
        combined_ext_img.append(image)
        combined_ext_label.append(label)

    #4) Create numpy arrays from the above arrays
    numpy_ext_img = np.array(combined_ext_img)
    numpy_ext_label = np.array(combined_ext_label)

    #5) Use tensorflows resize() function to change the shape into the correct dimensions
    numpy_ext_img_reshape = tf.image.resize(numpy_ext_img, (image_size[0],image_size[1])).numpy()

    #6) Split the original data array into 700, 150, 150
    #useful for making the perfect test split size of 150
    desired_sample_size = 150
    #calculate the percent value needed to get 150 samples in our test set
    perfect_test_split = desired_sample_size/len(numpy_ext_label)
    train_X, test_X, train_Y, test_Y = train_test_split(numpy_ext_img_reshape, numpy_ext_label, test_size=perfect_test_split, random_state=seed)
    
    #calculate the percent value needed to get 150 samples in our validation set
    perfect_test_split = desired_sample_size/len(train_Y)
    train_X, validation_X, train_Y, validation_Y = train_test_split(train_X, train_Y, test_size=perfect_test_split, random_state=seed)

    #7) Create a dictionary to store our split data and class names 
    data_dictionary = {
        "train_X": train_X,
        "validation_X": validation_X,
        "test_X": test_X,
        "train_Y": train_Y,
        "validation_Y": validation_Y,
        "test_Y": test_Y,
        "class_names": class_names
    }
    return data_dictionary

def data_embedding(data_dictionary, image_size):
    """
    1) Load the MobileNetV2 model
        - With the weights it learnt from ImageNet
        - With the image_size we want our model input to be
        - Without the final "original" classification layer
    2) Use the model to get embeddings of our 3 image datasets and store them back into the data_dictionary
        - Training, Validation and Testing

    This function is important as my computer is unable to train the complete model and as we won't be using fine-tuning 
    it is easier just to convert our data into embedded versions before training and classifying on a simpler network. 
    
    @params: 
        data_dictionary - the original data_dictionary from the data_load_and_preprocessing() function
        image_size - the size we want our model input shape to be

    @return:
       data_dictionary - a dictionary containing all the data split into training, validation and testing
        - NOTE: This data_dictionary differs from data_load_and_preprocessing() data_dictionary by being embeddings of the data 
                from MobileNetV2
    """
    #1) Load the MobileNetV2 from tensorflow
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        #load the weights from the model being trained on ImageNet
        weights="imagenet",
        #use image_size to modify the input shape
        input_shape=image_size,
        # removes the final dense layer (the ImageNet classification layer)
        include_top = False 
    )
    #print(str(len(model.layers)))

    #2) Use the above model to create embeddings from our original data
    #NOTE: The new values are of dimension 7x7x1280 (which is the output of the MobileNetV2 without the final layer)
    data_dictionary["train_X"] = model.predict(data_dictionary["train_X"]);
    data_dictionary["validation_X"] = model.predict(data_dictionary["validation_X"]);
    data_dictionary["test_X"] = model.predict(data_dictionary["test_X"]);
    return data_dictionary


def build_model(image_size, full_model=False):
    """
    If full_model is set to true
        1) Load the MobileNetV2 model
            - With the weights it learnt from ImageNet
            - With the image_size we want our model input to be
            - Without the final "original" classification layer
        2) Add additionally layers to the end of the model
            - GlobalAveragePooling2D layer
            - Dense layer of length 256 and 'relu' activation
            - Dense layer of length 64 and 'relu' activation
            - Final Dense layer of length 5 and 'softmax' activation for our class classification
        3) Freeze the last 7 layers
            - This includes all the layers we added as well as 2 convolutional layers

        NOTE: If this full_model is used then the data_embedding() function should not be used

    If full_model is set to false
        1) Create a simple sequential network that will be used for the training and classification
            - GlobalAveragePooling2D layer (input shape of 7x7x1280)
                -IMPORTANT: this may need to be modified if you dont use the default 224x224x3 input size in the data_embedding() function
            - Dense layer of length 256 and 'relu' activation
            - Dense layer of length 64 and 'relu' activation
            - Final Dense layer of length 5 and 'softmax' activation for our class classification

        NOTE: This network is to be used when the data_embedding() function is used. 
        As mentioned above if the default image_size is not used then is input shape may need to be modified
            - Done by printing the model created in the data_embedding() function and copying the output layer shape
    
    @params: 
        image_size - the size we want our model input shape to be
        full_model - a flag that determines if the user wants the full MobileNetV2 network or a simple classification one

    @return:
        Depending on the full_model flag being true or false this function will return one of the following 
            model_full - A full MobileNetV2 network with the additional layer required for our flower classification task
            model_simple - A simple classification network intended to be used with the embedding output of the data_embedding() function
    """ 
    if full_model:
        #1) Load the MobileNetV2 model with the correct input shape, no final layer and weights from ImageNet
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            weights="imagenet",
            input_shape=image_size,
            include_top = False 
        )
        #2) Add the addtional layer on the end of the model
        # NOTE: Final dense layer of 5 required for our classification task  
        outputs = layers.GlobalAveragePooling2D()(model.layers[-1].output)
        outputs = layers.Dense(256, activation='relu')(outputs)
        outputs = layers.Dense(64, activation='relu')(outputs)
        outputs = layers.Dense(5, activation='softmax')(outputs)
        model_full = Model(inputs=model.input, outputs=outputs)

        # model_basic.summary()

        #3) Freeze the final 7 layers so we do not train the whole model
        for layer in model_full.layers[:-7]:
            layer.trainable = False
        
        # for layer in model_basic.layers:
        #     print(layer.name, layer.trainable)

        return model_full

    else:
        #1) Create a simple sequential model for classification
        # NOTE: our input shape needs to match our output from the data_embedding() function. 
        # may need to be changed if we are using a different value for image_size
        model_simple = tf.keras.Sequential([
        layers.GlobalAveragePooling2D(input_shape = (7, 7, 1280)),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        # Dense layer of 5 for our classification task
        layers.Dense(5, activation='softmax')
        ])
        return model_simple

def compile_fit(data_dictionary, model, batch_size, epochs, learning_rate, momentum, verbose_value=1):
    """
    1) Compile the model with the given learning_rate and momentum values
    2) Train the model on the given data
        - Training X (images) and Y (labels) values
        - batch_size
        - epochs
        - verbose_value
        - Validation X (images) and Y (labels) values
        - With the additional callback checkpoints
    3) Load the weights of the best model that are stored in the "current_best.h5" file
        - NOTE: The best model (according to validation accuracy) will be results. Not the final model 
    
    @params: 
        data_dictionary - a dictionary with the data required to train the network
            - Contains the data split into Training, Validation and Testing datasets
            - The class names

        model - a model object created from the build_model() function that will be the model that we train on our data

        batch_size - The size of our batches used by our model

        epochs - The number of epochs our model will train for

        learning_rate - The learning rate used by the SGD optimizer

        momentum - The momentum used by the SGD optimizer

        verbose_value - The value used to determine if the training progress bar will be hidden or visible
                        as well as if the checkpoint values are displayed (Used for debugging)

    @return:
        model - Returns the trained model with its learnt weights and baises. Used for predicting our testing images in the eval_model() function
        history - Contains the history of our epoch training values (used for graphing our model performance value over epochs)
        train_time - The time taken to run the training process (uses process_time() function that calculates processing time not real time)
    """
    #1) Compile the model with the given learning rate and momentum values
    # SparseCategoricalCrossentropy because we havent defined our data as a Categorical
    # from_logits=False because our model output uses a 'softmax' activation
    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False),
              # using SGD as the optimizer
              optimizer=optimizers.SGD(
                  learning_rate=learning_rate, momentum=momentum, nesterov=False
              ),
              metrics=['accuracy'])
    # ModelCheckpoint() function used to ensure our final model weights will be the best
    checkpoint = ModelCheckpoint("current_best.h5", verbose=verbose_value, monitor='val_accuracy', save_best_only=True, mode='auto')
    
    # Calculate the process time of training
    time_1 = process_time()
    #2) train the model with all the required variables
    history = model.fit(data_dictionary["train_X"], data_dictionary["train_Y"],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose = verbose_value,
                            validation_data=(data_dictionary["validation_X"], data_dictionary["validation_Y"]),
                            callbacks=[checkpoint])
    time_2 = process_time()

    #3) Load our saved best model (with regards to validation accuracy)
    model.load_weights("current_best.h5")
    model.save("current_best.h5")
    train_time = time_2 - time_1
    return model, history, train_time

def eval_model(data_dictionary, history, model, train_time, name="Basic Model"):
    """
    Designed to display all the performance stats of a trained model
    1) Creates a plot of the model history over time (epochs)
        - Training Loss
        - Training Accuracy
        - Validation Loss
        - Validation Accuracy
    2) Creates a confusion matrix of the models predicted values on our Training Data
    3) Using process_time() to calculate the time it take for the model to predict all of our testing data
    4) Creates a confusion matrix of the models predicted values on our Testings Data
    5) Prints our Training and Prediction times
    6) Prints a classification report which gives many different accuracy measurements based on our Test data predictions
        - NOTE: this is were we get the F1-Score from that we use in the report
    
    @params: 
        data_dictionary - a dictionary with the data required evaluate the trained model
            - Contains the data split into Training, Validation and Testing datasets
            - The class names
        history - Contains the history of our epoch training values (used for graphing our model performance value over epochs)
        model - trained model with its learnt weights and baises that will be used for predicting our data
        train_time - The processing time taken to train the model
        name - The name of the model we are evaluating

    @return:
        N/A
    """ 
    print("---------------" + name + "---------------")
    # 1) Creates the main plot used in the report thats show our Validation and Training, loss and accuracy over time (epochs)
    fig = plt.figure(figsize=[30, 10])
    ax = fig.add_subplot(1, 3, 1)
    # Plot based on our history variable
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    ax.legend()
    ax.set_title('Training Performance')

    #2) Creates a confusion matrix on our training data accuracy
    ax = fig.add_subplot(1, 3, 2)
    # Use our model to predict our training data classes
    pred = model.predict(data_dictionary["train_X"]);
    indexes = tf.argmax(pred, axis=1)
    cm = confusion_matrix(data_dictionary["train_Y"], indexes)
    c = ConfusionMatrixDisplay(cm, display_labels=data_dictionary["class_names"])
    c.plot(ax = ax)    
    ax.set_title('Training')
    
    ax = fig.add_subplot(1, 3, 3)

    #3) Use process_time() to calculate the time it take to predict our training data
    time_3 = process_time()
    pred = model.predict(data_dictionary["test_X"]);
    time_4 = process_time()
    
    predict_time = time_4 - time_3

    #4) Uses the above predictions to create a confusion matrix on our testing data
    indexes = tf.argmax(pred, axis=1)
    cm = confusion_matrix(data_dictionary["test_Y"], indexes)
    c = ConfusionMatrixDisplay(cm, display_labels=data_dictionary["class_names"])
    c.plot(ax = ax)    
    ax.set_title('Testing')
    fig.patch.set_facecolor('white')
    fig.show()
    #5) Print out our training and predction processing times
    print("*******Training and Prediction Times*******")
    print("Training Time of " + name + " - " + str(train_time))
    print("Prediction Time of  " + name + " - " + str(predict_time))
    #6) Print out the classification report on our Test data prediction results
    print(classification_report(data_dictionary["test_Y"], indexes))    


def task_1(data_dictionary, image_size, epochs, batch_size, verbose_value):
    """
    From the Marking Criteria - Task Completion section
    This task completes points 1, 3 and 4:
        - 1: Building of the new network based on MobileNet v2 for a 5 class problem
        - 3: Compilation and training of your model with an SGD optimizer using the following parameters 
             learning_rate=0.01, momentum=0.0, nesterov=False
        - 4: Plots of the training and validation errors vs time as well as the training and validation accuracies
    
    This function:
        1) Builds a model
        2) Trains that model using the provided model parameters
        3) Evaluated that model

    @params: 
        data_dictionary - a dictionary with the data required to train the network
            - Contains the data split into Training, Validation and Testing datasets
            - The class names

        image_size - the size we want our model input shape to be

        epochs - The number of epochs our model will train for
        
        batch_size - The size of our batches used by our model

        verbose_value - The value used to determine if the training progress bar will be hidden or visible
                        as well as if the checkpoint values are displayed (Used for debugging)

    @return:
        N/A
    """ 
    #1) Build the model
    model = build_model(image_size)
    #2) Compile and train our model
    model, history, train_time = compile_fit(data_dictionary, model, batch_size, epochs, 0.01, 0, verbose_value=verbose_value)
    #3) Evaluate the performance of that model
    eval_model(data_dictionary, history, model, train_time)

def task_2(data_dictionary, image_size, epochs, batch_size, verbose_value):
    """
    From the Marking Criteria - Task Completion section
    This task completes point 5:
        - 5: Experiment with 3 different orders of magnitude for the learning rate. Plot of the results, drawing of conclusions
    
    This function:
        1) Creates an array of "3 different orders of magnitude for the learning rate" and uses it to create 3 models
            a) Builds a model
            b) Trains that model using the different learning rates
            c) Evaluated that model

    @params: 
        data_dictionary - a dictionary with the data required to train the network
            - Contains the data split into Training, Validation and Testing datasets
            - The class names

        image_size - the size we want our model input shape to be

        epochs - The number of epochs our model will train for
        
        batch_size - The size of our batches used by our model

        verbose_value - The value used to determine if the training progress bar will be hidden or visible
                        as well as if the checkpoint values are displayed (Used for debugging)

    @return:
        N/A
    """
    #1) Create an array of 3 different learning rate values
    learning_rate_values = [0.0001, 0.001, 0.1]
    for learning_rates in learning_rate_values:
        #a) Build the model
        model = build_model(image_size)
        # Set the name of this model (based on the learning rate)
        model_name = "Learning Rate - " + str(learning_rates)
        #b) Train the model
        model, history, train_time =  compile_fit(data_dictionary, model, batch_size, epochs, learning_rates, 0, verbose_value=verbose_value)
        #c) Evaluated the model
        eval_model(data_dictionary, history, model, train_time, model_name)

def task_3(data_dictionary, image_size, epochs, batch_size, verbose_value):
    """
    From the Marking Criteria - Task Completion section
    This task completes point 6:
        - 6: Experiment with a non zero momentum for the SGD. Report how your results change
    
    This function:
        1) Creates an array of 3 different momentum values and uses them to create 3 models
            a) Builds a model
            b) Trains that model using the different momentum values
            c) Evaluated that model

    @params: 
        data_dictionary - a dictionary with the data required to train the network
            - Contains the data split into Training, Validation and Testing datasets
            - The class names

        image_size - the size we want our model input shape to be

        epochs - The number of epochs our model will train for
        
        batch_size - The size of our batches used by our model

        verbose_value - The value used to determine if the training progress bar will be hidden or visible
                        as well as if the checkpoint values are displayed (Used for debugging)

    @return:
        N/A
    """ 
    #1) Create an array of 3 different momentum values
    momentum_values = [0.01, 0.1, 1]
    for momentum in momentum_values:
        #a) Build the model
        model = build_model(image_size)
        # Set the name of this model (based on the momentum)
        model_name = "Momentum - " + str(momentum)
        #b) Train the model
        model, history, train_time = compile_fit(data_dictionary, model, batch_size, epochs, 0.001, momentum, verbose_value=verbose_value)
        #c) Evaluated the model
        eval_model(data_dictionary, history, model, train_time, model_name)

def task_4(data_dictionary, image_size, batch_size, verbose_value):
    """
    From the Marking Criteria - Task Completion section
    This task completes point 7:
        - 7: Recommendation (uses the recommended values as laid out in the report)
    
    This function:
        1) Creates new values based on recommendation laid out in the report
        2) Builds a model
        3) Trains that model using the those values
        4) Evaluated that model


    @params: 
        data_dictionary - a dictionary with the data required to train the network
            - Contains the data split into Training, Validation and Testing datasets
            - The class names

        image_size - the size we want our model input shape to be
        
        batch_size - The size of our batches used by our model

        verbose_value - The value used to determine if the training progress bar will be hidden or visible
                        as well as if the checkpoint values are displayed (Used for debugging)

    @return:
        N/A
    """
    #1) Creates to model parameter values (based on report suggestions)
    epochs = 1000
    learning_rate = 0.001
    momentum = 0
    #2) Build the model
    model = build_model(image_size)
    # Set the name of this model
    model_name = "Recommended Model"
    #3) Train the model
    model, history, train_time = compile_fit(data_dictionary, model, batch_size, epochs, learning_rate, momentum, verbose_value=verbose_value)
    #4) Evaluated that model
    eval_model(data_dictionary, history, model, train_time, model_name)

if __name__ == "__main__":   
    pass

    #NOTE: General customizable model parameters
    #Batch Size - 30 so as it divides easily into the 750 samples of our training dataset 
    batch_size = 30

    #Epochs - 100 so as to see model trends over long time (additionally it is better for our chosen low learning rate in task 3)
    epochs = 100

    # Image Size - used by build_model() and data_load_and_preprocessing() to determine the image size and input layer dimensions
    # 224x224 to ensure it matches "MobileNetV2" trained on "ImageNet" 
    image_size = (224, 224, 3)

    # Verbose - Comment this line out if you want to turn on training progress bars
    verbose = 0

    print(my_team())

    #NOTE: Data Loading and Preprocessing
    data_dictionary = data_load_and_preprocessing(image_size) # Completes point 2

    #Optional debugger function to double check loading and preprocessing worked 
    #debugger(data_dictionary)

    data_dictionary = data_embedding(data_dictionary, image_size) # Completes point 2

    #NOTE: Task Completion - from machine_learning_assignment.pdf
    # Create basic model
    task_1(data_dictionary, image_size, epochs, batch_size, verbose) # Completes point 1, 3 and 4
    # Learning Rate experimentation
    task_2(data_dictionary, image_size, epochs, batch_size, verbose) # Completes point 5
    # Momentum experimentation
    task_3(data_dictionary, image_size, epochs, batch_size, verbose) # Completes point 6
    # Recommended model (as talked about in report)
    task_4(data_dictionary, image_size, batch_size, verbose) # Completes point 7

    
    