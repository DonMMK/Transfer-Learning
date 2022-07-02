# CAB320_Assignment_2

Basic Classification task

1) Break up data into training and test sets (validation will be taken from the training data at model fit() time)
    - there is a way to create a dataset from a directory 
    - https://keras.io/api/data_loading/image/#image_dataset_from_directory-function 
    - dont forget to look for a .seed of the image_dataset_from_directory function so it doesnt get the same data for training and testingand      splits it properly

2) Get the "MobileNetV2" model from keras

3) Get a pretrained model and modify for our problem (snip off the end layer and just add 5 neurons for our 5 classes)
    - https://keras.io/guides/transfer_learning/

4) Compile and train model on specific parameters

5) Plot the training and validation errors over time

6) Exeriment with 3 different orders of magnitude for learning (10, 1, 0.1)

7) Use the best learning rate to create the final model

6) Create a report