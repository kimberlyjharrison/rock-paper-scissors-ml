# Rock, Paper, Scissors Game using Computer Vision with Convolutional Neural Networks (CNN)

### Contributors: Kim Harrison, Hunter Orrantia

<hr>

#### Overview
This project takes a machine-learning approach to the classic game of Rock, Paper, Scissors using Computer Vision and a simple Convolutional Neural Network model to determine the hand gesture of a user and evaluate the output against a randomly generated computer choice.

#### Steps

1.	Recommended: create new virtual environment using conda
2. Install Dependencies in requirements.txt
2.	Run `create_gesture_images.py` to create a library of images that will be used to train and test a CNN (utlizes OpenCV)
3. Run `to_csv.ipynb` to format and convert those images into a csv file that will be used as the training dataset (utlizes Pandas, Numpy)
4. Run `create_model.ipynb` to create, train, evaluate and save the CNN model (utlizes Numpy, TensorFlow, Keras, sci-kit learn)
5. Run `rps.py` to play Rock, Paper, Scissors using machine learning!

<hr>

#### Libraries
* Open CV
* TensorFlow with Keras
* sci-kit learn
* Pandas, Numpy, MatplotLib, Jupyter Nobteook

<br>
<hr>
Acknowledgments: Thanks to the creators of the  [Emojintaor App] (https://github.com/akshaybahadur21/Emojinator) for inspiration and guidance during this project!