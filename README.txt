Name: Yue Teng
CWID: 10819567
Programming Language: Python


Structure:
First, the program loads the data by reading image files from the directory, resizing and converting them to grayscale, and assigning labels to them. The CNN model is built using the Sequential API of Keras. It contains several layers of convolution, max pooling, and dense layers. The model is trained using the fit method with validation data, and predictions are made after that.


How to Run the Code:
1. Please make sure you have the correct sys.path. For example, my sys.path is "/Users/yueteng/opt/anaconda3/lib/python3.9/site-packages".
2. Please make sure you have all libraries installed. 
3. Please make sure the file path is correct. I use a macbook, thus I have to use the full absolute path. Please modify the path based on your machine.
4. Add/change images for prediction. The image should be in the faces folder. Please also make sure the cv2.imread path is changed.
5 Run the code. Hopefully this provides the correct result.