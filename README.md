# Chest X-Ray Classification Project

## Project Overview
The Chest X-Ray Classification Project is a machine learning application designed to classify chest x-ray images into two categories: NORMAL and PNEUMONIA. The tool utilizes a convolutional neural network (CNN) built with TensorFlow and Keras, allowing for effective image processing and classification.

## Purpose
This project was created to provide an accurate and efficient method for identifying pneumonia from chest x-rays. It is my first large project to train a model and aims to help teach me and may be used to help others.

## Features
- **Image Preprocessing:** Converts and normalizes x-ray images for optimal model performance.
- **CNN Model:** A custom-built convolutional neural network for image classification.
- **Transfer Learning:** Integrates the VGG16 model to improve accuracy and speed up training.
- **Visualization:** Generates confusion matrices and other visual aids to assess model performance.

## Technologies Used
- **TensorFlow & Keras:** The core frameworks for building and training the neural network.
- **OpenCV:** Used for image loading and preprocessing.
- **Pandas:** For data handling and manipulation.
- **Matplotlib & Seaborn:** Libraries used for plotting and data visualization.

## File Structure
- **main.py:** The main script containing the logic for data loading, model training, and evaluation.
- **requirements.txt:** List of required Python libraries.
- **train, val, test directories:** Folders containing the training, validation, and test datasets respectively.

## Dataset
Due to the large size of the dataset, it cannot be uploaded directly to GitHub. Please follow the instructions below to download the dataset:

1. **Download the Dataset**:
   - Visit the following link: [Chest X-Ray Images (Pneumonia) Dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
   - Sign in or create a Kaggle account if you don't have one.
   - Click the "Download" button to download the dataset.

2. **Extract and Place the Data**:
   - After downloading, make sure the file is in the data/ folder
   - Extract the contents of the ZIP file.

## Installation
To run this project locally:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/chest-xray-classification.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd chest-xray-classification
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download and Prepare Dataset:**

    - Ensure you have the chest x-ray dataset stored in the following structure:
      - `chest_xray/train/`
      - `chest_xray/val/`
      - `chest_xray/test/`
      
    - Each of these directories should contain subdirectories named `NORMAL` and `PNEUMONIA` with the corresponding images.

## How to Use

1. **Execute the Script:**
    - Run the main script to start training and evaluation:
    ```bash
    python main.py
    ```

2. **Review Results:**
    - After training, the script will output model performance metrics and generate visualizations to help evaluate the results.

## Possible Future Plans
- **Enhanced Model Architecture:** Explore more advanced architectures for better accuracy.
- **Additional Classes:** Expand the model to classify other lung diseases.
- **User Interface:** Develop a GUI for easier interaction with the model.

## Contributing
Contributions to this project are welcome! If you have suggestions for improvements or new features, feel free to fork the repository, make your changes, and submit a pull request.

## License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed.

## Acknowledgments
This project was developed as a practical application of machine learning techniques in medical image processing. It is intended to assist in the early detection of pneumonia and improve healthcare outcomes.
