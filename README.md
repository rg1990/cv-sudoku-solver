# Sudoku Solver with Computer Vision and Deep Learning

## Introduction
This is a Python project using computer vision and deep learning to solve sudoku puzzles from natural images. The process is to take an image of a sudoku puzzle, extract the puzzle grid, identify and classify digits in each cell, solve the puzzle using a recursive backtracking algorithm, and finally display the solution back on the original image. [OpenCV](https://opencv.org/) was used for image processing, and [Keras](https://keras.io/) for the deep learning component.

A number of sample sudoku images with varied lighting, angles, and backgrounds, are provided in `data/sudoku_images/`. Please try using your own images too!

## Demo
Here is a GIF that illustrates the process:<br>
![gif_draft_6_smaller-min](https://github.com/rg1990/cv-sudoku-solver/assets/70291897/8019e24c-edb3-4dbd-9adf-083936127012)

## Implementation Details
For image processing, an adaptive threshold is applied to get a binary image. The contours of the binary image are computed and used to locate the main puzzle grid. A perspective transform is applied to obtain a bird's-eye view of the puzzle grid. Then, the contours within the transformed grid are computed to locate individual cells. We determine which cells contain digits, and store some information about each cell, to reconstruct the sudoku grid. After the puzzle is solved, the solution numbers are added to the grid cells, and the inverse perspective transform is applied to place the solution back onto the original image.

For digit classification, a simple convolutional neural network was constructed and trained using Keras. The training data consisted of images of the numbers 1 to 9 in various fonts (~9k images), as well as the numbers 1 to 9 from the MNIST hand-written digits dataset (~55k images). Some of the font images were removed in order to improve the model's performance classifying the digits in the sudoku images (none of which were seen during training.)

For puzzle solving, a 2D array representing the puzzle grid is passed to an instance of the `SudokuSolver` class. The grid contains the model's digit class predictions for the populated cells, and zeros where the cells are blank. A recursive backtracking algorithm is used to solve the sudoku puzzle. The solver returns only one solution, even if more than one solution exists.

## Installation
Follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/rg1990/sudoku-solver.git
   cd sudoku-solver
2. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate
3. Install the required dependencies:
   ``` bash
   pip install -r requirements.txt

## Usage
The Keras model can be trained by running `train.py`. There is already a trained model in the `models` folder, so training is optional.<br>

Using the `--data` argument, you can choose whether to use font data, MNIST data, or both, by passing the values `"fonts"`, `"mnist"`, or `"both"`, respectively. Training parameters can be specified using `--batch_size` and `epochs`. Trained models are saved in the `models` folder and the model save path can be specified using `--model_save_fpath`. Example:
   ``` bash
   python train.py --data "fonts" --epochs 10 --batch_size 128 --model_save_fpath "models/my_trained_model.keras"
   ```


Follow the steps below to use the solver with your own sudoku image:
- Prepare an image of the sudoku puzzle you want to solve.
- Place the image in the directory `data/sudoku_images/`, or provide the image file path as the `--img_fpath` argument.
- Run the solver:

  ``` bash
  python sudoku_main.py --img_fpath "data/sudoku_images/22.jpg"

- The solver will process the image, detect the puzzle grid, classify the digits, try to solve the puzzle, and display the result.

## Notes
There are some images included in this repo for which the solver fails. This can be due to one of two reasons: (1) The image processing component was unable to properly detect the grid, or (2) the deep learning model wrongly classified a digit in the sudoku puzzle, rendering the resulting puzzle unsolvable.

## Contributing
Contributions are always welcome. If you have any suggestions, bug reports, or improvements, please feel free to create a pull request or open an issue.

## License
This project is open-source software licensed under the MIT License. Feel free to modify and distribute the code as per the terms of this license.

