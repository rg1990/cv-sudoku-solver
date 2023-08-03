# Sudoku Solver with Computer Vision and Deep Learning

## Introduction
This is a Python project using computer vision and deep learning to solve sudoku puzzles from natural images. The process is to take an image of a sudoku puzzle, extract the puzzle grid, identify and classify digits in each cell, solve the puzzle using a recursive backtracking algorithm, and finally display the solution back on the original image. [OpenCV](https://opencv.org/) was used for image processing, and [Keras](https://keras.io/) for the deep learning component.

A number of sample sudoku images with varied lighting, angles, and backgrounds, are provided in `data/sudoku_images/`. Please try using your own images too!

TODO - Add some information about the image processing steps, the data used for training, and the network.

## Demo
![sudoku_summary_image](https://github.com/rg1990/cv-sudoku-solver/assets/70291897/bbe14f3c-5db0-43d7-8e0a-da9457b5fb02)
![sudoku_summary_image_2](https://github.com/rg1990/cv-sudoku-solver/assets/70291897/81810852-5189-4cb4-b099-c7a3de96faea)


Left: Original image of unsolved puzzle.<br>
Centre: Adaptive thresholding applied to perspective-warped image of extracted puzzle grid.<br>
Right: Original puzzle image annotated with solutions.


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
Follow the steps below to use the solver with your own sudoku image:
- Prepare an image of the sudoku puzzle you want to solve.
- Place the image in the directory `data/sudoku_images/`, or provide the image file path as the `--img_fpath` argument.
- Run the solver:

  ``` bash
  python sudoku_main.py --img_fpath "data/sudoku_images/22.jpg"

4. The solver will process the image, detect the puzzle grid, classify the digits, solve the puzzle, and display the result.

## Notes
There are some images included in this repo, for which the solver fails. This can be due to one of two reasons: (1) The image processing component was unable to properly detect the grid, or (2) the deep learning model wrongly classified a digit in the sudoku puzzle, rendering the resulting puzzle unsolvable.

## Contributing
Contributions are always welcome. If you have any suggestions, bug reports, or improvements, please feel free to create a pull request or open an issue.

## License
This project is open-source software licensed under the MIT License. Feel free to modify and distribute the code as per the terms of this license.

