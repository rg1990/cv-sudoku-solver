# Sudoku Solver with Image Processing and Deep Learning

![basic_process_example_weird_colours](https://github.com/rg1990/cv-sudoku-solver/assets/70291897/d727b004-29f8-4665-b35f-921bc217b229)


## Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This is a Python project using image processing and deep learning to solve Sudoku puzzles from natural images. The process is to take an image of a Sudoku puzzle, extract the puzzle grid, identify and classify digits in each cell, solve the puzzle using a recursive backtracking algorithm, and finally display the solution back on the original image. [OpenCV](https://opencv.org/) was used for image processing, and [Keras](https://keras.io/) for the deep learning component.

## Demo
Add a GIF and/or video link here.

## Installation
Follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/sudoku-solver.git
   cd sudoku-solver
2. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate
3. Install the required dependencies:
   ``` bash
   pip install -r requirements.txt

## Usage
Follow the steps below to solve a Sudoku puzzle using the Sudoku Solver:
- Prepare an image of the Sudoku puzzle you want to solve.
- !!! Place the image in the root directory of the project or provide the image file path as an argument.
- Run the solver:

  ``` bash
  # Enter some Python code here

4. The solver will process the image, detect the puzzle grid, classify the digits, solve the puzzle, and display the result.


## Contributing
Contributions are always welcome. If you have any suggestions, bug reports, or improvements, please feel free to create a pull request or open an issue.

## License
This project is open-source software licensed under the MIT License. Feel free to modify and distribute the code as per the terms of this license.

