import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse

import sudoku_utils as sutils
from sudoku_solver_class import SudokuSolver


def solve_sudoku_puzzle(args):
    img_fpath = args['img_fpath']
    model_fpath = args['model_fpath']
    
    # Check for valid filepath because cv2.imread fails silently
    if not os.path.exists(img_fpath):
        raise FileNotFoundError (f"File not found: '{img_fpath}'")
    # Load image, change color space from BGR to RGB, and resize
    img = cv2.imread(img_fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = sutils.resize_and_maintain_aspect_ratio(input_image=img, new_width=1000)

    # Plot the original image
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show(block=False)

    # Load the trained model and make prediction
    loaded_model = tf.keras.models.load_model(model_fpath)

    # Locate grid cells in image
    cells, M, board_image = sutils.get_valid_cells_from_image(img)

    # Get the 2D array of the puzzle grid to be passed to the solver
    grid_array = sutils.get_predicted_sudoku_grid(loaded_model, cells)

    # Create an instance of SudokuSolver and try to solve the puzzle
    solver = SudokuSolver(board=grid_array)
    solver.solve()

    # If there are no zeros left, the puzzle is solved. Display annotated image
    if not np.any(solver.board == 0):
        # Get the image of the board annotated with the solution
        print("Success - sudoku solved!")
        annotated_board_img = sutils.generate_solution_image(img, board_image, cells, solver.board, M)
        fig, ax = plt.subplots()
        ax.imshow(annotated_board_img)
        ax.axis('off')
        plt.tight_layout()
        plt.show(block=False)

    else:
        # We failed to solve the board.
        print("Could not solve the puzzle. Check for misclassified digits.\n")

    # Print the board whether or not it's solved
    solver.print_board()



if __name__ == "__main__":
    # Construct an argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_fpath", default="data/sudoku_images/22.jpg", type=str, help="Path to sudoku image file")
    ap.add_argument("--model_fpath", default="models/model_15_epochs_font_mnist.keras", type=str, help="Path to saved Keras CNN model")
    args = vars(ap.parse_args())

    solve_sudoku_puzzle(args)
    plt.show()
