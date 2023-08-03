# Functions for solving sudoku puzzles

import cv2
import imutils
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sudoku_solver_class import SudokuSolver


def resize_and_maintain_aspect_ratio(input_image, new_width):
    '''
    A simple utility function to resize an image by specifying a new width,
    while maintaining the original aspect ratio.
    
    Args:
        input_image: Array containing image data.
        new_width: Desired width of the resulting reshaped image.
    
    Returns:
        reshaped_image: Array containing reshaped image data.

    '''
    
    # Resize maintaining aspect ratio
    orig_width, orig_height = input_image.shape[1], input_image.shape[0]
    # Determine the aspect ratio
    ratio = new_width / float(orig_width)
    # Calculate the new height to match new_width, given aspect ratio
    new_height = int(orig_height * ratio)
    dim = (new_width, new_height)
    reshaped_image = cv2.resize(input_image, dim, interpolation=cv2.INTER_AREA)
    
    return reshaped_image


def apply_grayscale_blur_and_threshold(img, method="mean", blocksize=91, c=7):
    '''
    Utility function to convert RGB image to thresholded image.
    
    Args:
        img: Array containing image data.
        
        method: Method to use for adaptive thresholding. Options are "mean" or
                "gaussian", corresponding to cv2.ADAPTIVE_THRESH_MEAN_C and
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, respectively.
        
        blocksize: Parameter blockSize passed to cv2.adaptiveThreshold.
                    Block size defines the neighbourhood of pixels considered
                    by adaptive threshold.
        
        c: Parameter C passed to cv2.adaptiveThreshold. Constant value
           subtracted from the weighted sum of neighbourhood pixels.
    
    Returns:
        thresh: Thresholded binary image.
    
    '''
    
    # Apply blur and convert to grayscale
    img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if method == "mean":
        adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method == "gaussian":
        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray,
                                    maxValue=255,
                                    adaptiveMethod=adaptiveMethod,
                                    thresholdType=cv2.THRESH_BINARY,
                                    blockSize=blocksize,
                                    C=c)

    # Invert the resulting threshold image (black <-> white)
    thresh = cv2.bitwise_not(thresh)

    return thresh


def get_quadrilateral_points_in_order(approx_arr):
    '''
    Given an array of [x, y] pairs representing corner locations of a
    quadrilateral, identify which points are top-left (tl), top-right (tr),
    bottom-right (br), and bottom-left (bl) and return them in this exact
    order.
    
    For each point in approx_arr, calculate the Euclidean distance to the
    origin (0, 0). Then, define a second origin that resides past the rightmost
    point in approx_arr. Calculate Euclidean distances between all points and 
    this new origin.
    
    TL and BR are determined by finding the shortest and longest distances
    wrt origin (0, 0). TR and BL are determined by finding the shortest and
    longest distances wrt the new origin, excluding those points already
    assigned to TL and BR (this is important).    

    Args:
        approx_arr: Array of values representing the corner locations of a quadrilateral. 
    
    Returns:
        Array of [x, y] pairs in the order tl, tr, br, bl.
    
    '''
    
    try:
        assert(approx_arr.shape == (4, 1, 2) or approx_arr.shape == (4, 2))
    except:
        raise ValueError(f"Incorrect shape for approx_arr: {approx_arr.shape}. Requires shape of (4, 1, 2) or (4, 2).")
    
    # Squeeze away the redundant dimension to give us a 2D array if necessary
    if approx_arr.shape == (4, 1, 2):
        approx_arr = np.squeeze(approx_arr, axis=1)
    
    # Define a point to place the second origin
    max_x = int(1.1 * np.max(approx_arr[:,0]))
    
    # Define the origin locations
    origin_1 = [0, 0]
    origin_2 = [max_x, 0]

    # Find the Euclidean distances of all 4 points in the array, to the origin (0, 0)
    distances_1 = [np.linalg.norm(point - origin_1) for point in approx_arr]
    # Find the Euclidean distances of all 4 points in the array, to the new origin (max_x, 0)
    distances_2 = [np.linalg.norm(point - origin_2) for point in approx_arr]

    # Find the indices of the top-left and bottom-right corners
    tl_idx = np.argmin(distances_1)
    br_idx = np.argmax(distances_1)
    
    # To avoid identifying an index more than once, we set these to np.inf
    # or -np.inf when selecting the indices for the distances from origin_2
    
    # Identify the top right corner
    dist_arr = distances_2.copy()
    # Mask out the indices that have already been selected
    dist_arr[tl_idx] = np.inf
    dist_arr[br_idx] = np.inf
    tr_idx = np.argmin(dist_arr)
    
    # Identify the bottom left corner
    dist_arr = distances_2.copy()
    # Mask out the indices that have already been selected
    dist_arr[tl_idx] = -np.inf
    dist_arr[br_idx] = -np.inf
    bl_idx = np.argmax(dist_arr)
    
    # Get the (x, y) points for each corner
    tl = approx_arr[tl_idx]
    br = approx_arr[br_idx]
    tr = approx_arr[tr_idx]
    bl = approx_arr[bl_idx]
    
    return np.array([tl, tr, br, bl])


def perform_four_point_transform(input_img, src_corners, pad=10):
    '''
    Perform a four-point perspective transform to an image such that the four
    corners in img are mapped to new specified points in the resulting image. 

    Args:
        input_img: An image array.
        src_corners: An array of shape (4, 2) containing the (x, y) locations
                     of the four reference points in input_img.
        pad: Pixel value for padding applied to all sides of warped image.
             The warped image then contains pixels extending past the corners,
             which is useful for accommodating curves in the paper surface for
             locating the puzzle grid later.
    
    Returns:
        M: transformation matrix.
        warped: Image resulting from the perspective transform applied to
                input_img.
    
    '''
    
    # Get the corner points in the order we want
    src_corners = get_quadrilateral_points_in_order(src_corners)
    src_corners = src_corners.astype("float32")
    tl, tr, br, bl = src_corners
    
    # Define the desired dimensions of the destination (warped) image (max_width, max_height)
    # Calculate the width of the top and bottom edges, and take the maximum
    bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(bottom_width), int(top_width))

    # Calculate the height of the left and right edges, and take the maximum
    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(left_height), int(right_height))

    # Define the set of points in the destination image that our four corners 
    # from the input image should map to, such that we obtain a bird's eye view
    dest_img_corners = np.array([[0+pad, 0+pad],
                                 [max_width-1-pad, 0+pad],
                                 [max_width-1-pad, max_height-1-pad],
                                 [0+pad, max_height-1-pad]], dtype="float32")

    # Compute our transformation matrix
    M = cv2.getPerspectiveTransform(src=src_corners, dst=dest_img_corners)
    warped_img = cv2.warpPerspective(input_img, M, (max_width, max_height))

    return M, warped_img


def find_grid_contour_candidates(img, to_plot=False):
    '''
    Given an input image, build a list of contours that may represent the
    main puzzle grid outline. For each candidate identified, perform a
    perspective transformation and store the contour, the transformation
    matrix, and the perspective-warped image.
    
    Args:
        img: RGB image of sudoku puzzle.
        to_plot: Flag to produce plot showing all contours.
        
    Returns:
        contours: A list of contours identified as potential grid candidates.
                    Sorted in reverse order by area of contour.
        M: A list of perspective transformation matrices.
        warped: A list of warped images that may represent the main puzzle grid.
        
    '''
    
    # Initialise lists for return values
    M_matrices = []
    warped_images = []
    contour_grid_candidates = []

    # Calculate the area of the whole image
    img_area = img.shape[0] * img.shape[1]

    # Apply blur, grayscale and adaptive threshold
    thresh = apply_grayscale_blur_and_threshold(img, blocksize=41, c=8)

    # Get the contours from the thresholded image
    contours = cv2.findContours(image=thresh.copy(),
                                mode=cv2.RETR_EXTERNAL,
                                method=cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Convenience function to extract contours
        contours = imutils.grab_contours(contours)
        # Sort the contours according to contour area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        if to_plot:
            # Plot all the contours on the original RGB image
            with_contours = cv2.drawContours(img.copy(), contours, -1, (0, 255, 75), thickness=2)
            plt.imshow(with_contours)
            plt.show(block=False)
        
        for contour in contours:
            # Approximate the contour in order to determine whether the contour is a quadrilateral
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            
            # Calculate the area of the identified contour
            contour_area = cv2.contourArea(contour)
            contour_fractional_area = contour_area / img_area
            
            # We are looking for a quadrilateral contour with sufficiently large area
            if len(approx) == 4 and contour_fractional_area > 0.1:
                # Get the corners in the required order
                approx = get_quadrilateral_points_in_order(approx)
                # Use the approximation to apply the perspective transform
                # on the candidate grid region
                M, warped_img = perform_four_point_transform(input_img=img,
                                                              src_corners=approx,
                                                              pad=30)
                
                M_matrices.append(M)
                warped_images.append(warped_img)
                contour_grid_candidates.append(contour)
                        
    if warped_images:
        return M_matrices, warped_images, contour_grid_candidates
    else:
        raise Exception("No grid contour candidates were found in image")


def check_for_digit_in_cell_image(img, area_threshold=5, apply_border=False):
    '''
    Determine whether or not a digit is present in an image of a single
    sudoku cell. If a contour is located whose area exceeds area_threshold,
    it is determined that this contour represents a digit.
    
    Args:
        img: An image of a single sudoku cell.
        area_threshold: Threshold value whose units are percentage of image area.
                        A contour is considered a digit if its area exceeds
                        the threshold.
                        
        apply_border: Whether or not to apply a mask to remove non-digit pixels
                      around the edges of the cell image.
            
    Returns:
        image_contains_digit: Boolean - whether the cell is considered to contain a digit.
        cell_img: Image of the sudoku cell, optionally with border masked out.
    
    '''
    
    cell_img = img.copy()
    
    if apply_border:
        # Crude way to eliminate the unwanted pixels around the borders
        border_fraction = 0.07
        replacement_val = 0
        
        y_border_px = int(border_fraction * cell_img.shape[0])
        x_border_px = int(border_fraction * cell_img.shape[1])
        
        cell_img[:, 0:x_border_px] = replacement_val
        cell_img[:, -x_border_px:] = replacement_val
        cell_img[0:y_border_px, :] = replacement_val
        cell_img[-y_border_px:, :] = replacement_val
    
    # Get the contours for the image
    contours = cv2.findContours(image=cell_img,
                                mode=cv2.RETR_TREE,
                                method=cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    if len(contours) > 0:
        # Sort the contours according to contour area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)        
        largest_contour_area = cv2.contourArea(contours[0])
        image_area = cell_img.shape[0] * cell_img.shape[1]
        contour_percentage_area = 100 * largest_contour_area / image_area
        
        if contour_percentage_area > area_threshold:
            image_contains_digit = True
        else:
            image_contains_digit = False
        
    else:
        image_contains_digit = False
        
    return image_contains_digit, cell_img


def locate_cells_within_grid(grid_img, to_plot=False):
    '''
    Identify the individual sudoku cells contained within the grid image.
    
    Args:
        grid_img:
        to_plot:
            
    Returns:
        valid_cells:
    
    '''
    
    # Initialise a list to store the detected cells
    valid_cells = []
    
    # Calculate the area of the sudoku grid.
    # Used later to check if a contour is a valid cell
    grid_area = grid_img.shape[0] * grid_img.shape[1]
    
    # Apply blur, grayscale and adaptive threshold
    grid_img = apply_grayscale_blur_and_threshold(grid_img, method="mean", blocksize=91, c=7)
    
    if to_plot:
        fig, ax = plt.subplots()
        ax.imshow(grid_img, cmap='gray')
        plt.show(block=False)
    
    # Get external and internal contours from the sudoku grid image
    contours = cv2.findContours(image=grid_img.copy(),
                                mode=cv2.RETR_TREE,
                                method=cv2.CHAIN_APPROX_NONE)
    
    if contours:
        # Convenience function to extract contours
        contours = imutils.grab_contours(contours)
        # Sort the contours according to contour area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            # Approximate the contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            # Find the contour area wrt the grid area
            contour_fractional_area = cv2.contourArea(contour) / grid_area
            
            # We are looking for a contour that is quadrilateral and has an area
            # of approximately 1% of the grid area (0.5% to 1.5% works well)
            if len(approx) == 4 and contour_fractional_area > 0.005 and contour_fractional_area < 0.015:
                # We have found a valid contour
                # Use a mask to extract the identified cell from grid_img
                mask = np.zeros_like(grid_img)
                # Use the contour to mask out the cell
                cv2.drawContours(image=mask,
                                contours=[contour],
                                contourIdx=0,
                                color=255,
                                thickness=cv2.FILLED)
                
                # Get the indices where the mask is white
                y_px, x_px = np.where(mask==255)
                # Use these indices to crop out the cell from the image
                cell_image = grid_img[min(y_px):max(y_px)+1, min(x_px):max(x_px)+1]
                # Determine whether or not there's a digit present in the cell
                digit_is_present, cell_image = check_for_digit_in_cell_image(img=cell_image,
                                                                             area_threshold=5,
                                                                             apply_border=True)
                
                # See if erosion improves predictions
                kernel = np.ones((3, 3), np.uint8)
                cell_image = cv2.erode(cell_image, kernel, iterations=1)
                
                # Resize the cell image to be 28x28 pixels for classification later
                cell_image = cv2.resize(cell_image, dsize=(28, 28), interpolation=cv2.INTER_AREA)
                
                # Use the contour to calculate the centroid for the cell. This is
                # so we know the cell's location on the grid, so we know where to
                # place the text when entering the puzzle solution
                moments = cv2.moments(contour)
                x_centroid = int(moments['m10'] / moments['m00'])
                y_centroid = int(moments['m01'] / moments['m00'])
    
                # Create a dictionary for every valid cell
                valid_cells.append({'img': cell_image,
                                    'contains_digit': digit_is_present,
                                    'x_centroid': x_centroid,
                                    'y_centroid': y_centroid})
                        
    else:
        print("No valid cells found in image")
    
    return valid_cells

    
def get_valid_cells_from_image(img):
    '''
    The aim is to locate and extract the puzzle grid from an image.
    First, identify all candidates for the puzzle grid, then search within
    these candidates for valid individual grid cells. If no candidate puzzle
    grid contains exactly 81 cells, the search has failed and an exception
    is raised.
    
    Args:
        img: RGB image of a sudoku puzzle.
            
    Returns:
        valid_cells: A list of dictionary objects, one for each cell.
                     These contain the cell's image data, whether or not it
                     contains a digit, and the cell's x- and y-centroids in
                     the perspective warped image.
        
        M: The perspective transform matrix used to transform the identified
            grid to the bird's-eye view. This will be used for inverse
            transform when placing solved digits back on the original grid image.
            
        
        grid_image: The perspective transformed puzzle grid, isolated from the rest
                    of the image.
            
    '''
    # Find grid contour candidates for a particular image
    M_matrices, warped_images, grid_candidates = find_grid_contour_candidates(img)
    
    if not warped_images:
        raise Exception("No grid candidates were found in the image.")
    
    # For each candidate grid, check if it's valid and get inverse transform matrix
    for i, grid_image in enumerate(warped_images):
        valid_cells = locate_cells_within_grid(grid_image)
        M = M_matrices[i]
        if len(valid_cells) == 81:
            # Assume we have found the puzzle grid
            valid_cells = sort_cells_into_grid(valid_cells)
            plot_cell_images_in_grid(valid_cells)
            return valid_cells, M, grid_image
    
    # If we don't obtain a list of valid cells, we don't return anything
    raise Exception("Unable to find the required number of cells in image.")


def sort_cells_into_grid(cells):
    '''
    Given a list of cells, use the x and y centroid values to arrange the
    cells into the same order as the original puzzle grid (left to right,
    top to bottom). This sorted list is used later to construct a 2D numpy
    array representing the board, which is passed to a solver.
    
    Each cell is a dictionary containing the cell image, and x- and y-centroids.
    
    Args:
        cells: List of dictionary objects, each containing cell information.
    
    Returns:
        sorted_cells_list: List of cell dictionaries, sorted such that the
                            cells are in the same order as the original puzzle
                            grid (left to right, top to bottom).
    
    '''
    
    x_vals = [cell['x_centroid'] for cell in cells]
    y_vals = [cell['y_centroid'] for cell in cells]
    
    # Get a 2D array of all x, y points from the dictionary
    points = np.array([[cell['x_centroid'], cell['y_centroid']] for cell in cells])
    # Sort by x value
    points_sorted = np.array(sorted(points, key=lambda x: x[1]))
    # Reshape to give an array with 9 rows
    rows = np.reshape(points_sorted, newshape=(9, 9, 2))
    # Sort by y value for every row in rows
    final = np.array([sorted(row, key=lambda x: x[0]) for row in rows])
    
    # Make sure all value combinations in final are actually present in the dictionary
    final_reshaped = np.reshape(final, newshape=(81, 2))
    for i in range(len(x_vals)):
        assert any(np.equal(final_reshaped, [x_vals[i], y_vals[i]]).all(1))
        
    # Find the index of the cell in cells (list) for each point in final_reshaped
    indices = []
    for x, y in final_reshaped:
        x_indices = np.where(np.array(x_vals) == x)
        y_indices = np.where(np.array(y_vals) == y)
        index = np.intersect1d(x_indices, y_indices)[0]
        indices.append(index)
    
    # Sort the cells list according to the indices list
    sorted_cells_list = [cells[idx] for idx in indices]
    return sorted_cells_list


def plot_cell_images_in_grid(cells):
    '''
    A quick utility function to take all the cell images from a list and plot
    them in a grid in the same layout as the original puzzle.
    
    Args:
        cells: List of dictionary objects, each containing one cell image.
    
    '''
    # Create a blank image (cell images have side 28px)
    width, height = 9*28, 9*28
    main_img = np.zeros((height, width))
    
    for i, cell in enumerate(cells):
        # Get row and col of cell, used to determine position in main_img
        row, col = np.array(divmod(i, 9))
        # Make a copy of the cell image and place it on main_img
        cell_image = cells[i]['img'].copy()
        main_img[row*28:(row+1)*28, col*28:(col+1)*28] = cell_image
    
    fig, ax = plt.subplots()
    ax.imshow(main_img, cmap='gray')
    plt.show(block=False)


# This was previously called get_cell_predictions
def get_predicted_digits_and_sudoku_grid(model, cells):
    '''
    Use a model to make predictions about which digit is present in each cell
    containing a digit. Use these predictions to construct a 2D array 
    representing the sudoku board, where blank squares are assigned a value of 0.
    
    Args:
        model: A trained Keras model for classifying digits 1 to 9 in images.
        cells: A 1D list of cells sorted in the same order as in the puzzle,
                read left to right, top to bottom.
            
    Returns:
        pred_labels: An array of integers representing predicted digits.
        grid_array: A 2D array representing the sudoku board, containing
                    the predicted digits and using zeros for empty cells.
    
    '''
    # Extract the digit images from the list of identified cells
    digit_images = np.array([np.expand_dims(cell['img'], -1) for cell in cells if cell['contains_digit']])
    # Predict the digit in each image
    pred_labels = model.predict(digit_images)
    pred_labels = np.argmax(pred_labels, axis=1) + 1
    # Construct a sudoku grid containing digit classification predictions
    indices = np.where([cell['contains_digit'] for cell in cells])[0]
    grid_array = np.zeros((81), dtype=int)
    grid_array[indices] = pred_labels
    grid_array = np.reshape(grid_array, newshape=(9, 9))
    return pred_labels, grid_array


def generate_solution_image(full_image, board_image, cells_list, solved_board_arr, M_matrix):
    '''
    Annotate the original image of the sudoku board with the solution.
    
    Args:
        full_image:
        board_image:
        cells_list:
        solved_board_arr:
        M_matrix:
        
    Returns:
        annotated: Original RGB puzzle image, annotated with solutions.
    
    '''
    
    # Specify the font used for annotating the solutions
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Create a white grid the same shape as the warped image, to place solutions
    solution_img = np.ones_like(board_image) * 255
    # Flatten the solved 2D board array
    flattened_board_array = solved_board_arr.reshape((-1))
    
    # Place the solved digits from blank puzzle cells on the blank white image
    for i, cell in enumerate(cells_list):
        if not cell['contains_digit']:
            # Get cell centroids for text positioning
            x_pos = cell['x_centroid']
            y_pos = cell['y_centroid']
            text = str(flattened_board_array[i])
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            # Specify placement of text
            text_x = int((x_pos - textsize[0] / 2))
            text_y = int((y_pos + textsize[1] / 2))
            # Annotate number with black text
            cv2.putText(solution_img, text, (text_x, text_y), font, 1.3, (0, 0, 0), 2)
    
    # Apply the inverse perspective transform to the solution image
    unwarped_img = cv2.warpPerspective(
        solution_img,
        M_matrix,
        (full_image.shape[1], full_image.shape[0]),
        flags=cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    # Copy original image to annotate with the solution digits
    annotated = full_image.copy()
    # Locate black pixels (number text) and change colour to red
    annotated[np.where(unwarped_img[:,:,0] == 0)] = (255, 15, 0)
    
    return annotated
