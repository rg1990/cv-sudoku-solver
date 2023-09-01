import pygame
from pygame import Vector2
from pygame import Rect
import numpy as np
import sys
import copy
import cv2
import tensorflow as tf

from sudoku_solver_class import SudokuSolver
import sudoku_utils as sutils


class Button:
    def __init__(self, ui, text, x_pos, y_pos, width, height, enabled):
        self.ui = ui
        self.text = text
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.enabled = enabled
        
        self.button_width = width
        self.button_height = height
        
        self.draw()
        
        
    def draw(self):
        # Define colours based on clicked and enabled states
        if self.enabled:
            button_bg_colour = "dark gray" if self.check_click() else "light gray"
            fg_colour = "black"
        else:
            button_bg_colour = "light gray"
            fg_colour = "dark gray"
            
            
        font = pygame.font.Font("freesansbold.ttf", 18)
        button_text = font.render(self.text, True, fg_colour)
        button_text_rect = button_text.get_rect()
        text_width, text_height = button_text_rect.width, button_text_rect.height        
        
        text_x_offset = (self.button_width - text_width) // 2
        text_y_offset = (self.button_height - text_height) // 2
        
        button_rect = Rect((self.x_pos, self.y_pos), (self.button_width, self.button_height))
        
        # Draw the button with rounded corners
        pygame.draw.rect(self.ui.window, button_bg_colour, button_rect, 0, 3)
        # Create a border around the button
        pygame.draw.rect(self.ui.window, fg_colour, button_rect, 2, 3)
        self.ui.window.blit(button_text, (self.x_pos + text_x_offset, self.y_pos + text_y_offset))


    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()
        left_click = pygame.mouse.get_pressed()[0]
        button_rect = Rect((self.x_pos, self.y_pos), (self.button_width, self.button_height))
        
        if left_click and button_rect.collidepoint(mouse_pos) and self.enabled:
            return True
        else:
            return False


class Cell:
    def __init__(self):
        self.selected = False
        self.num = 0
        self.is_fixed = False if self.num == 0 else True


class GameState():
    def __init__(self):
        # Specify which files contain which difficulty of puzzles
        self.puzzle_files = {"easy": 0,
                             "medium": 2,
                             "hard": 4}
        self.difficulty = "easy"
        
        # Load all the puzzles from the files
        self.puzzle_dict = {k: self.load_sudoku_puzzles(v) for k, v in self.puzzle_files.items()}
        self.puzzles = self.puzzle_dict[self.difficulty]
        #self.puzzles = self.load_sudoku_puzzles(self.puzzle_files[self.difficulty])
        
        # Define the size of the board
        self.world_size = Vector2(9, 9)        
        # Create the 2D array that holds the Cell objects
        self.board = np.array([[Cell() for _ in range(int(self.world_size.x))] for _ in range(int(self.world_size.y))])
        # Currently selected cell - nothing to begin with
        self.sel_row = None
        self.sel_col = None
        self.current_puzzle = None
        self.solved_puzzle = None
        self.hint_requested = False
        self.hint_row = None
        self.hint_col = None
        
        # Answer check
        self.check_requested = False
        self.wrong_coords = []
        self.right_coords = []
        
        self.is_solved = False
        self.start_time = None
        
        # Load the trained model to make predictions on digits
        model_path = "models/model_15_epochs_font_mnist.keras"
        self.trained_model = tf.keras.models.load_model(model_path)
        
    
    # Use property decorator to introduce world_width and world_height
    @property
    def world_width(self):
        return int(self.world_size.x)
    
    
    @property
    def world_height(self):
        return int(self.world_size.y)
       
    
    def update_puzzle_difficulty(self, new_difficulty):
        if new_difficulty in list(self.puzzle_files.keys()):
            self.difficulty = new_difficulty
            self.puzzles = self.puzzle_dict[self.difficulty]
        else:
            self.puzzles = self.puzzle_dict["easy"]
    
    
    def load_sudoku_puzzles(self, puzzle_fnum=0):
        txt_fpath = f"data/sudoku_puzzles/{puzzle_fnum}.txt"
        with open(txt_fpath, 'r') as f:
            puzzles = f.read().split('\n')
        reshaped = [self.reshape(puzzle, 9, 9) for puzzle in puzzles]
        reshaped_puzzle_list = [[[int(char) for char in row] for row in puzzle] for puzzle in reshaped]
        return reshaped_puzzle_list
        
 
    def reshape(self, board: str, num_rows: int, num_cols: int):
        if num_rows * num_cols != len(board):
            print(f"Board of length {len(board)} cannot be reshaped into shape ({num_rows}, {num_cols})")
            return

        reshaped = [board[(current_row*num_cols):(current_row*num_cols)+num_cols] for current_row in range(num_rows)]
        return reshaped
    
    
    def blank_out_board(self):
        ''' Set all entries on the board to zero so no numbers show '''
        self.board = np.array([[Cell() for _ in range(int(self.world_size.x))] for _ in range(int(self.world_size.y))])
    
    
    def reset_current_puzzle(self):
        ''' Set all entries to zero except the "fixed" numbers in the current puzzle '''
        for y in range(int(self.world_size.y)):
            for x in range(int(self.world_size.x)):
                if not self.board[y][x].is_fixed:
                    self.board[y][x].num = 0
    
    
    def get_new_puzzle(self):
        # Get a new puzzle from self.puzzles list
        self.puzzles = self.puzzle_dict[self.difficulty]
        idx = np.random.choice(len(self.puzzles), replace=False)
        self.current_puzzle = self.puzzles[idx]
        # Store the solution for the new puzzle
        board_copy = copy.deepcopy(self.current_puzzle)
        solver = SudokuSolver(board_copy)
        solver.solve()
        self.solved_puzzle = solver.board
                
        
    def get_puzzle_from_array(self, arr):
        # Use the 2D array from image processing to update the game board
        self.current_puzzle = arr
        # Store the solution for the new puzzle
        board_copy = copy.deepcopy(self.current_puzzle)
        solver = SudokuSolver(board_copy)
        solver.solve()
        self.solved_puzzle = solver.board
        
        
    def initialise_board(self):
        # Update the numbers in the Cell objects in self.board
        self.blank_out_board()
        for y in range(int(self.world_size.y)):
            for x in range(int(self.world_size.x)):
                self.board[y][x].num = self.current_puzzle[y][x]
                if self.current_puzzle[y][x] != 0:
                    self.board[y][x].is_fixed = True
    
    
    def add_puzzle_solution_to_board(self):
        for y in range(int(self.world_size.y)):
            for x in range(int(self.world_size.x)):
                # Find the cells the user can edit (retains colour formatting)
                if not self.board[y][x].is_fixed:
                    self.board[y][x].num = self.solved_puzzle[y][x]
    
    
    def get_empty_cell_coords(self):
        zero_coords = []
        for y, row in enumerate(self.board):
            for x, col in enumerate(self.board[y]):
                if self.board[y][x].num == 0:
                    zero_coords.append([y, x])
        return zero_coords
    
    
    def update_hint_on_board(self):
        
        # Find the cells without numbers
        zero_coords = self.get_empty_cell_coords()
        # If the puzzle is solved, we can't add a hint
        if zero_coords:
            if self.hint_requested:
                idx = np.random.choice(range(len(zero_coords)))
                self.hint_row, self.hint_col = zero_coords[idx]
                self.board[self.hint_row][self.hint_col].num = self.solved_puzzle[self.hint_row][self.hint_col]
            else:
                self.board[self.hint_row][self.hint_col].num = 0
        
    
    def check_user_answers(self):
        '''
        For each number the user has typed, check if it is right or wrong
        If it is wrong, add the grid coords to a list so the UI can render
        the cell(s) using different colours
        '''
        
        if self.solved_puzzle is None:
            return
        
        self.wrong_coords = []
        self.right_coords = []
        
        for y in range(int(self.world_size.y)):
            for x in range(int(self.world_size.x)):
                # Find the cells the user has entered numbers into
                if not self.board[y][x].is_fixed and self.board[y][x].num != 0:
                    # Determine whether the user's answer is right or wrong
                    if self.board[y][x].num == self.solved_puzzle[y][x]:
                        self.right_coords.append([y, x])
                    else:
                        self.wrong_coords.append([y, x])
    
    
    def is_puzzle_solved(self):
        # Get a 2D array of the current board
        current_entries = np.array([[self.board[row][col].num for col in range(self.world_width)] for row in range(self.world_height)])
        if np.all(current_entries == self.solved_puzzle):
            return True
        else:
            return False
    
    
    def update(self, row, col, move_cell_command, num=None):

        new_row = row
        new_col = col
        
        if move_cell_command.x != 0 or move_cell_command.y != 0:
            # Change which cell is selected according to move_cell_command
            if row + move_cell_command.y < 0 or row + move_cell_command.y >= self.world_size.y \
                or col + move_cell_command.x < 0 or col + move_cell_command.x >= self.world_size.x:
                    #print("Invalid arrow key move")
                    return
            else:
                new_row = int(row + move_cell_command.y)
                new_col = int(col + move_cell_command.x)
        
        # Set the selected state of the currently selected cell to False
        if self.sel_row is not None and self.sel_col is not None:
            self.board[self.sel_row][self.sel_col].selected = False
        
        # Set the new cell's selected state to True
        if new_row is not None and new_col is not None:
            self.board[new_row][new_col].selected = True
            # Update the currently selected cell in GameState
            self.sel_row = new_row
            self.sel_col = new_col
            
        if num is not None:
            self.board[new_row][new_col].num = num
            
        # Check the solved status of the puzzle
        self.solved = self.is_puzzle_solved()
            
    

class UserInterface():
    def __init__(self):
        pygame.init()
        
        self.game_state = GameState()
        self.cell_size = Vector2(64, 64)
        self.move_cell_command = Vector2(0, 0)
        
        # Define the window
        window_size = self.game_state.world_size.elementwise() * self.cell_size
        self.window = pygame.display.set_mode((int(window_size.x)+300, int(window_size.y)))
        pygame.display.set_caption("Sudoku")
        
        # There will initially be no selected cell
        self.row = None
        self.col = None
        self.number = None
        
        # Instantiate the buttons
        y_offset = 64
        self.easy_button = Button(ui=self, text="Easy", x_pos=600, y_pos=14+y_offset, width=75, height=45, enabled=True)
        self.medium_button = Button(ui=self, text="Medium", x_pos=687, y_pos=14+y_offset, width=75, height=45, enabled=True)
        self.hard_button = Button(ui=self, text="Hard", x_pos=775, y_pos=14+y_offset, width=75, height=45, enabled=True)
        self.new_game_button = Button(ui=self, text="New Puzzle", x_pos=600, y_pos=74+y_offset, width=250, height=45, enabled=True)
        self.reset_button = Button(ui=self, text="Reset Puzzle", x_pos=600, y_pos=138+y_offset, width=250, height=45, enabled=True)
        self.hint_button = Button(ui=self, text="Show Hint", x_pos=600, y_pos=202+y_offset, width=250, height=45, enabled=True)
        self.check_button = Button(ui=self, text="Check Answers", x_pos=600, y_pos=266+y_offset, width=250, height=45, enabled=True)
        self.solution_button = Button(ui=self, text="Show Solution", x_pos=600, y_pos=330+y_offset, width=250, height=45, enabled=True)
        self.quit_button = Button(ui=self, text="Quit", x_pos=600, y_pos=520, width=250, height=45, enabled=True)
        
        
        # Define some colours
        self.WINDOW_BG_COLOUR = (180, 180, 180)
        self.ROW_COL_CELL_COLOUR = (220, 220, 220)
        self.SELECTED_CELL_COLOUR = (160, 160, 160)
        self.UNSELECTED_CELL_COLOUR = (255, 255, 255)
        self.HINT_CELL_COLOUR = (252, 231, 134)
        self.CORRECT_COLOUR = (114, 252, 127)
        self.INCORRECT_COLOUR = (252, 114, 114)
        
        # Loop properties
        self.clock = pygame.time.Clock()
        self.running = True
        
    
    @property
    def cell_width(self):
        return int(self.cell_size.x)
    
    
    @property
    def cell_height(self):
        return int(self.cell_size.y)
    
    
    def process_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:    
                if event.button == 1:
                    # Get the indices of the new row and col based on click position
                    dest_row = event.pos[1] // self.cell_height
                    dest_col = event.pos[0] // self.cell_width
                    
                    # Check if we clicked within the bounds of the board
                    if (dest_row < self.game_state.world_height) \
                    and (dest_col < self.game_state.world_width):
                        self.row = dest_row
                        self.col = dest_col
                        
                    # Check if any of the buttons are being pressed
                    if self.quit_button.check_click():
                        pygame.quit()
                        sys.exit()
                        
                    # Get a new puzzle and start a new game
                    if self.new_game_button.check_click():
                        self.game_state.get_new_puzzle()
                        self.game_state.initialise_board()
                        
                    # Reset the current puzzle
                    if self.reset_button.check_click():
                        # Remove any hints or user-entered numbers
                        self.game_state.reset_current_puzzle()
                        # Reset the hint state and button text
                        self.game_state.hint_requested = False
                        self.hint_button.text = "Show Hint"
                    
                    # Handle a request for a hint
                    if self.hint_button.check_click():
                        # We need a puzzle before we can provide hints
                        if self.game_state.solved_puzzle is not None:
                            # Toggle hint request state
                            self.game_state.hint_requested = not self.game_state.hint_requested
                            self.game_state.update_hint_on_board()
                            
                            if self.game_state.hint_requested:
                                self.hint_button.text = "Hide Hint"
                            else:
                                self.hint_button.text = "Show Hint"
                    
                        
                    # Handle a request for the solution
                    if self.solution_button.check_click():
                        if self.game_state.solved_puzzle is not None:
                            self.game_state.add_puzzle_solution_to_board()
                            
                    
                    # Update game dificulty - eliminate this copy and paste
                    if self.easy_button.check_click() and self.game_state.difficulty != "easy":
                        self.game_state.difficulty = "easy"
                        self.game_state.get_new_puzzle()
                        self.game_state.initialise_board()
                        
                    if self.medium_button.check_click() and self.game_state.difficulty != "medium":
                        self.game_state.difficulty = "medium"
                        self.game_state.get_new_puzzle()
                        self.game_state.initialise_board()
                        
                    if self.hard_button.check_click() and self.game_state.difficulty != "hard":
                        self.game_state.difficulty = "hard"
                        self.game_state.get_new_puzzle()
                        self.game_state.initialise_board()
            
            
            # Listen for number key presses
            if event.type == pygame.KEYDOWN:
                # Don't allow the provided numbers to be changed or deleted
                if not self.game_state.board[self.row][self.col].is_fixed:
                    # Num pad and num keys above alpha keys
                    if (pygame.K_KP1 <= event.key <= pygame.K_KP9) or (pygame.K_1 <= event.key <= pygame.K_9):
                        self.number = int(event.unicode)
                        
                    if event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
                        self.number = 0
                        
                # Try to navigate using the arrow keys
                if event.key == pygame.K_RIGHT:
                    self.move_cell_command.x = 1
                elif event.key == pygame.K_LEFT:
                    self.move_cell_command.x = -1
                elif event.key == pygame.K_DOWN:
                    self.move_cell_command.y = 1
                elif event.key == pygame.K_UP:
                    self.move_cell_command.y = -1
                        
    
            # Check for files dropped into the window
            if event.type == pygame.DROPFILE:
                path = event.file
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = sutils.resize_and_maintain_aspect_ratio(input_image=img, new_width=1000)
                print("Processing image...")
                try:
                    # Locate grid cells in image
                    cells, M, board_image = sutils.get_valid_cells_from_image(img)
                    # Get the 2D array of the puzzle grid to be passed to the solver
                    grid_array = sutils.get_predicted_sudoku_grid(self.game_state.trained_model, cells)
                    
                    # Check if any zeros remain in the solved board
                    solver = SudokuSolver(board=copy.deepcopy(grid_array))
                    solver.solve()
                    if not np.any(solver.board == 0):
                        self.game_state.get_puzzle_from_array(grid_array)
                        self.game_state.initialise_board()
                    else:
                        # TODO - this needs to be sorted out properly
                        #raise RuntimeError("Board could not be solved. Try another image")
                        print("Board could not be solved. Try another image")
                     
                except RuntimeError as err:
                    print(err)
                    print("Try a different image")
                    #raise
    
        # Momentary button to check correctness user's entries    
        if self.check_button.check_click():
            self.game_state.check_requested = True
            self.game_state.check_user_answers()
        else:
            self.game_state.check_requested = False
    
    
    def update(self):
        # Transmit all commands (move and target) to the game state
        #self.game_state.update(self.move_tank_command, self.target_command)
        self.game_state.update(self.row, self.col, self.move_cell_command, self.number)
        # Reset self.number (should this be in here?)
        self.number = None
        self.row = self.game_state.sel_row
        self.col = self.game_state.sel_col
        self.move_cell_command = Vector2(0, 0)
    
    
    def render(self):
        self.window.fill(self.WINDOW_BG_COLOUR)
                
        for iy, row_of_cells in enumerate(self.game_state.board):
            for ix, cell in enumerate(row_of_cells):
                
                # Determine cell colour based on selected and hint states
                if cell.selected:
                    color = self.SELECTED_CELL_COLOUR
                else:
                    color = self.UNSELECTED_CELL_COLOUR
                
                    
                # Try colouring the row, column and 3x3 square
                if self.row is not None:
                    if self.row == iy and not cell.selected:
                        color = self.ROW_COL_CELL_COLOUR
                        
                if self.col is not None:
                    if self.col == ix and not cell.selected:
                        color = self.ROW_COL_CELL_COLOUR
                        
                                
                # Handle the user asking to check the answer
                if self.game_state.check_requested:
                    if [iy, ix] in self.game_state.right_coords:
                        color = self.CORRECT_COLOUR
                    elif [iy, ix] in self.game_state.wrong_coords:
                        color = self.INCORRECT_COLOUR
                
                # Colour the hint cell, if shown
                if self.game_state.hint_requested:
                    if iy == self.game_state.hint_row and ix == self.game_state.hint_col:
                        if cell.selected:
                            color = self.SELECTED_CELL_COLOUR
                        else:
                            color = self.HINT_CELL_COLOUR                
                
                
                pygame.draw.rect(self.window, color, (ix*self.cell_width+1, iy*self.cell_height+1, self.cell_width-2, self.cell_height-2))
                
                # Display the Cell's num
                text_colour = "black" if cell.is_fixed else "blue"
                if cell.num != 0:
                    num_text = font.render(str(cell.num), True, text_colour)
                    cell_centre_x = (ix * self.cell_width) + int(0.5 * self.cell_width)
                    cell_centre_y = (iy * self.cell_height) + int(0.5 * self.cell_height)
                    num_text_rect = num_text.get_rect(center=(cell_centre_x, cell_centre_y))
                    #self.window.blit(num_text, (ix*self.cell_width+1, iy*self.cell_height+1))
                    self.window.blit(num_text, num_text_rect)
        
        
        # Add some lines to the board
        line_positions = [self.cell_width * i for i in range(10)]
        for i, x_pos in enumerate(line_positions):
            # Use heavy lines to mark 3x3 boxes and light lines otherwise
            line_width = 5 if i % 3 == 0 else 1
            pygame.draw.line(surface=self.window, color=(0, 0, 0), start_pos=(x_pos, 0), end_pos=(x_pos, self.game_state.world_height*self.cell_height), width=line_width)
            pygame.draw.line(surface=self.window, color=(0, 0, 0), start_pos=(0, x_pos), end_pos=(self.game_state.world_width*self.cell_width, x_pos), width=line_width)

        
        # If the puzzle is solved, display a message
        if self.game_state.solved:
            victory_text = font.render("Solved!", True, "dark green")
            victory_text_rect = victory_text.get_rect(center=(725, 40))
            self.window.blit(victory_text, victory_text_rect)

            # Disable some buttons
            self.hint_button.enabled = False
            self.check_button.enabled = False
            self.hint_button.text = "Show Hint"
        else:
            self.hint_button.enabled = True
            self.check_button.enabled = True


        # Render the buttons
        self.easy_button.draw()
        self.medium_button.draw()
        self.hard_button.draw()
        self.new_game_button.draw()
        self.reset_button.draw()
        self.hint_button.draw()
        self.solution_button.draw()
        self.check_button.draw()
        self.quit_button.draw()

        # Redraw the screen
        pygame.display.flip()        
        
    def run(self):
        while self.running:
            self.process_input()
            self.update()
            self.render()
            self.clock.tick(60)


pygame.font.init()
font = pygame.font.Font("freesansbold.ttf", 36)

user_interface = UserInterface()
user_interface.run()

pygame.quit()