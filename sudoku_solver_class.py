import math


class SudokuSolver():
    def __init__(self, board):
        self.board = board
    
    
    def print_board(self):
        num_rows = len(self.board)
        num_cols = len(self.board[0])
        square_size = int(math.sqrt(num_rows))
        
        # Print one row at a time
        for row in range(num_rows):
            # Insert characters to define grid lines
            if row != 0 and row % square_size == 0:
                print("- " * (num_cols + square_size - 1))
            
            # Print numbers at each column
            for col in range(num_cols):
                # Insert characters to define grid lines
                if col != 0 and col % square_size == 0:
                    print("| ", end="")
                
                # Print numbers - only include newline at last column
                number = self.board[row][col]
                if col < num_cols-1:
                    print(f"{number} ", end="")
                else:
                    print(number)
                

    def find_next_empty(self, empty_val: int=0):
        # Iterate through the board one row at a time, from left to right,
        # checking for zeros. Return the index of the first zero found, otherwise
        # return None
        num_rows = len(self.board)
        num_cols = len(self.board[0])
        
        for row in range(num_rows):
            for col in range(num_cols):
                if self.board[row][col] == empty_val:
                    return (row, col)
                
        return None


    def is_valid_number(self, number: int, position: tuple):
        '''
        Check whether or not it is valid to insert a particular number at a
        specified position on the board.
        
        Args:
            number (int) - the number we are testing for validity
            position (tuple) - the position on the board where number is being inserted
        Returns:
            valid (bool) - whether or not the number is valid in the stated position
        '''
        
        # Assign the board dimensions to variables. Assume a square board.
        num_rows = len(self.board)
        square_size = int(math.sqrt(num_rows))
        
        # Unpack row and column from position tuple for readability
        row_idx, col_idx = position
        
        # Check if number is already present in current row
        if number in self.board[row_idx]:
            return False
        
        # Check if number is already present in current column
        current_column_values = [self.board[row][col_idx] for row in range(num_rows)]
        if number in current_column_values:
            return False
        
        # Check if number is already present in current square
        # Get indices of the square that our position lies in
        square_x_idx = col_idx // square_size
        square_y_idx = row_idx // square_size
        
        for row in range(square_y_idx * square_size, (square_y_idx * square_size) + square_size):
            for col in range(square_x_idx * square_size, (square_x_idx * square_size) + square_size):
                if self.board[row][col] == number and (row, col) != position:
                    return False
        
        # If we reach this point, the number is valid
        return True
        
        
    def solve(self):
        # Base case - no more empty squares, so we are finished
        next_empty_pos = self.find_next_empty()
        
        if not next_empty_pos:
            return True
        else:
            row, col = next_empty_pos
            
        # Try every number at the current empty position
        for i in range(1, 10):
            # Check if number is valid in this position
            if self.is_valid_number(number=i, position=(row, col)):
                # Put the number in the board
                self.board[row][col] = i
                # Now continue solving with the updated board (calling solve again)
                if self.solve():
                    # This means there are no more empty positions
                    return True
                else:
                    # the call to solve with the value i placed on the board
                    # returned False, so we now continue execution of the previous
                    # call to solve. 
                    self.board[row][col] = 0
        
        return False