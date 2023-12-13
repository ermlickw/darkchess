import numpy as np
import chess
from IPython.display import display, clear_output
import re


class DarkBoardViewer:
    """ used for displaying games only 
        was made before this info was encoded in state tensor
        still make it easy to see based on a particular vantage point instead 
        of only from the active players perspective
    """

    def __init__(self) -> None:
        #: Ring buffer of recent board encodings; stored boards are always
        #: oriented towards the White player. 
        self.current_board = np.zeros((8, 8, 3), dtype=int)
        self.current_attack_sq_white = []
        self.current_attack_sq_black = []


    def encode(self, board: chess.Board) -> np.array:
        """Converts a board to numpy array representation.
            Simple encoding:
            one 8x8 plane with values: 
            -6 to 6 for known pieces, 0 being empty square, then the number -9 for unseen squares
            for the active player
            
            then the regular 2 planes for repetitions
        """
        array = np.zeros((8, 8, 3), dtype=int)
        attacked_squares_white = []
        attacked_squares_black = []
        for square, piece in board.piece_map().items():
            rank, file = chess.square_rank(square), chess.square_file(square)
            piece_type, color = piece.piece_type, piece.color
            color_int = 1 if color else -1 # white is 1
            if color == chess.WHITE: # get the squares that white player's pieces attack
                x = [(chess.square_rank(sq), chess.square_file(sq)) for sq in board.attacks(square)]
                attacked_squares_white.extend(x)
            else:
                y = [(chess.square_rank(sq), chess.square_file(sq)) for sq in board.attacks(square)]
                attacked_squares_black.extend(y)
            array[rank, file, 0] = piece_type * color_int # place the piece 
        array[:, :, 1] = board.is_repetition(2)
        array[:, :, 2] = board.is_repetition(3)
        #update current state
        self.current_board = array
        self.current_attack_sq_white = list(set(attacked_squares_white))
        self.current_attack_sq_black = list(set(attacked_squares_black))
        return array, attacked_squares_white, attacked_squares_black
    
    def view(self, orientation: bool = chess.WHITE, board = chess.Board, hide_pieces:bool = False) -> np.array:
        """Returns an array view of the board history.

        This method returns a (8, 8, k * 14) array view of the k most recently
        added positions. If less than k positions have been added since the 
        last reset (or since the class was instantiated), missing positions are
        zeroed out. 
        
        By default, positions are oriented towards the white player; setting the
        optional orientation parameter to 'chess.BLACK' will reorient the view 
        towards the black player.
        
        Args:
            orientation: The player from which perspective the positions should
            be encoded.
        """

        # Copy buffer to not let reorientation affect the internal buffer
        array = self.current_board.copy()
        # now remove all pieces of the opponent that we can't capture due to dark chess rules
        hidden_array = np.ones_like(array.copy())*-9
        
        if orientation == chess.WHITE:
            my_squares = np.where(array[:,:,0]>0)
            my_squares = list(zip(my_squares[0],my_squares[1]))
            preserve_vision_indicies = list(set(self.current_attack_sq_white + my_squares))
        else:
            my_squares = np.where(array[:,:,0]<0)
            my_squares = list(zip(my_squares[0],my_squares[1]))
            preserve_vision_indicies = list(set(self.current_attack_sq_black + my_squares))
        
        for x,y in preserve_vision_indicies:
            hidden_array[x,y,0] = array[x,y,0]
        invis_indicies = np.where(hidden_array[:,:,0]==-9)
        invis_squares = list(zip(invis_indicies[0],invis_indicies[1])) 
        invis_squares = [chess.square(y,x) for (x,y) in invis_squares] # board renderer wants opposite order
        
        #unhide pawn moves
        pawn_moves = []
        for move in list(board.pseudo_legal_moves):
            if board.piece_at(move.from_square).piece_type == 1 and board.piece_at(move.from_square).color == orientation:
                pawn_moves.append(move.to_square)
        invis_squares = [s for s in invis_squares if s not in pawn_moves]
        
        if hide_pieces:
            for s in invis_squares:
                board.remove_piece_at(s)
        return hidden_array, invis_squares, board

    def render(self, game:chess.Board, vantage:int=1, clear=True,hide_pieces=False, piece_viz:chess.KING|chess.PAWN|chess.BISHOP|chess.QUEEN|chess.KNIGHT|chess.ROOK=None, \
                    agent_to_viz=None):
        sq_dict = None
        if piece_viz is not None and agent_to_viz is not None:
            if hasattr(agent_to_viz,'current_belief_tensor'):
                if agent_to_viz.current_belief_tensor is not None:
                    sq_dict = {}
                    idx = piece_viz
                    for x in range(8):
                        for y in range(8):
                            value = int(round(agent_to_viz.current_belief_tensor[x,y,idx+5],2) * 100)
                            if value > .5:
                                value *= 5
                            if value > 90:
                                value = 90
                            if len(str(value))== 1: value = f"0{value}"
                            sq_dict[chess.square(y,x)] = f"#007ED5{str(value)}"
        # add option to hide pieces
        p_board = chess.Board(str(game))
        p_board.turn = vantage
        self.encode(p_board)
        np_state, invis_squares, clean_board = self.view(vantage, p_board, hide_pieces=hide_pieces)
        if clear: clear_output(wait=True)
        
        if sq_dict is not None:
            display(chess.svg.board(
                p_board,
                squares=chess.SquareSet( invis_squares),
                fill=sq_dict,
                size=350,) )
        else:
            display(chess.svg.board(
                p_board,
                squares=chess.SquareSet( invis_squares),
                size=350,) )
        return
    
        
    
    def reset(self) -> None:
        """Clears the history."""
        self.__init__()