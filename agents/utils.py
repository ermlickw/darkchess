import chess
import re
import numpy as np
import stockfish
import pyspiel
import copy 
import random

NUM_DARK_CHESS_ACTIONS = 4672
[PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)

PIECE_VALUES = {PAWN:1,
          KNIGHT:3,
          BISHOP:3,
          ROOK:5,
          QUEEN:9,
          KING:100
}

RATIO_OF_EVAL_TO_WIN_VALUE = 0.5  # eval of win is only half as good as final reward

# https://github.com/official-stockfish/Stockfish/wiki/UCI-&-Commands
STOCKFISH_PARAMS = {
    "Debug Log File": "",
    "Contempt": 0,
    "Min Split Depth": 0,
    "Threads": 12, # More threads will make the engine stronger, but should be kept at less than the number of logical processors on your computer.
    "Ponder": "false",
    "Hash": 2048, # Default size is 16 MB. It's recommended that you increase this value, but keep it as some power of 2. E.g., if you're fine using 2 GB of RAM, set Hash to 2048 (11th power of 2).
    "MultiPV": 1,
    "Skill Level": 20,
    "Move Overhead": 10,
    "Minimum Thinking Time": 10,
    "Slow Mover": 10,
    "UCI_Chess960": "false",
    "UCI_LimitStrength": "false",
    "UCI_Elo": 3000
}

def convert_chess_board_action_pyspeil_action(pystate:pyspiel.State, board:chess.Board,move:str)->int:
    """converts a move in chess.Board to a move in pyspeil since pyspeil does not have as many helper functions 
       dark chess does not have checks or checkmate so we need to remove that 

    Args:
        board (chess.Board): _description_
        move (str): _description_

    Returns:
        int: _description_
    """
    string_of_all_actions = {pystate.action_to_string(a):a for a in pystate.legal_actions()}
    # print(string_of_all_actions)
    try:
        mv = board.san(move).replace('+','').replace('#','')
        return pystate.string_to_action(mv)
    except Exception as e: # occasionally the move is not valid in chess so we have to try some hacks
        print(e)
        try:
            return string_of_all_actions[board.san(move)]
        except Exception as e:
            print(e)
            return random.choice(pystate.legal_actions()) # if all fails just take a random move
        
    # else take the move they suggest and convert it to pyspeil integer action
    


def convert_obs_to_blank_board(obs_string:str):
    
    def replace_numbers_with_periods(s):
        def replace(match):
            return '.' * int(match.group(0))

        pattern = r'\d'
        result = re.sub(pattern, replace, s)

        return result

    def replace_and_return_count(s):
        #convert all numbers to period and then all periods to question marks 
        s = s.split()
        s[0] = replace_numbers_with_periods(s[0])
        s[0] = s[0].replace('.','?')
        
        def replace(match):
            return str(len(match.group(0)))
        pattern = r'(\?)\1+'
        s[0] = re.sub(pattern, replace, s[0]).replace('?','1')
        s = " ".join(s)
        return s       
    
    cleared_board = replace_and_return_count(obs_string)
    return chess.Board(cleared_board)

def convert_fen_to_board(fen:str):
    return chess.Board(fen)
      
#encode boards to tensors and decode tensors to boards       
def encode_board_to_tensor(board:chess.Board, viewpoint=None)->np.ndarray:
    """tensor of the current board state
        0-5 active player piece positions for [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]
        6-11 opponent piece positions or belief for [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]
        12-13 2 and 3 repetitions
        14 - active player
        15 - full move number
        16-19 castling rights for active and opponent
        20 half move clock
        21 en passant
        22 visible pieces map for active player
        23 -- piece count for all pieces for both side (always known in dark chess)
    Args:
        board (chess.Board): chess board

    Returns:
        np.ndarray[8,8,24]: encoded state
    """
    if viewpoint is None:
        viewpoint = board.turn
    board = copy.deepcopy(board)
    piece_count_dict = {}
    array = np.zeros((8, 8, 14), dtype=float)
    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        piece_type, color = piece.piece_type, piece.color
        # if piece_type != PAWN:
        if (color,piece_type) not in piece_count_dict:
            piece_count_dict[(color,piece_type)] = 1
        else:
            piece_count_dict[(color,piece_type)] += 1
        # else:
        #     if (color,piece_type) not in piece_count_dict:
        #         piece_count_dict[(color,piece_type)] = [file]
        #     else:
        #         piece_count_dict[(color,piece_type)].append(file)
        offset = 0 if color == chess.WHITE else 6
        
        # Chess enumerates piece types beginning with one, which we have
        # to account for
        idx = piece_type - 1

        array[rank, file, idx + offset] = 1

    # Repetition counters
    array[:, :, 12] = board.is_repetition(2)
    array[:, :, 13] = board.is_repetition(3)
    
    
    # if it is blacks turn we need to flip the orientation and order of black and white piece positions so the array evaluation positions
    # stay the same regardless who the active player is 
    if viewpoint == chess.BLACK: 
        # Rotate all planes encoding the position by 180 degrees
        rotated = np.rot90(array[:, :, :12].copy(), k=2)

        # In the buffer, the first six planes encode white's pieces; 
        # swap with the second six planes
        rotated = np.roll(rotated, axis=-1, shift=6)

        np.copyto(array[:, :, :12], rotated)
         

    meta = np.zeros(
        shape=(8 ,8, 10),
        dtype=int
    )

    # Active player color
    meta[:, :, 0] = int(board.turn)

    # Total move count
    meta[:, :, 1] = board.fullmove_number

    # Active player castling rights
    meta[:, :, 2] = board.has_kingside_castling_rights(viewpoint)
    meta[:, :, 3] = board.has_queenside_castling_rights(viewpoint)

    # Opponent player castling rights
    meta[:, :, 4] = board.has_kingside_castling_rights(not viewpoint)
    meta[:, :, 5] = board.has_queenside_castling_rights(not viewpoint)

    # No-progress counter
    meta[:, :, 6] = board.halfmove_clock

    #en passent target square
    if board.ep_square:
        meta[:, :, 7] = board.ep_square # integer
    else:
        meta[:, :, 7] = -1
        
    # encode one more layer of visible squares
    #attacked or your pieces are there
    squares = []
    for s in range(64):
        if board.is_attacked_by(viewpoint,s):
            squares.append(s)
        piece = board.piece_at(s)
        if piece:
            if piece.symbol().isupper():
                color = 1
            else:
                color = 0
            if viewpoint == color:
                squares.append(s)

    #pawn moves
    pawn_moves = []
    flipped_for_visibility = False
    if viewpoint != board.turn: 
        flipped_for_visibility = True
        board.turn = viewpoint
    for move in list(board.pseudo_legal_moves): 
        if board.piece_at(move.from_square).piece_type == 1:
            pawn_moves.append(move.to_square)
    visible_squares = list(set(pawn_moves + squares))
    visible_squares = [(chess.square_file(s), chess.square_rank(s)) for s in visible_squares]
    for x in visible_squares:
        meta[x[1], x[0], 8] = 1
    if flipped_for_visibility:
        if viewpoint == chess.WHITE: 
            board.turn = chess.BLACK
        else: 
            board.turn = chess.WHITE
    
        
    # if it is blacks turn we need to flip the orientation and order of black and white piece positions so the array evaluation positions
    # stay the same regardless who the active player is 
    if viewpoint == chess.BLACK: 
        # Rotate all planes encoding the position by 180 degrees
        rotated = np.rot90(meta[:, :, 8].copy(), k=2)
        np.copyto(meta[:, :, 8], rotated)
        
    #array of piece count for each side -- this will always be known
    #row 0-1 will be the active player, row 6-7 will be the opponent
    # print(piece_count_dict)
    for (c,pice), piece_no in piece_count_dict.items():
        if c == viewpoint:#active player inserts
            meta[0+1,pice,9] = piece_no
        else: # opponent inserts
            meta[6+1,pice,9] = piece_no

    return np.concatenate([array, meta], axis=-1)

def decode_tensor_to_board(state_tensor:np.ndarray, belief_tensor=True, viewpoint=None):
    
    #if its a belief state we ignore en passant due to complications
    
    # if it is black's turn we need to reverse the orientation and positions of the white and black planes to 
    # process them appropriately
    if viewpoint is None:
        viewpoint = state_tensor[0,0,14]
    
    state_tensor = copy.deepcopy(state_tensor)
    if viewpoint == chess.BLACK: 
        # Rotate all planes encoding the position by 180 degrees
        rotated = np.rot90(state_tensor[:, :, :12].copy(), k=2)

        # In the buffer, the first six planes encode white's pieces; 
        # swap with the second six planes
        rotated = np.roll(rotated, axis=-1, shift=6)

        np.copyto(state_tensor[:, :, :12], rotated)
        
    # decode state tensor to board position
    piece_definitions = {0:'P',
                        1:'N',
                        2:'B',
                        3:'R',
                        4:'Q',
                        5:'K',
                        6:'p',
                        7:'n',
                        8:'b',
                        9:'r',
                        10:'q',
                        11:'k',
                        }

    board = chess.Board()
    board.clear_board()
    piece_map = {}
    for layer_idx in range(12):
        for x in range(8):
            for y in range(8):
                if state_tensor[:,:,layer_idx].T[x,y] == 1:
                    piece_map[chess.square(x,y)] = chess.Piece.from_symbol(piece_definitions[layer_idx])
                    
    board.set_piece_map(piece_map)

    #castling rights
    castling_tuple = (state_tensor[0,0,16], state_tensor[0,0,17],state_tensor[0,0,18],state_tensor[0,0,19])
    if viewpoint == chess.WHITE: # get the castling permissions based on active player
        castling_string = 'K'*int(castling_tuple[0]) + 'Q'*int(castling_tuple[1]) +'k'*int(castling_tuple[2]) +'q'*int(castling_tuple[3])
    else:
        castling_string = 'K'*int(castling_tuple[2]) +'Q'*int(castling_tuple[3]) +'k'*int(castling_tuple[0]) + 'q'*int(castling_tuple[1])
    board.set_castling_fen(castling_string)

    # turn
    board.turn = int(state_tensor[0,0,14])

    #half move
    board.halfmove_clock = int(state_tensor[0,0,20])

    #full move
    board.fullmove_number = int(state_tensor[0,0,15])

    #enpassent 
    if state_tensor[0,0,21] == -1 or belief_tensor:
        board.ep_square = None
    else:
        board.ep_square = chess.square_name(int(state_tensor[0,0,21]))
    return board
        
    
#convert tensor to uniform belief tensor based on visible squares
def convert_board_tensor_to_uniform_belief_tensor(board_tensor:np.ndarray):
    num_zero = np.count_nonzero(board_tensor[:,:,22]==0)
    non_zero_value = 1 / num_zero
    board_tensor = copy.deepcopy(board_tensor)
    non_active_player_idx = 7 #always 
    for layer_idx in range(6,12):
        if board_tensor[non_active_player_idx,layer_idx-5,23]: # if THE OPPONENT has a piece of that type
            # all visible squares are visible 
            board_tensor[:,:,layer_idx] *=  board_tensor[:,:,22]
            #then we need to count to see if we found them all
            if board_tensor[:,:,layer_idx].sum() == board_tensor[non_active_player_idx,layer_idx-5,23]:
                continue
            elif board_tensor[:,:,layer_idx].sum() < board_tensor[non_active_player_idx,layer_idx-5,23]:
                # all non viz squares are random prob
                board_tensor[:,:,layer_idx][board_tensor[:,:,22]==0] = non_zero_value
            else:
                raise ValueError('Too many pieces discovered than we thought existed')
        else: # if they don't have that piece type then it's all zeros
            np.copyto(board_tensor[:,:,layer_idx], np.zeros_like(board_tensor[:,:,layer_idx]))
    # #pawns
    # for file in range(8):
    #     num_zero_file = np.count_nonzero(board_tensor[:,file,22]==0)
    #     non_zero_value_file = 1 / (8-num_zero_file)
        
    #     # copy to the row of enemy pawns, the true value if visible and has a pawn on that file
    #     np.copyto(board_tensor[:,file,6], board_tensor[:,file,6] * board_tensor[active_player_idx,file,22])
    #     if board_tensor[:,file,6].sum() != board_tensor[active_player_idx,file,22]: # else do the zero value if they have a piece of that type
    #         board_tensor[:,file,6][board_tensor[:,file,22]==0] = non_zero_value_file*board_tensor[active_player_idx,file,22]
            
    return board_tensor
#sample from a belief tensor
def sample_possible_board_state(belief_tensor:np.ndarray)->chess.Board:
    belief_tensor = sample_possible_board_state_tensor(belief_tensor=belief_tensor)
    return decode_tensor_to_board(belief_tensor,belief_tensor=True)

def sample_possible_board_state_tensor(belief_tensor:np.ndarray)->np.ndarray:
    # for number of pieces in 23 open row 7 of each type
    # go into plane and sample a location from the 2d belief state until number is met 
    # then set to 1, and set all others to zero
    # then return the chess board
    belief_tensor_copy = copy.deepcopy(belief_tensor)
    def sample_index(p, num_samples):
        positions = []
        #keep sampling unique positions
        sample_indicies = np.random.choice(np.arange(p.size), size=num_samples, p=p.ravel(),replace=False)
        for i in sample_indicies:
            positions.append(np.unravel_index(i, p.shape))
        return list(set(positions))
    
    for layer_idx in range(6,12):
        layer = copy.deepcopy(belief_tensor_copy[:,:,layer_idx]) 
        #count number of ones 
        num_pieces_of_type = layer[layer==1].sum()
        tracked_num_pieces = int(belief_tensor_copy[7,layer_idx-5,23])
        
        if int(num_pieces_of_type) > tracked_num_pieces:
            raise ValueError(f'belief tensor has more of piece {layer_idx-5} than has been traced by the tensor {(int(num_pieces_of_type) , tracked_num_pieces,layer_idx-5)}')

        elif num_pieces_of_type == tracked_num_pieces:
            continue # no need to sample we have all of that piece type 
        else:
            #sample 
            
            num_needed_pieces = tracked_num_pieces - num_pieces_of_type
            
            layer[layer==1] = 0 # don't pick places where the pieces already are
            layer[layer<0] = 1e-3 # or things that we have made zero
            if (layer==0).all(): # then just pick uniform
                layer = copy.deepcopy(belief_tensor_copy[:,:,layer_idx]) 
                layer[layer==0] = 1/layer[layer==0].size
                layer[layer==1] = 0
            
            layer /= layer.sum() #normalize

            add_positions = sample_index(layer,int(num_needed_pieces))
            for add_position in add_positions:
                belief_tensor_copy[add_position[0],add_position[1],layer_idx] = 1
        # zero the rest now
        other_indicies = (belief_tensor_copy[:,:,layer_idx]!=0) & (belief_tensor_copy[:,:,layer_idx]!=1)
        belief_tensor_copy[:,:,layer_idx][other_indicies] = int(0)
    return belief_tensor_copy 

#stock fish engine for evaluating positions, hyperparams above
class StockFishEngine():
    def __init__(self) -> None:
        self.engine = stockfish.Stockfish(depth=3,parameters=STOCKFISH_PARAMS, path='/usr/games/stockfish')
    
    
    def type_converter(self, fen:str|pyspiel.State|chess.Board):
        if type(fen) == pyspiel.State:
            fen = str(fen)
        if type(fen) == chess.Board:
            fen = fen.fen()
        return fen
    def get_candidate_moves(self, board:chess.Board, num_candidate_moves_per_board):
        fen = board.fen()
        try:
            valid = self.engine.is_fen_valid(fen)
        except Exception as e:
            print(e)
            return None
        if valid:
            try:
                self.engine.set_fen_position(fen)
                best_moves = self.engine.get_top_moves(num_candidate_moves_per_board)
            except Exception as e:
                print(e)
                return None
            return [chess.Move.from_uci(move['Move']) for move in best_moves]
        else:
            return None
        
    def evaluate_fen(self,fen:str | pyspiel.State):
        fen = self.type_converter(fen)
        try:
            valid = self.engine.is_fen_valid(fen)
        except Exception as e:
            print(e)
            valid = False
        if valid:
            try:
                self.engine.set_fen_position(fen)
                eval = self.engine.get_evaluation()
            except Exception as e:
                print(e)
                return 0
            value = eval["value"]
            return (np.clip(value,-150,150)/150 ) * RATIO_OF_EVAL_TO_WIN_VALUE
        else:
            # print('fen is impossible in regular chess')
            #this has to do with checks or king captures
            #check if one of the kings captured
            board = chess.Board(fen=fen)
            if chess.Piece.from_symbol('K') not in board.piece_map().values(): return -1 
            if chess.Piece.from_symbol('k') not in board.piece_map().values(): return 1
            #otherwise need to do a simple piece count evaluation
            white = 0
            black = 0
            for value in board.piece_map().values(): 
                if str(value).islower():
                    black += PIECE_VALUES[value.piece_type]
                else:
                    white += PIECE_VALUES[value.piece_type]
            return (white-black)/ sum(PIECE_VALUES.values()) * RATIO_OF_EVAL_TO_WIN_VALUE


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    if len(X) == 1: return 0 # likely due to not enough sampling
    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    
    return p