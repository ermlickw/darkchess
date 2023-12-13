import random
import pyspiel
import numpy as np
import chess
from agents.utils import convert_obs_to_blank_board, PIECE_VALUES, PAWN, KING, convert_chess_board_action_pyspeil_action

class GreedyAgent():
    """an agent that does the following
        (1) if you can take a piece of higher or equal value, take the highest one
        (2) if there are no pieces to capture, make a random move
    """
    def __init__(self,color,conservative:bool=False,preservative:bool=False) -> None:
        self.ordered_past_observations = []
        self.color = color
        self.conservative = conservative
        self.preservative = preservative
    
    def get_observation_after_move(self, state:pyspiel.Game):
        pass
    
    def find_move(self, state:pyspiel.Game):
        best_capture, clear_board, _ = self.get_greedy_move(state)
        if best_capture is not None:
            action = convert_chess_board_action_pyspeil_action(state,clear_board,best_capture)
        else: #(2) random move
            action = random.choice(state.legal_actions())
        return action
    
    def get_greedy_move(self, state:pyspiel.Game):
        # (1) find all captures
        # obs = state.observation_tensor(self.color)
        obs = state.observation_string(self.color)
        self.ordered_past_observations.append(obs)
        clear_board = convert_obs_to_blank_board(obs)
        return self.greedy_move_for_board(clear_board)
        
    def greedy_move_for_board(self, clear_board:chess.Board)->[str,chess.Board,bool]:
        
        captures = list(clear_board.generate_pseudo_legal_captures())
        
        best_capture = (None,0) # move, differential
        for capture in captures: 
            our_piece = clear_board.piece_type_at(capture.from_square)
            their_piece = clear_board.piece_type_at(capture.to_square)
            # if their piece is king the game is over
            if their_piece == KING: return capture,clear_board, True
            is_our_piece_attacked_too = clear_board.is_attacked_by(chess.BLACK if self.color else chess.WHITE,capture.from_square)
            # if taking en passant we are taking a pawn
            if their_piece is None: their_piece = PAWN
            # if our_piece == KING: our_piece = PAWN # agressive defending king
            differential = PIECE_VALUES[their_piece]
            if self.conservative: differential -= PIECE_VALUES[our_piece] #only take it if it is equal or higer
            if self.preservative: differential += PIECE_VALUES[our_piece]*is_our_piece_attacked_too # encourage taking if our piece is attacked
            
            if differential >= best_capture[1]: # if we gain or equal from material exchange
                best_capture = (capture,differential)
        return best_capture[0], clear_board, False
        
        

                
    def get_observation(self,state:pyspiel.State):
        # obs = state.observation_tensor(self.color)
        obs = state.observation_string(self.color)
        self.ordered_past_observations.append(obs)