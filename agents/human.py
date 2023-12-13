import random
import pyspiel
import numpy as np
import chess
from agents.utils import convert_obs_to_blank_board, PIECE_VALUES, PAWN, convert_fen_to_board, convert_chess_board_action_pyspeil_action

class HumanAgent():
    """a human makes the moves
    """
    def __init__(self, color) -> None:
        self.color = color
    
    def get_observation_after_move(self, state:pyspiel.Game):
        pass
    
    def find_move(self, state:pyspiel.Game):
        # obs = state.observation_tensor(self.color)
        board = convert_fen_to_board(str(state))
        legal_moves = [board.san(move).replace('+','').replace('#','') for move in board.legal_moves]
        
        selected_move = None
        while selected_move == None:
            print(legal_moves)
            selected_move = input('input your move from the list above')
            if selected_move not in legal_moves:
                selected_move = None
        return state.string_to_action(selected_move)