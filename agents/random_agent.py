import random
import pyspiel
import numpy as np
import chess
from agents.utils import convert_obs_to_blank_board, PIECE_VALUES, PAWN

class RandomAgent():
    """an agent that plays a random move
    """
    def __init__(self,color) -> None:
        self.ordered_past_observations = []
        self.color = color

    def get_observation_after_move(self, state:pyspiel.Game):
        pass
    
    def find_move(self, state:pyspiel.Game):
        action = random.choice(state.legal_actions())
        return action
    