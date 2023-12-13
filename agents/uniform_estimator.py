import random
import pyspiel
import numpy as np
import chess
import copy
from agents.greedy import GreedyAgent
from agents.utils import convert_obs_to_blank_board, PIECE_VALUES, PAWN, convert_fen_to_board, encode_board_to_tensor, \
                convert_board_tensor_to_uniform_belief_tensor, sample_possible_board_state, StockFishEngine, \
                    convert_chess_board_action_pyspeil_action

class UniformEstimatorAgent():
    """an agent that does the following:
        (1) find greedy move
        (2) samplea possible board from a uniform belief space given current observation
        (3) find a good stock fish move based on the sampled board
        (4) if using random_stock_fish_move -> do the greedy move if possible, else take stockfish move, else take random move
        (3) if not using random_stock_fish_move 
            -> find best move in multiple sampled board, store as candidate moves
                -> for each candidate move run stockfish eval across all possible sampled boards and take mean value to get avg case score
                    -> play candidate move that has the best score for you across all sampled boards
    """
    def __init__(self,color:bool,num_sampled_boards:int=3, random_stockfish_move:bool=False, num_candidate_moves_per_board:int=1, most_common_move:bool=False) -> None:
        self.ordered_past_observations = []
        self.color = color
        self.num_sampled_boards = num_sampled_boards
        self.num_candidate_moves_per_board =num_candidate_moves_per_board
        self.stockfish = StockFishEngine()
        self.random_stockfish_move = random_stockfish_move
        self.most_common_move = most_common_move
        if self.random_stockfish_move: self.num_sampled_boards = 1 # only need one if picking random anyway
        self.greedy_agent = GreedyAgent(color=self.color,preservative=True)
        self.current_belief_tensor = None
        
        
    def get_observation_after_move(self, state:pyspiel.Game):
        board = convert_fen_to_board(str(state))
        board_tensor = encode_board_to_tensor(board,viewpoint=self.color)
        uniform_belief_tensor = convert_board_tensor_to_uniform_belief_tensor(board_tensor=board_tensor)
        self.current_belief_tensor = uniform_belief_tensor
        
    def find_move(self, state:pyspiel.Game):
        #1 sample a set of possible opponent piece positions
        # first need to get a belief state tensor of possible piece positions
        board = convert_fen_to_board(str(state))
        board_tensor = encode_board_to_tensor(board)
        uniform_belief_tensor = convert_board_tensor_to_uniform_belief_tensor(board_tensor=board_tensor)
        self.current_belief_tensor = uniform_belief_tensor
        possible_boards = []
        for i in range(self.num_sampled_boards):
            # print('board',i)
            possible_boards.append(sample_possible_board_state(uniform_belief_tensor))
            
            
        # get the best moves in each position as candidate moves
        # only evaluate those moves, also include greedy moves in eval
        greedy_move,_, capture_king= self.greedy_agent.get_greedy_move(state=state)
        if capture_king: return convert_chess_board_action_pyspeil_action(state,board,greedy_move) # capture king if we can
        if greedy_move is None: 
            candidate_moves = []
        else:
            candidate_moves = [greedy_move]
            
        # get good candidate moves for the possible boards
        for possible_board in possible_boards:
            sf_moves = self.stockfish.get_candidate_moves(possible_board, self.num_candidate_moves_per_board)
            if sf_moves is not None: candidate_moves.extend(sf_moves)
        
        # for each candidate move that is legal in the current position
        # either randomly pick one or have stock fish evaluate the move over
        # all possible board states and pick the best one
        moves = {}
        if len([move for move in candidate_moves if move in board.pseudo_legal_moves]) ==0: 
            return random.choice(state.legal_actions())
        
        #greedy if greedy else random stockfish move
        if self.random_stockfish_move:
            move = [move for move in candidate_moves if move in board.pseudo_legal_moves][0]
            return convert_chess_board_action_pyspeil_action(state,board,move)
        
        # or have stock fish evaluate them over all the possible board state and take the best/safest
        if self.most_common_move:
            def most_common(lst):
                return max(set(lst), key=lst.count)
            best_move = most_common([move for move in candidate_moves if move in board.pseudo_legal_moves])
            return convert_chess_board_action_pyspeil_action(state,board,best_move)
        
        
        for legal_move in candidate_moves:
            if legal_move not in board.pseudo_legal_moves:
                continue
            # print('checking move')
            rewards = []
            for possible_board in possible_boards:
                # print('checking board')
                copy_board = copy.deepcopy(possible_board)
                copy_board.push(legal_move)
                rewards.append(self.stockfish.evaluate_fen(copy_board.fen()))
            # if self.color == chess.WHITE:
            #     rwd = min(rewards) # the worst it could be for white
            # else:
            #     rwd = max(rewards) # the worst it could be for black
            rwd = np.mean(rewards)
            moves[legal_move] = rwd
        if len(moves) == 0: 
            print('random move by estimator agent')
            return random.choice(state.legal_actions())
        if self.color == chess.WHITE: # pick best worst case move
            best_move = max(moves, key=moves.get)
        else:
            best_move = min(moves, key=moves.get)

        return convert_chess_board_action_pyspeil_action(state,board,best_move)

    def get_observation(self,state:pyspiel.State):
        # obs = state.observation_tensor(self.color)
        obs = state.observation_string(self.color)
        self.ordered_past_observations.append(obs)