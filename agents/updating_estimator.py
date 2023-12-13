import random
import pyspiel
import numpy as np
import chess
import copy
from agents.greedy import GreedyAgent
from agents.utils import convert_fen_to_board, encode_board_to_tensor, \
                 sample_possible_board_state,sample_possible_board_state_tensor, StockFishEngine, \
                    convert_chess_board_action_pyspeil_action, softmax, decode_tensor_to_board

GREEDY_PERCENTAGE = 0.1

class ProgressiveEstimatorAgent():
    """an agent that does the following:
    """
    
    def __init__(self,color,num_sampled_boards=3, random_stockfish_move=False, num_candidate_moves_per_board=1, most_common_move:bool=False, number_of_state_explorations:int = 400) -> None:
        self.ordered_past_observations = []
        self.color = color
        self.num_sampled_boards = num_sampled_boards
        self.num_candidate_moves_per_board =num_candidate_moves_per_board
        self.stockfish = StockFishEngine()
        self.random_stockfish_move = random_stockfish_move
        self.most_common_move = most_common_move
        if self.random_stockfish_move: self.num_sampled_boards = 1 # only need one if picking random anyway
        self.current_belief_tensor = None
        self.greedy_agent = GreedyAgent(color=chess.WHITE if self.color==chess.BLACK else chess.WHITE, preservative=True)
        self.number_of_explorations = number_of_state_explorations # how many moves to sample to update belief
    
    def get_observation_after_move(self, state:pyspiel.Game):
        board = convert_fen_to_board(str(state))
        board_tensor_from_your_perspective = encode_board_to_tensor(board,viewpoint=self.color)
        self.update_belief_state(board_tensor_from_your_perspective,observation=True)
        
    def find_move(self, state:pyspiel.Game):
        #1 sample a set of possible opponent piece positions
        # first need to get a belief state tensor of possible piece positions
        board = convert_fen_to_board(str(state))
        board_tensor = encode_board_to_tensor(board)
        if self.current_belief_tensor is None: # first move
            self.current_belief_tensor = encode_board_to_tensor(chess.Board(),viewpoint=self.color)
            if self.color == chess.BLACK:
                self.update_belief_state(board_tensor, observation=False)
        else:  
            #then we update based on current observed state -- observation indicates we are not up to date
            self.update_belief_state(board_tensor, observation=False)
        possible_boards = []
        for i in range(self.num_sampled_boards):
            # print('board',i)
            possible_boards.append(sample_possible_board_state(self.current_belief_tensor))
            
        ##same as uniform agent below ##
        
        
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

    def update_belief_state(self,board_tensor,observation=False):
        #layers 6-12 cannot be directly copied! those are our guesses in the belief state
        #they can only be observed through the squares visible in layer 22
        #update non opponent pieces
        new_belief_tensor = copy.deepcopy(self.current_belief_tensor)
        new_belief_tensor[:,:,:6] = board_tensor[:,:,:6].copy()
        new_belief_tensor[:,:,12:] = board_tensor[:,:,12:].copy()
        
        if not observation:
            # early exit update belief if we saw their last move
            we_saw_last_move = False
            for layer_idx in range(6,12):
                if (self.current_belief_tensor[:,:,layer_idx][self.current_belief_tensor[:,:,22]==1] != board_tensor[:,:,layer_idx][self.current_belief_tensor[:,:,22]==1]).any(): # one of observed squares changed during their move
                    we_saw_last_move=True
                    #only update the squares that changed, nothing else needs to change
                    new_belief_tensor[:,:,layer_idx][self.current_belief_tensor[:,:,22]==1] = board_tensor[:,:,layer_idx][self.current_belief_tensor[:,:,22]==1] # can see the piece that captured ours if we dont stil have vision
                    new_belief_tensor[:,:,layer_idx][board_tensor[:,:,22]==1] = board_tensor[:,:,layer_idx][board_tensor[:,:,22]==1] # make sure our vision matches the new board vision for that piece type
                    #for the other squares we are not sure where that piece came from necessarily -- just assume we don't know 
                    new_belief_tensor[:,:,layer_idx][self.current_belief_tensor[:,:,22]==0] -= 1e-5 # subtract from where we think they were to show uncertainty
                    unknown_indicies = (new_belief_tensor[:,:,layer_idx]!=0) & (new_belief_tensor[:,:,layer_idx]!=1)
                    if unknown_indicies.any():
                        new_belief_tensor[:,:,layer_idx][unknown_indicies] =  softmax(new_belief_tensor[:,:,layer_idx][unknown_indicies])
                      
            if we_saw_last_move:    
                self.current_belief_tensor = new_belief_tensor
                return 
            
            #simulate possible moves the opponent could have made to update belief
            new_belief_tensor[:,:,6:12] =  np.zeros_like(new_belief_tensor[:,:,6:12]) # new belief tensor starts from zero when moves happened 
            really_possible_boards = []
            potentially_possible_boards = [sample_possible_board_state_tensor(self.current_belief_tensor)  for _ in range(self.number_of_explorations)] # wrong side when sampling? black is making two moves
            #then make move for opponent!
            for potential_board_tensor in potentially_possible_boards:
                assert (potential_board_tensor[:,:,:6] == board_tensor[:,:,:6]).all()
                potential_board = decode_tensor_to_board(potential_board_tensor, viewpoint=self.color) #needs to be from this players perspective since samples are from this players perspective
                if random.random() <= GREEDY_PERCENTAGE:
                    move = self.greedy_agent.greedy_move_for_board(potential_board)[0]
                else:
                    move = random.choice(list(potential_board.pseudo_legal_moves))
                if move is None: move = random.choice(list(potential_board.pseudo_legal_moves))
                potential_board.push(move)
                potential_board_tensor = encode_board_to_tensor(potential_board, viewpoint=self.color) # take board with random move and back convert to tensor
                
                if self.check_board_is_possible(board_tensor,potential_board_tensor):
                    really_possible_boards.append(potential_board_tensor)
            
            no_possible_boards = False
            if len(really_possible_boards)==0: # none of our simulated boards match reality
                print('no possible boards')
                no_possible_boards = True
                agg_of_real_possible_current_boards = None
            else: 
                agg_of_real_possible_current_boards = np.stack(really_possible_boards,axis=-1)
               
        # update belief tensor
        #piece type by piece type
        non_active_player_idx = 7 #always 
        #incorporate new info
        for layer_idx in range (6,12):
            num_of_opponents_of_this_type = board_tensor[non_active_player_idx,layer_idx-5,23]
            if num_of_opponents_of_this_type: # if THE OPPONENT has a piece of that type
                 #visible squares
                most_recent_obs = board_tensor[:,:,layer_idx] * board_tensor[:,:,22]
                num_missing = num_of_opponents_of_this_type - most_recent_obs.sum()
                if num_missing == 0: # if we see them all right now
                    self.current_belief_tensor = board_tensor[:,:,layer_idx].copy()
                    continue
                if num_missing < 0:
                    # we saw them and now we lost them
                    assert observation == False
                if observation:
                    # we must update our current belief state to match
                    new_belief_tensor[:,:,layer_idx][board_tensor[:,:,22]==1] = board_tensor[:,:,layer_idx][board_tensor[:,:,22]==1]
                    #renormalize probs after updates for those squares that are not known
                    unknown_indicies = (new_belief_tensor[:,:,layer_idx]!=0) & (new_belief_tensor[:,:,layer_idx]!=1)
                    if unknown_indicies.any():
                        new_belief_tensor[:,:,layer_idx][unknown_indicies] =  softmax(new_belief_tensor[:,:,layer_idx][unknown_indicies])
                else:
                    # need to go back to last generate random boards from belief state knowledge and of those that are consistent with 
                    # current observation, use them to make probabilties
                
                    #then get probs for each opp square based on real example boards
                    if not no_possible_boards:
                        new_belief_tensor[:,:,layer_idx] = agg_of_real_possible_current_boards[:,:,layer_idx,:].mean(axis=-1) 
                    else:
                        new_belief_tensor[:,:,layer_idx] = self.current_belief_tensor[:,:,layer_idx] # use the old belief state instead then if we don't have a guess
                    #then add in the real observations that we have right now
                    new_belief_tensor[:,:,layer_idx][board_tensor[:,:,22]==1] = board_tensor[:,:,layer_idx][board_tensor[:,:,22]==1]
                    #normalize the now unknown squares 

                    unknown_indicies = (new_belief_tensor[:,:,layer_idx]!=0) & (new_belief_tensor[:,:,layer_idx]!=1)
                    if unknown_indicies.any():
                        new_belief_tensor[:,:,layer_idx][unknown_indicies] =  softmax(new_belief_tensor[:,:,layer_idx][unknown_indicies]+1e-3) # some smoothing to all squares
                    
        self.current_belief_tensor = new_belief_tensor.copy()
        
    
    
    def check_board_is_possible(self,real_current_board:np.ndarray, board:np.ndarray):
        #make sure the squares we can see are consistent with the sampled board
        new_observed_indicies = (real_current_board[:,:,22]==1)
        if (board[:,:,6:12][new_observed_indicies] == real_current_board[:,:,6:12][new_observed_indicies]).all() and \
            (real_current_board[:,:,:6] == board[:,:,:6]).all() :
            return True
        return False
        