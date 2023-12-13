import random
import pyspiel
import numpy as np
from agents.utils import convert_obs_to_blank_board, convert_fen_to_board, PIECE_VALUES, PAWN, \
                        encode_board_to_tensor, decode_tensor_to_board, \
                            get_rewards, StockFishEngine
import chess

class TreeNode():
    """a particular hypothetical belief state node in a tree constructed during search
       represents both colors moves and holds children parent relationships as
       well as possible guess for the current belief space
    """
    def __init__(self, *, parent=None, move=None, 
                 root=False,b_tensor:np.ndarray=None, color=None):
        self.root = root
        self.color_turn = color
        self.b_tensor = b_tensor
        if parent is None:
            self.parent = None
            self.move = None
        else:
            self.parent = parent
            self.move = move
        self.value = 0
        self.n = 1e-2
        self.children = set([])
        self.current_piece_count = {}
    
    def sample_state_from_belief(self)->chess.Board:
        """if white then sample positions for all the remaining black piece type/count
           then write that to a game board
           
           if black could do the same or could just use the state of the parent with the applied move,
           we will assume the opponent is omnipotent and picks moves knowing 
           where our pieces are 
           

        Returns:
            chess.Board: possible chess board
        """
        
        return NotImplementedError()
    
    def update_belief_with_possible_state(self, observed_state:chess.Board)->None:
        """takes in a possible continuation state and updates the belief state tensor based on 
           that possible satte

        Args:
            state (chess.Board): _description_

        Returns:
            None: _description_
        """
        observered_tensor = encode_board_to_tensor(observed_state)
        for layer_idx in range(6,12):
            array = self.b_tensor[:,:,layer_idx].copy()
            
            self.b_tensor
        
        return NotImplementedError()
    

class PomcpAgent():
    """an agent that finds moves using montecarlo tree search based on belief space
    """
    def __init__(self,color, gamma=0.95,C=1, eps_threshold = 0.005, num_iterations=1000,
                 num_particles = 500) -> None:
        self.ordered_past_observations = []
        self.color = color
        self.action_space = 4672 # number of actions in dark chess
        #pomcp stuff
        self.C = C
        self.gamma = gamma
        self.eps_threshold = eps_threshold
        self.num_iterations = num_iterations
        self.search_tree_root = None
        self.current_real_state = None
        self.current_b_tensor = None
        self.stockfish = StockFishEngine()
        
    def find_move(self, state:pyspiel.State):
        self.current_real_state = state
        if self.search_tree_root is None:
            if self.color == chess.White:
                board = convert_fen_to_board(str(state))
                belief_tensor = encode_board_to_tensor(board=board) # should be belief tensor???
            else: # if black cheat for now -- in reality we need to search and propagate the 
                  # belief state after the first move
                board = convert_fen_to_board(str(state))
                belief_tensor = encode_board_to_tensor(board=board)
                
        self.search_tree_root = TreeNode(parent=None, move=None,
                                        root=True,b_tensor=belief_tensor,color=self.color)
        action = self.search(self.search_tree_root)
        return action
    
    def get_observation(self,state:pyspiel.State):
        # obs = state.observation_tensor(self.color)
        obs = state.observation_string(self.color)
        obs_state = convert_obs_to_blank_board(obs)
        obs_tensor = encode_board_to_tensor(obs_state)
        self.ordered_past_observations.append(obs_tensor)
        return obs_tensor
        
    
    def search(self, belief_node:TreeNode):
        for _ in range(self.num_iterations):
            possible_real_current_state_to_explore = belief_node.sample_state_from_belief()
            self.simulate(proposed_state=possible_real_current_state_to_explore,current_belief_node=belief_node,depth=0)
        return self.best_uct(belief_node=belief_node).move #best action at current belief node
    
    def simulate(self,proposed_state:chess.Board,current_belief_node:TreeNode,depth:int):
        if self.gamma**depth < self.eps_threshold:
            return 0 # too far away to matter
        if len(current_belief_node.children)==0: # leaf node of explored tree
            for action in range(self.action_space): # this is bad but we wont explore them all since we are sampling -- despot focuses this expansion! 
                if current_belief_node.parent is not None:
                    child_belief = current_belief_node.parent.b_tensor
                    #last color move + any observations it would have had after it moved and opponent moved
                else:
                    child_belief = current_belief_node.b_tensor
                    #just assume they know where the other went for the first move 
                current_belief_node.children.add(TreeNode(parent=current_belief_node,move=action,root=False,b_tensor=child_belief,
                                                          color=0 if current_belief_node.color_turn else 1))
            rollout_value = self.stockfish.evaluate_fen(proposed_state.fen()) # replacing rollout for now
            current_belief_node.n +=1
            current_belief_node.value = rollout_value
            return rollout_value
        cumulative_reward = 0
        best_continuation = self.best_uct(current_belief_node, proposed_state.copy()) # this should be best uct given hypothetical given state, so they are valid simulation
        next_state = proposed_state.copy().push(best_continuation.move) # this has to be legal now that we down sampled possible continuations
        reward = get_rewards(next_state)
        cumulative_reward += reward + self.gamma*self.simulate(next_state,best_continuation,depth=depth+1)
        current_belief_node.update_belief_with_possible_state(proposed_state) # reinforce that the sampled real state from belief state is likely 
        current_belief_node.n += 1
        best_continuation.n += 1 
        best_continuation.value += (cumulative_reward - best_continuation.value)/best_continuation.n
        return cumulative_reward
        

    def best_uct(self, belief_node:TreeNode, proposed_state:chess.Board)-> TreeNode:
        """Pick the best action according to the UCB/UCT algorithm"""
        children = list(belief_node.children)
        children = [c for c in children if c.move in proposed_state.legal_moves] #only consider belief children that are relevant to current proposed board state so legal sims
        num_pulls_per_child = np.array([child.n for child in children])
        reward_per_child= np.array([child.value for child in children])
        q = reward_per_child + np.sqrt((self.C*np.log(np.sum(num_pulls_per_child))) / num_pulls_per_child)
        best_ucb_child = np.argmax(q)
        return children[best_ucb_child]
            
        
        
        
          
    
