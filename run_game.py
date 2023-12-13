import pyspiel
import chess
from agents.playing_utilities import play_game,calculate_elo
from agents.human import HumanAgent
from agents.greedy import GreedyAgent
from agents.random_agent import RandomAgent
from agents.uniform_estimator import UniformEstimatorAgent
from agents.updating_estimator import ProgressiveEstimatorAgent
import time
import numpy as np
from typing import Optional
np.set_printoptions(precision=3)

#CONFIG 
num_games= 100
invalid_games = 0
view_point = chess.WHITE
color_of_belief_state_to_watch:Optional[chess.WHITE|chess.BLACK] = None
piece_of_belief_state_to_watch:Optional[chess.KING|chess.PAWN|chess.BISHOP|chess.QUEEN|chess.KNIGHT|chess.ROOK] = None
display_game:bool = False
hide_unseen_pieces:bool = False
wait_for_user_input:bool = False
white = ProgressiveEstimatorAgent(chess.WHITE, num_sampled_boards=5, num_candidate_moves_per_board=3, most_common_move=True) # ProgressiveEstimatorAgent(chess.WHITE, num_sampled_boards=5, num_candidate_moves_per_board=3, most_common_move=True, number_of_state_explorations=1000) 
black = UniformEstimatorAgent(chess.BLACK, num_sampled_boards=5, num_candidate_moves_per_board=3, most_common_move=True)# UniformEstimatorAgent(chess.BLACK, num_sampled_boards=5, num_candidate_moves_per_board=3, most_common_move=True) 
    
#play games
white_rewards = []
for i in range(num_games):
    try:
        white_reward, state = play_game(white=white,black=black,view_point=view_point, wait_for_user=wait_for_user_input, display=display_game, hide_pieces=hide_unseen_pieces,\
                                    piece_to_viz_belief=piece_of_belief_state_to_watch,color_to_viz_belief=color_of_belief_state_to_watch)
        white_rewards.append(int(white_reward))
    except Exception as e:
        white_reward = 0
        invalid_games +=1
        print(e)
    
    print(f'game {i+1}')
    print(f'invald games: {invalid_games}')
    print(f'White Won {round(100*(white_rewards.count(1)/num_games),2)}% of games {white_rewards.count(1)}/{num_games-invalid_games}')
    time.sleep(2)

num_wins_minus_num_losses = sum(white_rewards)
print(f'White Won {round(100*(white_rewards.count(1)/(num_games-invalid_games)),2)}% of {num_games-invalid_games} games')
print(f"white elo in comparison is: {calculate_elo(num_wins_minus_num_losses=(white_rewards.count(1)-white_rewards.count(-1)),num_games=(num_games-invalid_games),opponent_ranking=400)}")