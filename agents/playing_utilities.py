from darkchess_viz_wrapper import DarkBoardViewer
import pyspiel
import chess
import chess.svg
import time
def calculate_elo(num_wins_minus_num_losses,num_games,opponent_ranking=400):
    return (opponent_ranking+400*(num_wins_minus_num_losses))/num_games


def play_game(white,black, view_point=chess.WHITE, wait_for_user:bool=True, display=True,hide_pieces:bool=False,color_to_viz_belief:bool=None, \
              piece_to_viz_belief:chess.KING|chess.PAWN|chess.BISHOP|chess.QUEEN|chess.KNIGHT|chess.ROOK=None):
  #rendering setup
  agent_to_viz = None
  if color_to_viz_belief is not None:
    if color_to_viz_belief == chess.WHITE:
      agent_to_viz = white
    else:
      agent_to_viz = black
    
  darkboardstate = DarkBoardViewer()
  game = pyspiel.load_game("dark_chess")
  state = game.new_initial_state()
  
  while not state.is_terminal():
    
    #display board
    if display: darkboardstate.render(state,view_point, hide_pieces=hide_pieces, piece_viz=piece_to_viz_belief, agent_to_viz=agent_to_viz)
    
    
    #one player finds a move
    if state.current_player() == white.color:
      action = white.find_move(state)
    else:
      action = black.find_move(state)
    
    #apply move to board
    state.apply_action(action)
    
    #get observation after your move
    if state.current_player() == white.color:
      action = black.get_observation_after_move(state)
    else:
      action = white.get_observation_after_move(state)
    
    #advance on user input
    if wait_for_user: input()
    
  #render final state
  if display: darkboardstate.render(state,view_point, hide_pieces=hide_pieces, piece_viz=piece_to_viz_belief, agent_to_viz=agent_to_viz)
  

  #rewards 
  black_win,white_win = state.rewards()
  if display:
    if white_win>black_win: print('white wins')
    elif black_win>white_win: print ('black wins')
    else: print('tie game')
    print('game over')
  return state.rewards()[1], state