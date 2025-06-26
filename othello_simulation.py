import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gc
import time
import multiprocessing
import numpy as np
from tqdm.auto import tqdm

from heuristics import HEURISTICS
import agents
import board
import config
import tree
from util import line_2_plane


class OthelloSimulation:
    """
    A class to handle the simulation of Othello games between agents.
    """
    
    def __init__(self, agent1_class, agent2_class, agent1_args: dict={}, agent2_args: dict={}):
        """
        Initializes the OthelloSimulation with two agents.

        Args:
            agent1_class: The first agent's class (plays as black).
            agent2_class: The second agent's class (plays as white).
            agent1_args: Arguments for initializing agent1.
            agent2_args: Arguments for initializing agent2.
        """
        if not issubclass(agent1_class, agents.Agent):
            raise TypeError(f"agent1 class must inherit from Agent class, got {agent1_class}")
        if not issubclass(agent2_class, agents.Agent):
            raise TypeError(f"agent2 class must inherit from Agent class, got {agent2_class}")
            
        self.agent1_class = agent1_class
        self.agent2_class = agent2_class
        self.agent1_args = agent1_args
        self.agent2_args = agent2_args

    def run_simulation(self, num_simulations: int, parallel: bool = True):
        """
        Runs the simulation for the specified number of games.

        Args:
            num_simulations (int): The number of games to simulate.
            parallel (bool): Whether to run simulations in parallel.
        """
        nb_cores = max(1, multiprocessing.cpu_count() - 1)
        game_results = []
        
        print("=============== Othello Simulation ===============")
        start = time.perf_counter()

        sim_args = (self.agent1_class, self.agent2_class, self.agent1_args, self.agent2_args)
        
        if parallel and num_simulations > 1:
            print(f"Starting simulation on {nb_cores} cores.\n")
            
            with multiprocessing.Pool(processes=nb_cores) as pool:
                args_list = [(self.agent1_class, self.agent2_class, self.agent1_args, self.agent2_args)] * num_simulations
                game_results = list(tqdm(
                    pool.imap_unordered(simulate_single_game, args_list), 
                    total=num_simulations
                ))
        else:
            print("Running simulations sequentially.\n")
            for i in range(num_simulations):
                mid = time.perf_counter()
                result = simulate_single_game(sim_args)
                tot = time.perf_counter() - mid
                
                game_results.append(result)
                print(f"Simulation {i+1} took {tot:<6.2f} sec, result: {result}")
        
        end_tot = time.perf_counter() - start
        
        # Count results
        agent1_wins = game_results.count(1)
        agent2_wins = game_results.count(-1)
        draws = game_results.count(0)
        
        print("\n===================== Results ====================")
        print(f"Player 1 (Black) | Wins: {agent1_wins:<3}, Draws: {draws:<3}")
        print(f"Player 2 (White) | Wins: {agent2_wins:<3}, Draws: {draws:<3}")
        print(f"Total games: {num_simulations}")
        print(f"Player 1 win rate: {agent1_wins/num_simulations:.3f}")
        print(f"Player 2 win rate: {agent2_wins/num_simulations:.3f}")
        print(f"Simulation took {end_tot:<7.2f} sec (avg:{end_tot/num_simulations:.2f})")
        print(f"LaTex format: ({agent1_wins}/{agent2_wins}/{draws}) {int((agent1_wins/num_simulations) * 100)}\\%")
        
        return game_results


def simulate_single_game(sim_args: tuple):
    agent1_class, agent2_class, agent1_args, agent2_args = sim_args
    agent1 = agent1_class(**agent1_args)
    agent2 = agent2_class(**agent2_args)
    
    game_board = board.Board()
    current_agent = config.black
    consecutive_passes = 0

    fake_parent = tree.FakeNode()
    temp_node = tree.Node(fake_parent, 0, current_agent, game_board)
    temp_node.is_game_root = True
    temp_node.is_search_root = True
    
    while consecutive_passes < 2:       
        if current_agent == config.black:
            agent_move = agent1.make_move(temp_node)
        else:
            agent_move = agent2.make_move(temp_node)
        if agent_move == -1:
            agent_move = config.pass_move
        if agent_move == config.pass_move:
            consecutive_passes += 1
        else:
            game_board = game_board.make_move(current_agent, agent_move)
            consecutive_passes = 0
        
        # black_count = np.sum(game_board.black_array2d)
        # white_count = np.sum(game_board.white_array2d)
        # print(f"Black: {black_count}, White: {white_count}, Move: {line_2_plane(agent_move)}, Player: {current_agent}")
        
        temp_node = make_move(temp_node, agent_move)
        current_agent = -current_agent

    
    black_count = np.sum(game_board.black_array2d)
    white_count = np.sum(game_board.white_array2d)
    
    if black_count > white_count:
        return 1
    elif black_count < white_count:
        return -1
    else:
        return 0

def validate(move):
    """Validate a move."""
    if not (isinstance(move, int) or isinstance(move, np.int64)) or not (0 <= move < config.N ** 2 or move == config.pass_move):
        raise ValueError("move must be integer from [0, 63] or {}, got {}".format(config.pass_move, move))


def make_move(node, move):
    """Make a move and return the new node."""
    validate(move)
    if move not in node.child_nodes:
        node = tree.Node(node, move, -node.player)
    else:
        node = node.child_nodes[move]
    node.is_search_root = True
    node.parent.child_nodes.clear()
    node.parent.is_search_root = False
    return node


if __name__ == "__main__":
    # Create agents
    zero_agent = agents.ZeroAgent
    random_agent = agents.RandomAgent
    minimax_agent = agents.MinimaxAgent
    mcts_agent = agents.MCTSAgent
    edax_agent = agents.EdaxAgent
    
    # Create simulation
    simulation = OthelloSimulation(
        agent1_class=mcts_agent,
        agent1_args={'player_id': config.black},
        agent2_class=edax_agent,
        agent2_args={'level': 4},
        )
    
    # Run simulation
    results = simulation.run_simulation(num_simulations=50, parallel=True)