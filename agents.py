import subprocess
from subprocess import PIPE, STDOUT, Popen
import random
import numpy as np
import time
from numba import uint64

import config
from util import line_2_plane, log, plane_2_line
import net_legacy as net
from heuristics import HEURISTICS
import minmax
import tree # tree for AlphaZero
import search_tree # tree for pure MCTS
from bitboard_utils import get_player_board


class Agent:
    def __init__(self):
        pass

    def make_move(self, current_node):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def copy(self):
        raise NotImplementedError("This method should be overridden by subclasses")


class Player:
    def __init__(self):
        pass

    def make_move(self, current_node):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def copy(self):
        raise NotImplementedError("This method should be overridden by subclasses")


class EdaxAgent(Agent):
    def __init__(self, level):
        edax_exec = config.edax_path + " -q -eval-file " + config.edax_eval_path \
            + " -book-file " + config.edax_book_path + " --level " + str(level) + " -book-randomness 10"
        self.edax = Popen(edax_exec, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        self.read_stdout()

    def make_move(self, current_node):
        if current_node.move == config.pass_move:
            self.write_stdin("pass")
        else:
            self.write_stdin(line_2_plane(current_node.move))
        self.read_stdout()

        self.write_stdin("go")
        edax_move_plane = self.read_stdout().split("plays ")[-1][:2]
        if edax_move_plane == "PS":
            return config.pass_move
        else:
            return plane_2_line(edax_move_plane)
        
    def copy(self):
        return EdaxAgent(config.edax_level)

    def write_stdin(self, command):
        self.edax.stdin.write(str.encode(command + "\n"))
        self.edax.stdin.flush()

    def read_stdout(self):
        out = b''
        while True:
            next_b = self.edax.stdout.read(1)
            if next_b == b'>' and ((len(out) > 0 and out[-1] == 10) or len(out) == 0):
                break
            else:
                out += next_b
        return out.decode("utf-8")

    def close(self):
        self.edax.terminate()


class HumanPlayer(Agent):
    def make_move(self, current_node):
        human_input = -1
        while True:
            human_input_str = input(">")
            if human_input_str == "pass":
                human_input = config.pass_move
            else:
                human_input = plane_2_line(human_input_str)

            if human_input is None or current_node.legal_moves[human_input] == 0:
                print("illegal.")
            else:
                return human_input
    
    def copy(self):
        return HumanPlayer()


class ZeroAgent(Agent):
    def __init__(self, nn=None, temperature=0):
        super().__init__()
        
        if nn is None:
            self.nn = net.restore_legacy_checkpoint(config.checkpoint_path)
        else:
            self.nn = nn
            
        if not isinstance(self.nn, net.NN):
            raise TypeError(f"nn must be an instance of net.NN, got {type(self.nn)}")
            
        self.mcts_batch = tree.MCTS_Batch(self.nn)
        self.temperature = temperature
    
    def make_move(self, current_node):
        pi = self.mcts_batch.alpha([current_node], self.temperature)[0]
        return self._pick_move_greedily(pi)
    
    def copy(self):
        return ZeroAgent(nn=self.nn, temperature=self.temperature)

    def _pick_move_greedily(self, pi):
        return np.argmax(pi)


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()
    
    def make_move(self, current_node):
        legal_moves = current_node.legal_moves
        legal_indices = np.where(legal_moves)[0]
        if len(legal_indices) == 0:
            return config.pass_move
        return random.choice(legal_indices)
    
    def copy(self):
        return RandomAgent()


class MinimaxAgent(Agent):
    def __init__(self, player_id, depth = 5, heuristic=HEURISTICS.HYBRID, verbose=False):
        """
        Initializes the MinimaxAgent with a specified search depth.

        Parameters:
            player_id (int): The ID of the player (1 or 2).
            depth (int): The depth to which the Minimax algorithm will search.
            verbose (bool): If True, prints debug information during search.
        """
        if player_id == -1:
            player_id = 2
        self.id = player_id
        self.depth = depth
        self.heuristic = heuristic
        self.verbose = verbose
        self.bot = None
    
    def make_move(self, current_node):
        """
        Determines the best move for the player using the Minimax algorithm.

        Parameters:
            board (UniTuple(uint64, 2)): The current game board.

        Returns:
            int: The best move for the player.
        """

        if self.bot is None:
            self.bot = minmax.Minmax(self.id, self.heuristic)
        
        board = (current_node.board.black, current_node.board.white)
        best_score, best_move = self.bot.negamax(board, self.depth, config.INT16_NEGINF, config.INT16_POSINF, 1)
        
        if self.verbose:
            print(f"Player {self.id} --> {best_move}/{best_score:<6}")  

        return best_move
    
    def copy(self):
        """
        Creates a copy of the MinimaxAgent.

        Returns:
            MinimaxAgent: A new instance of MinimaxAgent with the same parameters.
        """
        return MinimaxAgent(self.id, self.depth, self.heuristic, self.verbose)


class MCTSAgent(Agent):
    """
    Class for a Monte Carlo Tree Search (MCTS) Agent.
    """
    def __init__(self, player_id, time_limit = None, nb_iterations = 100000, nb_rollouts = 1, c_param = 1.4, verbose = False):
        """
        Initializes the MCTSAgent with specified parameters.

        Parameters:
            player_id (int): The ID of the player (1 or 2). If -1, defaults to 2.
            time_limit (float): The time limit for the MCTS search in seconds.
            nb_iterations (int): The number of iterations for the MCTS search.
            nb_rollouts (int): The number of rollouts for the MCTS search.
            c_param (float): The exploration parameter for the MCTS search.
            verbose (bool): If True, prints debug information during search.
        """
        if player_id == -1:
            player_id = 2
        self.id = player_id
        self.time_limit = time_limit
        self.nb_iterations = nb_iterations
        self.verbose = verbose
        self.c_param = c_param
        self.nb_rollouts = nb_rollouts
        
        self.tree = None
        
    def copy(self):
        """
        Returns a new instance of MCTSAgent with the same parameters as the current instance.

        Returns:
            MCTSAgent: A new instance with the same parameters.
        """
        player_copy = MCTSAgent(
            time_limit = self.time_limit,
            nb_iterations=self.nb_iterations,
            c_param=self.c_param,
            nb_rollouts=self.nb_rollouts,
            verbose=self.verbose
        )
        
        player_copy.set_id(self.id)
        
        return player_copy
    
    def timed_search(self, board):
        """
        Conducts a timed MCTS search to determine the best move.

        Parameters:
            board (UniTuple(uint64, 2)): The current game board.

        Returns:
            int: The best move for the player.
        """
              
        player_board, opponent_board = get_player_board(board, self.id)
        best_move = -1
        nb_loops = 0
        avg_exec_time = 0
        
        root = search_tree.define_root(self.tree, player_board, opponent_board)
        
        global_start_time = time.perf_counter()
        last_loop_end = global_start_time
        
        while last_loop_end-global_start_time < self.time_limit-avg_exec_time:
            search_tree.search_batch(self.tree, root, nb_rollouts=self.nb_rollouts, c_param=self.c_param)
            
            nb_loops += 1
            loop_end = time.perf_counter()
            loop_time = loop_end - last_loop_end
            last_loop_end = loop_end
                        
            if avg_exec_time == 0:
                avg_exec_time = loop_time
            else:
                avg_exec_time = (avg_exec_time*(nb_loops-1) + loop_time)/nb_loops
            
        best_move = self.tree.moves[search_tree.best_child(self.tree, root, c_param=0)]
        
        if self.verbose:
            print(f"Player {self.id} --> move:{best_move:<2} ({time.perf_counter()-global_start_time:>6.2f} sec, {nb_loops*config.MCTS_BATCH_SIZE:<7} iterations, {self.nb_rollouts} rollouts)")

        return best_move
    
    def make_move(self, current_node):
        """
        Determines the best move for the player using the MCTS algorithm.

        Parameters:
            board (UniTuple(uint64, 2)): The current game board.
            events (list): A list of game events.

        Returns:
            int: The best move for the player.
        """
        
        if self.tree is None:
            self.tree = search_tree.SearchTree()

        board = (current_node.board.black, current_node.board.white)
        
        player_board, opponent_board = get_player_board(board, self.id)
        
        start_time = time.perf_counter()
        if self.time_limit is not None:
            best_move = self.timed_search(board)
        else:
            best_move = search_tree.search(self.tree, player_board, opponent_board, self.nb_iterations, self.nb_rollouts, self.c_param)
            
            if self.verbose:
                print(f"Player {self.id} --> move:{best_move:<2} ({time.perf_counter()-start_time:>6.2f} sec, {self.nb_iterations} iterations, {self.nb_rollouts} rollouts)")

        return best_move


if __name__ == "__main__":
    try:
        minmax_agent = MinimaxAgent(depth=3, verbose=True)
        print("MinMax agent initialized successfully")
    except Exception as e:
        print(f"Error initializing MinMax agent: {e}")