import argparse
import gc
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
import time
import random
import traceback
import concurrent.futures
from multiprocessing import Pool, Process, Manager
from tqdm.auto import tqdm

import numpy as np
import tensorflow as tf

import player
import board
import config
import gui
import net_v2 as net
import tree
from util import log, plane_2_line


class SelfPlayGame:
    def __init__(
            self,
            worker_id,
            batch_size=config.self_play_batch_size,
            echo_max=config.self_play_echo_max,
            checkpoint_path=config.checkpoint_path,
            data_path=config.data_path
            ):
        self.version = 0
        self.echo = 0
        self.echo_max = echo_max
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.fake_nodes = [None] * batch_size
        self.current_nodes = [None] * batch_size

    def start(self):
        # Configure GPU memory growth for TF2
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # Set memory limit if needed
                tf.config.experimental.set_memory_limit(
                    gpus[0], 
                    int(1024 * config.self_play_woker_gpu_memory_fraction)
                )
            except RuntimeError as e:
                log(f"GPU configuration error: {e}")

        nn = net.NN()
        self.version = restore_from_last_checkpoint(
            nn.model, 
            checkpoint_path=self.checkpoint_path
        ).get("version", 0)
        mcts_batch = tree.MCTS_Batch(nn)

        try:
            for f_name in os.listdir(self.data_path):
                if f_name.endswith(".npz"):
                    os.remove(os.path.join(self.data_path, f_name)) # wipe
        except OSError as e:
            log(f"Error clearing data files: {e}")
        
        while self.echo < self.echo_max:
            log("selfplay worker", self.worker_id, "version:", self.version, "echo:", self.echo, "session start.")
            self.play(mcts_batch)
            self.save()
            self.echo += 1
        log("selfplay worker", self.worker_id, "session end.")

    def play(self, mcts_batch):
        terminals_num = 0
        moves_num = 0
        for i in range(self.batch_size):
            self.fake_nodes[i] = tree.FakeNode()
            self.current_nodes[i] = tree.Node(self.fake_nodes[i], 0, config.black, board.Board())
            self.current_nodes[i].is_game_root = True
            self.current_nodes[i].is_search_root = True

        while terminals_num != self.batch_size:
            terminals_num = 0
            moves_num += 1
            gc.collect()
            pi_batch = mcts_batch.alpha(self.current_nodes, get_temperature(moves_num))
            for i in range(self.batch_size):
                if self.current_nodes[i].is_terminal is True:
                    terminals_num += 1
                else:
                    move = pick_move_probabilistically(pi_batch[i])
                    self.current_nodes[i] = make_move(self.current_nodes[i], move)

    def save(self):
        features_array = []
        pi_array = []
        winner_array = []
        for node in self.current_nodes:
            winner = 0
            black_stones_num = np.sum(node.board.black_array2d)
            white_stones_num = np.sum(node.board.white_array2d)
            if black_stones_num > white_stones_num:
                winner = 1
            elif black_stones_num < white_stones_num:
                winner = -1
            
            current = node
            while True:
                features_array.append(current.to_features())
                pi_array.append(current.pi)
                winner_array.append(winner)
                if current.is_game_root:
                    break
                current = current.parent
        np.savez_compressed(
            self.data_path + "{0:03d}_{1:03d}_{2:02d}{3:02d}".format(self.batch_size, self.version, self.worker_id, self.echo),
            features_array=features_array,
            pi_array=pi_array,
            winner_array=winner_array
            )


class Train:
    def __init__(
            self,
            batch_size=config.train_batch_size,
            num_train_epoch=config.num_train_epoch,
            checkpoint_path=config.checkpoint_path,
            data_path=config.data_path
            ):
        self.version = 0
        self.state_data = np.zeros((0, config.N, config.N, config.history_num * 2 + 1), dtype=np.float32)
        self.pi_data = np.zeros((0, config.all_moves_num), dtype=np.float32)
        self.z_data = np.zeros((0, 1), dtype=np.float32)
        self.batch_size = batch_size
        self.num_train_epoch = num_train_epoch
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.data_len = self.load_data()
        self.batch_num = (self.data_len // self.batch_size) + 1
        self.global_step = 0
    
    def start(self):
        if self.data_len == 0:
            log("no data for training.")
            return
            
        nn = net.NN()

        self.version, self.global_step = restore_from_last_checkpoint(
            nn.model, 
            checkpoint_path=self.checkpoint_path).values()
        self.version += 1
        
        # New version checkpoint manager
        checkpoint = tf.train.Checkpoint(model=nn.model)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=checkpoint, 
            directory=config.checkpoint_path,
            max_to_keep=config.train_checkpoint_max_to_keep,
            checkpoint_name=f"v{self.version:03d}"
        )
        
        log("training version:", self.version, "global step:", self.global_step, "session start.")
        
        with open(config.log_path + "loss_log.csv", "a+") as loss_log_file:
            for epoch in range(self.num_train_epoch):
                for batch_index in tqdm(range(self.batch_num), desc="Training Epoch {0}".format(epoch + 1)):
                    self.global_step += 1
                    state_batch, pi_batch, z_batch = self.get_next_batch(batch_index, self.batch_size)
                    p_loss, v_loss = nn.train(state_batch, pi_batch, z_batch)
                    loss_log_file.write("{0},{1},{2}\n".format(self.global_step, p_loss, v_loss))
        
        log("Starting model evaluation...")
        passed_evaluation = evaluate_model(
            nn, 
            num_games=400, 
            pass_threshold=0.55
        )
        if passed_evaluation:
            log("New model passed evaluation, saving checkpoint.")
            # Save the new model
            checkpoint_manager.save(checkpoint_number=self.global_step)
        else:
            log("New model failed evaluation, not saving checkpoint.")

        log("training session end.")

    def load_data(self):
        npz_file_names = get_npz_file_names(self.data_path)
        if len(npz_file_names) == 0:
            self.data_len = 0
            return self.data_len

        for npz_file_name in npz_file_names:
            npz_data = np.load(self.data_path + npz_file_name)
            features_array = npz_data['features_array']
            pi_array = npz_data['pi_array']
            winner_array = npz_data['winner_array']
            data_len = len(features_array)
            _state_data = np.zeros((data_len, config.N, config.N, config.history_num * 2 + 1), dtype=np.float32)
            _pi_data = np.zeros((data_len, config.all_moves_num), dtype=np.float32)
            _z_data = np.zeros((data_len, 1), dtype=np.float32)
            for i in range(data_len):
                _state_data[i] = features_array[i]
                _pi_data[i] = pi_array[i]
                _z_data[i] = winner_array[i]
            self.state_data = np.concatenate((self.state_data, _state_data))
            self.pi_data = np.concatenate((self.pi_data, _pi_data))
            self.z_data = np.concatenate((self.z_data, _z_data))
        
        self.data_len = len(self.state_data)
        return self.data_len
    
    def get_next_batch(self, index, size):
        start = index * size
        end = (index + 1) * size
        if start >= self.data_len:
            start = self.data_len - size
        if end > self.data_len:
            end = self.data_len
        return self.state_data[start:end], self.pi_data[start:end], self.z_data[start:end]


class Evaluate:
    def __init__(
            self,
            num_games=400,
            pass_threshold=0.55,
            checkpoint_path=config.checkpoint_path,
            batch_size=config.self_play_batch_size
            ):
        self.num_games = num_games
        self.pass_threshold = pass_threshold
        self.checkpoint_path = checkpoint_path
        self.batch_size = min(batch_size, num_games)
        self.new_model_wins = 0
        self.total_games = 0
        
    def start(self, latest_nn):
        best_nn = net.NN()
        current_version = restore_from_last_checkpoint(
            best_nn.model, 
            checkpoint_path=self.checkpoint_path
        ).get("version", 0)
        
        if current_version == 0:
            log("No current model found, passing evaluation.")
            return True
        
        log(f"Evaluation: Latest vs Best (v{current_version}), {self.num_games} games, {self.pass_threshold:.1%} threshold")
        
        self.new_model_wins = 0
        self.total_games = 0
        
        while self.total_games < self.num_games:
            remaining_games = self.num_games - self.total_games
            current_batch_size = min(self.batch_size, remaining_games)
            
            batch_wins = self.play_batch(latest_nn, best_nn, current_batch_size)
            self.new_model_wins += batch_wins
            self.total_games += current_batch_size
            
            win_rate = self.new_model_wins / self.total_games
            log(f"Batch complete: {batch_wins}/{current_batch_size} wins. Total: {self.new_model_wins}/{self.total_games} = {win_rate:.3f}")
            
            if self._can_determine_result():
                log("Early stopping condition met")
                break
        
        win_rate = self.new_model_wins / self.total_games if self.total_games > 0 else 0
        passed = win_rate >= self.pass_threshold
        
        log(f"Evaluation complete: {self.new_model_wins}/{self.total_games} = {win_rate:.3f}")
        log(f"Result: {'PASS' if passed else 'FAIL'}")
        return passed
    
    def play_batch(self, latest_nn, best_nn, batch_size):
        current_nodes = []
        game_assignments = []
        
        for i in range(batch_size):
            latest_model_is_black = random.choice([True, False])
            game_assignments.append(latest_model_is_black)
            
            fake_node = tree.FakeNode()
            node = tree.Node(fake_node, 0, config.black, board.Board())
            node.is_game_root = True
            node.is_search_root = True
            current_nodes.append(node)
        
        new_mcts = tree.MCTS_Batch(latest_nn)
        current_mcts = tree.MCTS_Batch(best_nn)
        
        terminals_num = 0
        moves_num = 0
        
        while terminals_num != batch_size:
            terminals_num = 0
            moves_num += 1
            gc.collect()
            
            latest_model_nodes = []
            best_model_nodes = []
            latest_model_indices = []
            best_model_indices = []
            
            for i in range(batch_size):
                if current_nodes[i].is_terminal:
                    terminals_num += 1
                    continue
                    
                current_player_is_black = (current_nodes[i].player == config.black)
                is_latest_model_turn = (game_assignments[i] == current_player_is_black)
                
                if is_latest_model_turn:
                    latest_model_nodes.append(current_nodes[i])
                    latest_model_indices.append(i)
                else:
                    best_model_nodes.append(current_nodes[i])
                    best_model_indices.append(i)
            
            if latest_model_nodes:
                pi_batch = new_mcts.alpha(latest_model_nodes, get_temperature(moves_num))
                for idx, pi in zip(latest_model_indices, pi_batch):
                    move = pick_move_probabilistically(pi)
                    current_nodes[idx] = make_move(current_nodes[idx], move)
            
            if best_model_nodes:
                pi_batch = current_mcts.alpha(best_model_nodes, get_temperature(moves_num))
                for idx, pi in zip(best_model_indices, pi_batch):
                    move = pick_move_probabilistically(pi)
                    current_nodes[idx] = make_move(current_nodes[idx], move)
        
        batch_wins = 0.0
        for i in range(batch_size):
            winner = self._get_winner(current_nodes[i])
            latest_model_is_black = game_assignments[i]
            
            if (winner == 1 and latest_model_is_black) or (winner == -1 and not latest_model_is_black):
                batch_wins += 1.0
            elif winner == 0:
                batch_wins += 0.5
        
        return batch_wins
    
    def _get_winner(self, node):
        black_stones = np.sum(node.board.black_array2d)
        white_stones = np.sum(node.board.white_array2d)
        
        if black_stones > white_stones:
            return 1
        elif black_stones < white_stones:
            return -1
        else:
            return 0
    
    def _can_determine_result(self):
        if self.total_games == 0:
            return False
            
        games_remaining = self.num_games - self.total_games
        max_possible_wins = self.new_model_wins + games_remaining
        
        if max_possible_wins / self.num_games < self.pass_threshold:
            return True
            
        if self.new_model_wins / self.num_games >= self.pass_threshold:
            return True
            
        return False


def evaluate_model(latest_nn, num_games=400, pass_threshold=0.55):
    """
    Convenience function to evaluate a new model.
    Args:
        latest_nn: The new NN with newly trained model to evaluate
        num_games: Total number of evaluation games
        pass_threshold: Win rate threshold to pass evaluation
        
    Returns:
        bool: True if model passes evaluation
    """
    evaluator = Evaluate(
        num_games=num_games,
        pass_threshold=pass_threshold,
        batch_size=num_games // 10,
    )
    return evaluator.start(latest_nn)


def pick_move_probabilistically(pi):
    r = random.random()
    s = 0
    for move in range(len(pi)):
        s += pi[move]
        if s >= r:
            return move
    return np.argmax(pi)


def pick_move_greedily(pi):
    return np.argmax(pi)


def get_temperature(moves_num):
    if moves_num <= 6:
        return 1
    else:
        return 0.95 ** (moves_num - 6)


def validate(move):
    if not (isinstance(move, int) or isinstance(move, np.int64)) or not (0 <= move < config.N ** 2 or move == config.pass_move):
        raise ValueError("move must be integer from [0, 63] or {}, got {}".format(config.pass_move, move))


def make_move(node, move):
    validate(move)
    if move not in node.child_nodes:
        node = tree.Node(node, move, -node.player)
    else:
        node = node.child_nodes[move]
    node.is_search_root = True
    node.parent.child_nodes.clear()
    node.parent.is_search_root = False
    return node


def print_winner(node):
    black_stones_num = np.sum(node.board.black_array2d)
    white_stones_num = np.sum(node.board.white_array2d)
    if black_stones_num > white_stones_num:
        print("black wins.")
    elif black_stones_num < white_stones_num:
        print("white wins.")
    else:
        print("draw.")


def restore_from_last_checkpoint(model, checkpoint_path=config.checkpoint_path):
    """Restore model from the last checkpoint.
    Args:
        model: The model to restore.
    Returns:
        Metadata dict with version and global_step, default to 0 if no checkpoint found.
    """
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=1)
    
    metadata = {"version": 0, "global_step": 0}

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        log("restored from last checkpoint:", checkpoint_manager.latest_checkpoint)
        metadata["version"] = int(re.search(r'v(\d+)', checkpoint_manager.latest_checkpoint).group(1))
        metadata["global_step"] = int(checkpoint_manager.latest_checkpoint.split('-')[-1])
    else:
        log("checkpoint not found.")
    return metadata


def get_npz_file_names(data_path=config.data_path):
    npz_file_names = []
    walk = os.walk(data_path)
    for dpath, _, fnames in walk:
        if dpath == data_path:
            for fname in fnames:
                if fname.split('.')[-1] == "npz":
                    npz_file_names.append(fname)
    return npz_file_names


def self_play_woker(worker_id, checkpoint_path=config.checkpoint_path, data_path=config.data_path):
    try:
        game = SelfPlayGame(worker_id, checkpoint_path=checkpoint_path, data_path=data_path)
        game.start()
    except Exception as ex:
        traceback.print_exc()


def train_woker(checkpoint_path=config.checkpoint_path, data_path=config.data_path):
    try:
        train = Train(checkpoint_path=checkpoint_path, data_path=data_path)
        train.start()
    except Exception as ex:
        traceback.print_exc()    


def learning_loop(
        self_play_wokers_num=config.self_play_wokers_num,
        echo_max=config.learning_loop_echo_max,
        self_play=True,
        train=True,
        checkpoint_path=config.checkpoint_path,
        data_path=config.data_path
        ):
    
    for i in range(echo_max):
        if self_play:
            pool = Pool(self_play_wokers_num)
            for i in range(self_play_wokers_num):
                pool.apply_async(self_play_woker, (i, checkpoint_path, data_path))
            pool.close()
            pool.join()
        if train:
            process = Process(target=train_woker, args=(checkpoint_path, data_path))
            process.start()
            process.join()


def play_game(player1, player2):
    """Play a game between two players.
    Args:
        player1: First player (plays as black)
        player2: Second player (plays as white)
    Returns:
        int: 1 if player1 wins, -1 if player2 wins, 0 if draw
    """
    # Check if both players inherit from Player class
    if not isinstance(player1, player.Player):
        raise TypeError(f"player1 must inherit from Player class, got {type(player1)}")
    if not isinstance(player2, player.Player):
        raise TypeError(f"player2 must inherit from Player class, got {type(player2)}")
    
    moves_num = 0
    current_node = tree.Node(tree.FakeNode(), 0, config.black, board.Board())
    current_node.is_game_root = True
    current_node.is_search_root = True

    def make_move_with_gui(current_node, move):
        current_node = make_move(current_node, move)
        gui.print_node(current_node)
        return current_node

    while not current_node.is_terminal:
        gc.collect()
        moves_num += 1

        if current_node.player == config.black:
            player_move = player1.make_move(current_node)
            print(f"Player 1 move: {player_move}")
        else:
            player_move = player2.make_move(current_node)
            print(f"Player 2 move: {player_move}")
        
        current_node = make_move_with_gui(current_node, player_move)
    
    # Determine winner
    black_stones_num = np.sum(current_node.board.black_array2d)
    white_stones_num = np.sum(current_node.board.white_array2d)
    
    print_winner(current_node)
    
    if black_stones_num > white_stones_num:
        return 1  # Player 1 (black) wins
    elif black_stones_num < white_stones_num:
        return -1  # Player 2 (white) wins
    else:
        return 0  # Draw


class ZeroPlayer(player.Player):
    """AI player using the trained neural network and MCTS.
    Args:
        nn: The neural network to play.
    """
    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        if not isinstance(self.nn, net.NN):
            raise TypeError(f"nn must be an instance of net.NN, got {type(self.nn)}")
        self.mcts_batch = tree.MCTS_Batch(self.nn)
    
    def make_move(self, current_node):
        pi = self.mcts_batch.alpha([current_node], 0)[0]
        return pick_move_greedily(pi)


def play_with_edax(edax_level=config.edax_level):
    """Play Zero vs Edax and return result."""
    nn = net.NN()
    restore_from_last_checkpoint(nn.model, checkpoint_path=config.checkpoint_path)
    zero_player = ZeroPlayer(nn)
    edax_player = player.EdaxPlayer(edax_level)
    result = play_game(zero_player, edax_player)
    edax_player.close()
    return result


def play_with_human():
    """Play Zero vs Human and return result."""
    nn = net.NN()
    restore_from_last_checkpoint(nn.model, checkpoint_path=config.checkpoint_path)
    zero_player = ZeroPlayer(nn)
    human_player = player.HumanPlayer()
    return play_game(zero_player, human_player)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learning-loop", help='start a learning loop from the latest model, or a new random model if there is no any model', action="store_true")
    parser.add_argument("-e", "--play-with-edax", help='play with edax, and print every move. but you need compile edax and copy it to right path first', action="store_true")
    parser.add_argument("-m", "--play-with-human", help='play with you on the command line', action="store_true")
    args = parser.parse_args()
    if args.learning_loop:
        learning_loop()
    elif args.play_with_edax:
        play_with_edax()
    elif args.play_with_human:
        play_with_human()
    else:
        learning_loop(self_play=True)