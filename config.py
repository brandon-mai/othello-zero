import os

# don't touch
N = 8                  # 8x8 Othello board
board_length = 64      # Total squares
pass_move = 64         # Pass move index
all_moves_num = 65     # 64 moves + 1 pass
black = 1
white = -1

# mcts config
c_puct = 1             # UCB exploration constant (standard AlphaZero value)
simulations_num = 400  # MCTS simulations per move
noise_alpha = 0.5      # Dirichlet noise alpha for exploration
noise_weight = 0.25    # Weight of exploration noise

# nn config
history_num = 1          # Number of previous board states to include
residual_blocks_num = 9  # Number of ResNet blocks
filters_num = 64         # Number of filters in convolutional layers
momentum = 0.9           # SGD momentum
l2_weight = 1e-4         # L2 regularization weight
learning_rate = 1e-2     # Initial learning rate

# learning config
self_play_wokers_num = 4                    # Number of parallel self-play workers
self_play_woker_gpu_memory_fraction = 0.04  # Memory per worker

self_play_batch_size = 128  # Games played simultaneously per worker
train_batch_size = 128      # Training batch size

self_play_echo_max = 4            # Self-play iterations per worker: 4 * 128 * 4 = 2048 games
num_train_epoch = 50              # Epochs
train_checkpoint_max_to_keep = 1  # Checkpoint retention
learning_loop_echo_max = 1        # Number of times to run self-play + training loop (why should it be more than 1?)

# edax config
edax_level = 1
edax_path = "./edax/Edax"
edax_eval_path = "./edax/data/eval.dat"
edax_book_path = "./edax/data/book.dat"

# path config
checkpoint_path = "./checkpoint/"
data_path = "./data/"
log_path = "./log/"
if os.path.exists(checkpoint_path) is not True:
    os.mkdir(checkpoint_path)
if os.path.exists(data_path) is not True:
    os.mkdir(data_path)
if os.path.exists(log_path) is not True:
    os.mkdir(log_path)

# Hardware-optimized for 4-core CPU
simulations_num = 100
# residual_blocks_num = 4
self_play_wokers_num = 4
# self_play_batch_size = 32  # 2 workers * 2 echoes * 32 games = 128 games
# train_batch_size = 32
# num_train_epoch = 2
# learning_rate = 5e-3
