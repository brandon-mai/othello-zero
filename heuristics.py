import numpy as np
from numba import int16, uint64, int8, njit
from enum import IntEnum
from numba.types import UniTuple
from bitboard_utils import get_moves_index, possible_moves, find_empty_neighbors_of_player, find_stable_disks, find_unstable_disks, get_player_board, count_bits

class HEURISTICS(IntEnum):
    DISK_PARITY = 0
    MOBILITY = 1
    CORNER = 2
    STABILITY = 3
    STATIC_WEIGHTS = 4
    HYBRID = 5

@njit(cache = True)
def select_heuristic_function(board, player_id, heuristic):
    """
    Select the heuristic function based on the given heuristic type.

    Args:
        board (UniTuple(uint64, 2)): The current state of the board.
        player_id (int16): The ID of the player (1 or 2).
        heuristic (HEURISTICS): The heuristic type to use.

    Returns:
        int16: The heuristic value.
    """
    if heuristic == HEURISTICS.DISK_PARITY:
        return disk_parity_heuristic_standalone(board, player_id)
    elif heuristic == HEURISTICS.MOBILITY:
        return mobility_heuristic_standalone(board, player_id)
    elif heuristic == HEURISTICS.CORNER:
        return corner_heuristic_standalone(board, player_id)
    elif heuristic == HEURISTICS.STABILITY:
        return stability_heuristic_standalone(board, player_id)
    elif heuristic == HEURISTICS.STATIC_WEIGHTS:
        return static_weights_heuristic(board, player_id)
    elif heuristic == HEURISTICS.HYBRID:
        return hybrid_heuristic(board, player_id)
    else:
        raise ValueError("Invalid heuristic type")


@njit(int16(int16, int16), cache=True)
def disk_parity_heuristic(player_disks, opponent_disks):
    """
    Calculate the disk parity heuristic.

    The disk parity heuristic measures the difference in the number of disks 
    between the player and the opponent. It is computed as the percentage difference 
    relative to the total number of disks on the board.

    Args:
        player_disks (int16): The number of disks the player has.
        opponent_disks (int16): The number of disks the opponent has.

    Returns:
        int16: The disk parity heuristic value.
    """
    
    disk_parity_heuristic = int16(100 * (player_disks - opponent_disks) / (player_disks + opponent_disks))
    
    return disk_parity_heuristic

@njit(int16(UniTuple(uint64, 2), int8), cache=True)
def disk_parity_heuristic_standalone(board, player_id):
    """
    Calculate the standalone disk parity heuristic.

    This function calculates the disk parity heuristic based on the current state
    of the board. 
    
    Args:
        board (UniTuple(uint64, 2)): The current state of the board.
        player_id (int16): The ID of the player (1 or 2).

    Returns:
        int16: The disk parity heuristic value.
    """
    bitboard_player, bitboard_opponent = get_player_board(board, player_id)
    player_disks = count_bits(bitboard_player)
    opponent_disks = count_bits(bitboard_opponent)
    
    return disk_parity_heuristic(player_disks, opponent_disks)
        

@njit(int16(int16, int16, int16, int16), cache=True)
def mobility_heuristic(player_moves_a, opponent_moves_a,
                       player_moves_p, opponent_moves_p):
    """
    Calculate the mobility heuristic.

    The mobility heuristic evaluates both actual and potential mobility.
    - Actual mobility refers to the number of legal moves available to the player.
    - Potential mobility refers to the number of potential moves (empty cells adjacent to opponent's disks).

    The heuristic is the average of the actual and potential mobility values, 
    calculated as percentage differences relative to the total number of moves.

    Args:
        player_moves_a (int16): The number of actual moves available to the player.
        opponent_moves_a (int16): The number of actual moves available to the opponent.
        player_moves_p (int16): The number of potential moves for the player.
        opponent_moves_p (int16): The number of potential moves for the opponent.

    Returns:
        int16: The mobility heuristic value.
    """
    
    if(player_moves_a + opponent_moves_a !=0):
        actual_mobility_heuristic = 100 * (player_moves_a - opponent_moves_a)/(player_moves_a + opponent_moves_a)
    else:
        actual_mobility_heuristic = 0
        
    if(player_moves_p + opponent_moves_p !=0):
        potential_mobility_heuristic = 100 * (player_moves_p - opponent_moves_p)/(player_moves_p + opponent_moves_p)
    else:
        potential_mobility_heuristic = 0
        
    mobility_heuristic = (actual_mobility_heuristic + potential_mobility_heuristic)/2
    
    return mobility_heuristic

@njit(int16(UniTuple(uint64, 2), int8), cache=True)
def mobility_heuristic_standalone(board, player_id):
    """
    Calculate the standalone mobility heuristic.

    This function calculates the mobility heuristic based on the current state of the board.

    Args:
        board (UniTuple(uint64, 2)): The current state of the board.
        player_id (int16): The ID of the player (1 or 2).

    Returns:
        int16: The mobility heuristic value.
    """
    
    player_bb, opponent_bb = get_player_board(board, player_id)
    empty_squares = (player_bb | opponent_bb) ^ 0xFFFFFFFFFFFFFFFF
    
    player_possible_moves = possible_moves(player_bb, opponent_bb, empty_squares)
    opponent_possible_moves = possible_moves(opponent_bb, player_bb, empty_squares)
    
    player_actual_move_nb = count_bits(player_possible_moves)
    opponent_actual_move_nb = count_bits(opponent_possible_moves)
    
    player_adjacent_cells = find_empty_neighbors_of_player(board, player_id)
    opponent_adjacent_cells = find_empty_neighbors_of_player(board, 3 - player_id)
    
    player_potential_move_nb = count_bits(opponent_adjacent_cells)
    opponent_potential_move_nb = count_bits(player_adjacent_cells)
    
    return mobility_heuristic(player_moves_a=player_actual_move_nb, 
                              opponent_moves_a=opponent_actual_move_nb, 
                              player_moves_p=player_potential_move_nb, 
                              opponent_moves_p=opponent_potential_move_nb)

@njit(int16(UniTuple(uint64, 2), int8, int8[::1], int8[::1]), cache=True)
def corner_heuristic(bitboard, player_id, player_possible_moves, opponent_possible_moves):
    """
    Calculate the corner control heuristic.

    The corner heuristic evaluates the control of corner squares (A1, A8, H1, H8).
    It considers both the number of corners occupied and the potential to occupy corners 
    (possible moves to corner positions).

    The heuristic assigns a higher weight to actually captured corners and a lower 
    weight to potential corners, calculated as a percentage difference relative to 
    the total corner values.

    Args:
        bitboard (UniTuple(uint64, 2)): The bitboard representation of the board.
        player_id (int16): The ID of the player (1 or 2).
        player_possible_moves (int8[::1]): The possible moves for the player.
        opponent_possible_moves (int8[::1]): The possible moves for the opponent.

    Returns:
        int16: The corner control heuristic value.
    """
    
    corners_mask = uint64(0x8100000000000081) # Corners: a1, a8, h1, h8
    
    bitboard_player, bitboard_opponent = get_player_board(bitboard, player_id)
    
    player_corners = count_bits(bitboard_player & corners_mask)
    opponent_corners = count_bits(bitboard_opponent & corners_mask)
    
    player_potential_corners, opponent_potential_corners = 0, 0
    
    # Iterate through the list of possible moves
    for move in player_possible_moves:
        if (uint64(1) << move) & corners_mask:
            player_potential_corners += 1
            
    for move in opponent_possible_moves:
        if (uint64(1) << move) & corners_mask:
            opponent_potential_corners += 1
    
    corners_captured_weight, potential_corners_weight = 2, 1
    
    player_corner_value = corners_captured_weight * player_corners + potential_corners_weight * player_potential_corners
    opponent_corner_value = corners_captured_weight * opponent_corners + potential_corners_weight * opponent_potential_corners
    
    if player_corner_value + opponent_corner_value != 0:
        corner_heuristic = 100 * (player_corner_value - opponent_corner_value) / (player_corner_value + opponent_corner_value)
    else:
        corner_heuristic = 0
    
    return corner_heuristic

@njit(int16(UniTuple(uint64, 2), int8), cache=True)
def corner_heuristic_standalone(board, player_id):
    """
    Calculate the standalone corner control heuristic based on the current state of the board.

    Args:
        board (UniTuple(uint64, 2)): The current state of the board.
        player_id (int16): The ID of the player (1 or 2).

    Returns:
        int16: The corner control heuristic value.
    """
    
    player_bb, opponent_bb = get_player_board(board, player_id)
    
    empty_squares = (player_bb | opponent_bb) ^ 0xFFFFFFFFFFFFFFFF
    possible_moves_bb_player = possible_moves(player_bb, opponent_bb, empty_squares)
    player_possible_moves = get_moves_index(possible_moves_bb_player)
    
    possible_moves_bb_opponent = possible_moves(opponent_bb, player_bb, empty_squares)
    opponent_possible_moves = get_moves_index(possible_moves_bb_opponent)
    
    return corner_heuristic(board, player_id, player_possible_moves, opponent_possible_moves)

@njit(int16(UniTuple(uint64, 2), int8, int8[::1], int8[::1], uint64, uint64), cache=True)
def stability_heuristic(bitboard, player_id, player_possible_moves, opponent_possible_moves,
                        player_adjacent_cells, opponent_adjacent_cells):
    """
    Calculate the stability heuristic.

    The stability heuristic evaluates the number of stable and unstable disks.
    - Stable disks are those that cannot be flipped for the rest of the game.
    - Unstable disks are those that can be flipped at the next move of the opponent.

    The heuristic measures the difference between stable and unstable disks for 
    the player and the opponent, calculated as a percentage difference relative 
    to the total stability value.

    Args:
        bitboard (UniTuple(uint64, 2)): The bitboard representation of the board.
        player_id (int16): The ID of the player (1 or 2).
        player_possible_moves (int8[::1]): The possible moves for the player.
        opponent_possible_moves (int8[::1]): The possible moves for the opponent.
        player_adjacent_cells (uint64): The bitboard of empty cells adjacent to the player's disks.
        opponent_adjacent_cells (uint64): The bitboard of empty cells adjacent to the opponent's disks.

    Returns:
        int16: The stability heuristic value.
    """
    
    opponent_id = 3 - player_id
    
    player_stable_disks = find_stable_disks(player_id, bitboard, player_adjacent_cells)
    opponent_stable_disks = find_stable_disks(opponent_id, bitboard, opponent_adjacent_cells)
    
    player_unstable_disks = find_unstable_disks(player_id, bitboard, opponent_possible_moves)
    opponent_unstable_disks = find_unstable_disks(opponent_id, bitboard, player_possible_moves)
    
    stable_weight = 2
    unstable_weight = 1
    
    if(player_stable_disks + opponent_stable_disks != 0):
        stable_disk_heuristic = 100 * (player_stable_disks-opponent_stable_disks)/(player_stable_disks+opponent_stable_disks)
    else:
        stable_disk_heuristic = 0
    
    if(player_unstable_disks + opponent_unstable_disks != 0):
        unstable_disk_heuristic = 100 * (opponent_unstable_disks-player_unstable_disks)/(player_unstable_disks+opponent_unstable_disks)
    else:
        unstable_disk_heuristic = 0
        
    stability_heuristic = (stable_weight * stable_disk_heuristic + unstable_weight * unstable_disk_heuristic)/(stable_weight+unstable_weight)
        
    return stability_heuristic

@njit(int16(UniTuple(uint64, 2), int8), cache=True)
def stability_heuristic_standalone(board, player_id):
    """
    Calculate the standalone stability heuristic based on the current state of the board.

    Args:
        board (UniTuple(uint64, 2)): The current state of the board.
        player_id (int16): The ID of the player (1 or 2).

    Returns:
        int16: The stability heuristic value.
    """
    opponent_id = 3 - player_id
    
    player_adjacent_cells = find_empty_neighbors_of_player(board, player_id)
    opponent_adjacent_cells = find_empty_neighbors_of_player(board, 3 - player_id)
    
    player_bb, opponent_bb = get_player_board(board, player_id)
    
    empty_squares = (player_bb | opponent_bb) ^ 0xFFFFFFFFFFFFFFFF
    possible_moves_bb_player = possible_moves(player_bb, opponent_bb, empty_squares)
    player_possible_moves = get_moves_index(possible_moves_bb_player)
    
    possible_moves_bb_opponent = possible_moves(opponent_bb, player_bb, empty_squares)
    opponent_possible_moves = get_moves_index(possible_moves_bb_opponent)
    
    return stability_heuristic(bitboard=board, player_id=player_id, 
                               player_possible_moves=player_possible_moves, opponent_possible_moves=opponent_possible_moves, 
                               player_adjacent_cells=player_adjacent_cells, opponent_adjacent_cells=opponent_adjacent_cells)

@njit(int16(UniTuple(uint64, 2), int16), cache = True)
def hybrid_heuristic(board, player_id):
    """
    Evaluate the board state using a hybrid heuristic.

    The hybrid heuristic combines multiple heuristics to evaluate the board state:
    - Disk Parity: Measures the difference in the number of disks.
    - Mobility: Evaluates actual and potential mobility.
    - Corner Control: Assesses control and potential control of corner squares.
    - Stability: Measures the number of stable and unstable disks.

    Each heuristic is weighted and combined to compute a final evaluation score.
    
    Args:
        board (UniTuple(uint64, 2)): The current state of the board.
        player_id (int16): The ID of the player (1 or 2).

    Returns:
        int16: The final hybrid heuristic score.
    """
    opponent_id = 3 - player_id
    bitboard_player, bitboard_opponent = get_player_board(board, player_id)
    
    # Number of disks
    player_disks = count_bits(bitboard_player)
    opponent_disks = count_bits(bitboard_opponent)
    
    # Possible moves
    empty_squares = (bitboard_player | bitboard_opponent) ^ 0xFFFFFFFFFFFFFFFF
    possible_moves_bb_player = possible_moves(bitboard_player, bitboard_opponent, empty_squares)
    player_possible_moves = get_moves_index(possible_moves_bb_player)
    
    possible_moves_bb_opponent = possible_moves(bitboard_opponent, bitboard_player, empty_squares)
    opponent_possible_moves = get_moves_index(possible_moves_bb_opponent)
    
    # Adjacents cells
    player_adjacent_cells = find_empty_neighbors_of_player(board, player_id)
    opponent_adjacent_cells = find_empty_neighbors_of_player(board, opponent_id)
    
    player_actual_move_nb = player_possible_moves.shape[0]
    opponent_actual_move_nb = opponent_possible_moves.shape[0]
    
    # Potential moves for each players
    player_potential_move_nb = count_bits(opponent_adjacent_cells)
    opponent_potential_move_nb = count_bits(player_adjacent_cells)
    
    # =============== Game Over ===============
    
    if player_actual_move_nb + opponent_actual_move_nb == 0:
        disk_diff = player_disks - opponent_disks
        return 400*disk_diff
    
    # =============== Heuristics Values ===============
    
    disk_parity_heuristic_value = disk_parity_heuristic(player_disks, opponent_disks)

    mobility_heuristic_value = mobility_heuristic(player_moves_a=player_actual_move_nb, 
                                            opponent_moves_a=opponent_actual_move_nb, 
                                            player_moves_p=player_potential_move_nb, 
                                            opponent_moves_p=opponent_potential_move_nb)

    corner_heuristic_value = corner_heuristic(board, player_id, player_possible_moves, opponent_possible_moves)
        
    stability_heuristic_value = stability_heuristic(bitboard=board, player_id=player_id, 
                                        player_possible_moves=player_possible_moves, opponent_possible_moves=opponent_possible_moves, 
                                        player_adjacent_cells=player_adjacent_cells, opponent_adjacent_cells=opponent_adjacent_cells)
    
    # =============== Final Score ===============
    
    disk_nb = player_disks+opponent_disks
    
    # scale exponentially the weight of this heursitic as the game progresses 
    disk_parity_weight = (1+(disk_nb/64))**6
    mobility_weight = 20
    corner_weight = 50
    stability_weight = 40
    
    final_score = disk_parity_weight*disk_parity_heuristic_value + mobility_weight*mobility_heuristic_value \
                + corner_weight*corner_heuristic_value + stability_weight*stability_heuristic_value
                
    final_score /= (disk_parity_weight + mobility_weight + corner_weight + stability_weight)
    
    return int16(final_score)

STATIC_WEIGHTS = np.array([
    [ 4, -3,  2,  2,  2,  2, -3,  4],
    [-3, -4, -1, -1, -1, -1, -4, -3],
    [ 2, -1,  1,  0,  0,  1, -1,  2],
    [ 2, -1,  0,  1,  1,  0, -1,  2],
    [ 2, -1,  0,  1,  1,  0, -1,  2],
    [ 2, -1,  1,  0,  0,  1, -1,  2],
    [-3, -4, -1, -1, -1, -1, -4, -3],
    [ 4, -3,  2,  2,  2,  2, -3,  4]
], dtype=np.int16).flatten()

@njit(int16(UniTuple(uint64, 2), int8), cache=True)
def static_weights_heuristic(board, player_id):
    """
    Evaluates the board state with Static Weights Heuristic.

    This heuristic evaluates the board state by performing matrix multiplication between the board state
    matrix and the weights matrix. It sums up the products to obtain the final evaluation score.

    Args:
        board (UniTuple(uint64, 2)): Tuple of two bitboards representing the board state for both players.

    Returns:
        int16: The final score of the weighted piece value heuristic.
    """
    player1_board, player2_board = board

    player1_score = 0
    player2_score = 0

    for i in range(64):
        if (player1_board >> i) & 1:
            player1_score += STATIC_WEIGHTS[i]
        if (player2_board >> i) & 1:
            player2_score += STATIC_WEIGHTS[i]
            
    if player_id == 1:
        return player1_score - player2_score
    return player2_score - player1_score