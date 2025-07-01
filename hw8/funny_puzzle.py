import heapq

#used AI to help with implemting A* algorithm
#used AI to help debug code and use proper syntax for heapqueues
def state_check(state):
    """check the format of state, and return corresponding goal state.
       Do NOT edit this function."""
    non_zero_numbers = [n for n in state if n != 0]
    num_tiles = len(non_zero_numbers)
    if num_tiles == 0:
        raise ValueError('At least one number is not zero.')
    elif num_tiles > 9:
        raise ValueError('At most nine numbers in the state.')
    matched_seq = list(range(1, num_tiles + 1))
    if len(state) != 9 or not all(isinstance(n, int) for n in state):
        raise ValueError('State must be a list contain 9 integers.')
    elif not all(0 <= n <= 9 for n in state):
        raise ValueError('The number in state must be within [0,9].')
    elif len(set(non_zero_numbers)) != len(non_zero_numbers):
        raise ValueError('State can not have repeated numbers, except 0.')
    elif sorted(non_zero_numbers) != matched_seq:
        raise ValueError('For puzzles with X tiles, the non-zero numbers must be within [1,X], '
                          'and there will be 9-X grids labeled as 0.')
    goal_state = matched_seq
    for _ in range(9 - num_tiles):
        goal_state.append(0)
    return tuple(goal_state)

def get_manhattan_distance(from_state, to_state):
    """
    INPUT: 
        Two states (The first one is current state, and the second one is goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """

    before_state = {}
    after_state = {}

    for i in range(9):
        if from_state[i] > 0:
            before_state[from_state[i]] = i
        if to_state[i] > 0:
            after_state[to_state[i]] = i

    distance = 0

    for value, position1 in before_state.items():
        if value in after_state:
            position2 = after_state[value]

            row1 = position1 // 3
            col1 = position1 % 3

            row2 = position2 // 3
            col2 = position2 % 3

            man_dist = abs(row1 - row2) + abs(col1 - col2)
            distance += man_dist

    return distance

def naive_heuristic(from_state, to_state):
    """
    INPUT: 
        Two states (The first one is current state, and the second one is goal state)

    RETURNS:
        0 (but experimenting with other constants is encouraged)
    """
    return 0

def sum_of_squares_distance(from_state, to_state):
    """
    INPUT: 
        Two states (The first one is current state, and the second one is goal state)

    RETURNS:
        A scalar that is the sum of squared distances for all tiles
    """
    before_state = {}
    after_state = {}
    distance = 0

    for i in range(9):
        if from_state[i] > 0:
            before_state[from_state[i]] = i
        if to_state[i] > 0:
            after_state[to_state[i]] = i

    for value, position1 in before_state.items():
        if value in after_state:
            position2 = after_state[value]

            row1 = position1 // 3
            col1 = position1 % 3

            row2 = position2 // 3
            col2 = position2 % 3

            sum_of_squares_dist = (row1 - row2)**2 + (col1 - col2)**2
            distance += sum_of_squares_dist
    return distance

def print_succ(state, heuristic=get_manhattan_distance):
    """
    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """

    goal_state = state_check(state)
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(heuristic(succ_state,goal_state)))


def get_succ(state):
    """
    Get all valid successors of the given state.

    INPUT:
        A state (list of length 9)

    RETURNS:
        A list of all valid successors in the puzzle (sorted).
    """
    succ_states = []

    zero_positions = [i for i in range(len(state)) if state[i] == 0]

    for zero_pos in zero_positions:
        row, col = zero_pos // 3, zero_pos % 3

        possible_moves = []

        if row > 0:
            possible_moves.append(zero_pos - 3)

        if row < 2:
            possible_moves.append(zero_pos + 3)

        if col > 0:
            possible_moves.append(zero_pos - 1)

        if col < 2:
            possible_moves.append(zero_pos + 1)

        for move_pos in possible_moves:
            if state[move_pos] == 0:
                continue

            new_state = state.copy()
            new_state[zero_pos], new_state[move_pos] = new_state[move_pos], new_state[zero_pos]

            if new_state not in succ_states:
                succ_states.append(new_state)

    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0], heuristic=get_manhattan_distance):
    """
    Implement the A* algorithm to find the solution path.

    INPUT:
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along with h values,
        number of moves, and max queue number in the format specified.
    """
    state = list(state)

    goal_state = state_check(state)

    pq = []
    visited = set()
    state_lookup = {}
    max_len = 1

    initial_h = heuristic(state, goal_state)

    heapq.heappush(pq, (initial_h, tuple(state), (0, initial_h, None)))
    state_lookup[tuple(state)] = (None, 0, initial_h)

    solvable_condition = False

    while pq:
        max_len = max(max_len, len(pq))

        _, current_tuple, (g, h, parent) = heapq.heappop(pq)
        current = list(current_tuple)

        if current_tuple in visited:
            continue

        visited.add(current_tuple)

        if tuple(current) == tuple(goal_state):
            path = []
            curr = current_tuple
            while curr:
                moves = state_lookup[curr][1]
                h_value = state_lookup[curr][2]
                path.append((list(curr), h_value, moves))
                curr = state_lookup[curr][0]  # Get parent

            path.reverse()
            solvable_condition = True
            state_info_list = path
            break

        successors = get_succ(current)

        for successor in successors:
            successor_tuple = tuple(successor)

            # Skip if already visited
            if successor_tuple in visited:
                continue

            new_g = g + 1
            new_h = heuristic(successor, goal_state)
            new_f = new_g + new_h

            if successor_tuple not in state_lookup or new_g < state_lookup[successor_tuple][1]:
                heapq.heappush(pq, (new_f, successor_tuple, (new_g, new_h, current_tuple)))
                state_lookup[successor_tuple] = (current_tuple, new_g, new_h)

    if not solvable_condition:
        print("False")
        return
    else:
        print("True")

    for state_info in state_info_list:
        current_state = state_info[0]
        h = state_info[1]
        move = state_info[2]
        print(current_state, "h={}".format(h), "moves: {}".format(move))

    print("Max queue length: {}".format(max_len))

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    # print_succ([2,5,1,4,0,6,7,0,3])
    # print()
    #
    # print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()

    solve([2,5,1,4,0,6,7,0,3], heuristic=get_manhattan_distance)
    print()
