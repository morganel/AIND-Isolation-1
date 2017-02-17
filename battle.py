"""
Estimate the strength rating of student-agent with iterative deepening and
a custom heuristic evaluation function against fixed-depth minimax and
alpha-beta search agents by running a round-robin tournament for the student
agent. Note that all agents are constructed from the student CustomPlayer
implementation, so any errors present in that class will affect the outcome
here.

The student agent plays a fixed number of "fair" matches against each test
agent. The matches are fair because the board is initialized randomly for both
players, and the players play each match twice -- switching the player order
between games. This helps to correct for imbalances in the game due to both
starting position and initiative.

For example, if the random moves chosen for initialization are (5, 2) and
(1, 3), then the first match will place agentA at (5, 2) as player 1 and
agentB at (1, 3) as player 2 then play to conclusion; the agents swap
initiative in the second match with agentB at (5, 2) as player 1 and agentA at
(1, 3) as player 2.
"""

import itertools
import random
import warnings
import math
import numpy as np

from collections import namedtuple

from isolation import Board
from sample_players import RandomPlayer
from sample_players import null_score
from sample_players import open_move_score
from sample_players import improved_score
from game_agent import CustomPlayer
from game_agent import custom_score

NUM_MATCHES = 315  # number of matches
TIME_LIMIT = 150  # number of milliseconds before timeout

TIMEOUT_WARNING = "One or more agents lost a match this round due to " + \
                  "timeout. The get_move() function must return before " + \
                  "time_left() reaches 0 ms. You will need to leave some " + \
                  "time for the function to return, and may need to " + \
                  "increase this margin to avoid timeouts during  " + \
                  "tournament play."

DESCRIPTION = """
This script evaluates the performance of the custom heuristic function by
comparing the strength of an agent using iterative deepening (ID) search with
alpha-beta pruning against the strength rating of agents using other heuristic
functions.  The `ID_Improved` agent provides a baseline by measuring the
performance of a basic agent using Iterative Deepening and the "improved"
heuristic (from lecture) on your hardware.  The `Student` agent then measures
the performance of Iterative Deepening and the custom heuristic against the
same opponents.
"""

Agent = namedtuple("Agent", ["player", "name"])

def generate_all_starting_positions():
    start = []
    for i1 in range(7):
        for j1 in range(7):
            for i2 in range(7):
                for j2 in range(7):
                    # check if a rotation of these is already in there
                    if (i1,j1) != (i2,j2):
                        insert = True
                        for iteration in range(8):
                            i1_rot, j1_rot = rot(i1, j1, iteration)
                            i2_rot, j2_rot = rot(i2, j2, iteration)
                            if [(i1_rot, j1_rot),(i2_rot, j2_rot)] in start:
                                insert = False
                                break
                        if insert:
                            if [(i2, j2), (i1, j1)] in start:
                                pass
                            else:
                                start.append([(i1, j1), (i2, j2)])
    return start

def rot(x,y,it):
    if it == 0: return (x,y) #same
    elif it == 1: return (6-y, x) #rotation 90 degrees
    elif it == 2: return (6-x, 6-y) #rotation 180 degrees
    elif it == 3: return (y, 6-x) #rotation 270 degrees
    elif it == 4: return (6-x, y) # vertical symmetry
    elif it == 5: return (x, 6-y) #  horizontal symmetry
    elif it == 6: return (y, x) # diagonal y=x
    elif it == 7: return (6-y, 6-x) # other diagonal 

def play_match(player1, player2, match_number):
    """
    Play a "fair" set of matches between two agents by playing two games
    between the players, forcing each agent to play from randomly selected
    positions. This should control for differences in outcome resulting from
    advantage due to starting position on the board.
    """

    num_wins = {player1: 0, player2: 0}
    num_timeouts = {player1: 0, player2: 0}
    num_invalid_moves = {player1: 0, player2: 0}
    games = [Board(player1, player2), Board(player2, player1)]

    for gi in range(2):
        move = match_number[gi]
        games[0].apply_move(move)
        games[1].apply_move(move)

    # play both games and tally the results
    for game in games:
        if hasattr(player1, 'cache'): 
            player1.cache = dict()
        if hasattr(player2, 'cache'): 
            player2.cache = dict()

        winner, move_history, termination = game.play(time_limit=TIME_LIMIT)
        if player1 == winner:
            num_wins[player1] += 1

            if termination == "timeout":
                num_timeouts[player2] += 1
            else:
                num_invalid_moves[player2] += 1

        elif player2 == winner:
            num_wins[player2] += 1

            if termination == "timeout":
                num_timeouts[player1] += 1
            else:
                num_invalid_moves[player1] += 1

    if sum(num_timeouts.values()) != 0:
        warnings.warn(TIMEOUT_WARNING)

    return num_wins[player1], num_wins[player2]

def play_round(agents, num_matches):
    """
    Play one round (i.e., a single match between each pair of opponents)
    """
    agent_1 = agents[-1]
    print(agent_1)
    wins = 0.
    total = 0.

    print("\nPlaying Matches:")
    print("----------")

    l = generate_all_starting_positions()
    print(len(l))
    for idx, agent_2 in enumerate(agents[:-1]):

        counts = {agent_1.player: 0., agent_2.player: 0.}
        names = [agent_1.name, agent_2.name]
        print("  Match {}: {!s:^11} vs {!s:^11}".format(idx + 1, *names), end=' ')

        p1 = agent_1.player
        p2 = agent_2.player
        for change_order in range(2): 
            for match_number in range(num_matches):

                if (match_number > 0 and match_number % 10 == 0):
                    print(counts[p1]/total)

                print(match_number)
                moves = l[match_number]
                if change_order == 1:
                    moves = moves[::-1]
                score_1, score_2 = play_match(p1, p2, moves)
                counts[p1] += score_1
                counts[p2] += score_2
                total += score_1 + score_2

        wins += counts[agent_1.player]

        print("\tResult: {} to {}".format(int(counts[agent_1.player]),
                                          int(counts[agent_2.player])))

    return 100. * wins / total

def main():
    CUSTOM_ARGS = {"method": 'alphabeta', 'iterative': True}

    test_agents = [Agent(CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS), "ID_Improved"),
                   Agent(CustomPlayer(score_fn=custom_score, **CUSTOM_ARGS), "Student")]

    print(test_agents[1].name)
    agents = [test_agents[0]] + [test_agents[1]]
    win_ratio = play_round(agents, NUM_MATCHES)

    print("\n\nResults:")
    print("----------")
    print("{!s:<15}{:>10.2f}%".format(test_agents[1].name, win_ratio))

if __name__ == "__main__":
    main()
