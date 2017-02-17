"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def linear_combination_moves_score(game, player):
    """Calculate a linear combination of the moves available to each player.
    max_blanks represents the total number of cells on the board.
    At the beginning of the game, "max_blanks-blanks" is small and "blanks" is high. A higher 
    absolute coefficient is applied to the opponent's moves, which translates into a more aggressive style.

    At the end of the game, blanks is lower and max_blanks-blanks is high. A higher coefficient is applied to 
    the player's moves, which translates into a more defensive style.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    max_blanks = game.width**2
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    blanks = len(game.get_blank_spaces())

    return float((max_blanks-blanks)*own_moves - blanks*opp_moves) 


def linear_combination_moves_score_opposite(game, player):
    """Calculate a linear combination of the moves available to each player.
    max_blanks represents the total number of cells on the board.
    At the beginning of the game, "max_blanks-blanks" is small and "blanks" is high. A higher 
    absolute coefficient is applied to the opponent's moves, which translates into a more aggressive style.

    At the end of the game, blanks is lower and max_blanks-blanks is high. A higher coefficient is applied to 
    the player's moves, which translates into a more defensive style.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    max_blanks = game.width**2
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    blanks = len(game.get_blank_spaces())

    return float(blanks*own_moves - (max_blanks-blanks)*opp_moves) 

def cutoff_heuristics(game, player):
    """Changes the coefficients applied to player and its opponent based on where we are in the game.
    At the beginning of the game (blanks > 35), be more aggressive: own_moves - 3 * opp_moves
    At the end of the game, be more defensive: 3 * own_moves - opp_moves

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    blanks = len(game.get_blank_spaces())

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if blanks > 35: 
        return  float(own_moves - 3 * opp_moves)
    else:
        return float(3 * own_moves - opp_moves)

def moves_and_position(game, player):
    """
    Based on number of moves available and each player's distance to the center of the board.
    10 * (own_moves - opp_moves) + (own_dist_x + own_dist_y) - (opp_dist_x + opp_dist_y)

    Calculates the sum of the absolute number of squares from the player's position to the center of the board
    along the x and y axis.
    This heuristics puts a positive coefficient to the player's distance to center, pushing it towards the edges.
    """

    if game.is_winner(player):
        return float('inf')

    if game.is_loser(player):
        return float('-inf')
        
    center = game.width/2
    
    own_position = game.get_player_location(player)
    opp_position = game.get_player_location(game.get_opponent(player))

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    own_dist_x = abs(center - own_position[0])
    own_dist_y = abs(center - own_position[1])

    opp_dist_x = abs(center - opp_position[0])
    opp_dist_y = abs(center - opp_position[1])

    return float(10 * (own_moves - opp_moves) + (own_dist_x + own_dist_y) - (opp_dist_x + opp_dist_y))

def moves_and_position_opposite(game, player):
    """
    Based on number of moves available and each player's distance to the center of the board.
    10 * (own_moves - opp_moves) - (own_dist_x + own_dist_y) + (opp_dist_x + opp_dist_y)

    Calculates the sum of the absolute number of squares from the player's position to the center of the board
    along the x and y axis.
    This heuristics puts a negative coefficient to the player's distance to center, pushing it towards the center.
    """

    if game.is_winner(player):
        return float('inf')

    if game.is_loser(player):
        return float('-inf')
        
    center = game.width/2
    
    own_position = game.get_player_location(player)
    opp_position = game.get_player_location(game.get_opponent(player))

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    own_dist_x = abs(center - own_position[0])
    own_dist_y = abs(center - own_position[1])

    opp_dist_x = abs(center - opp_position[0])
    opp_dist_y = abs(center - opp_position[1])

    return float(10*(own_moves - opp_moves) - (own_dist_x + own_dist_y) + (opp_dist_x + opp_dist_y))
        

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    return moves_and_position(game, player)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_search_function(self):
        search_function = self.minimax
        if self.method == 'alphabeta':
            search_function = self.alphabeta
        if self.method == 'negamax':
            search_function = self.negamax
        return search_function
        


    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        m = (-1,-1)

        if len(legal_moves) == 0:
            return m

        search_function = self.get_search_function()

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            
            if not self.iterative:
                v, m = search_function(game, depth = self.search_depth)
            else:
                v, d = 0, 0
                while True: #abs(v) != float("Inf") not necessary since we can use all the time we have
                    v, m = search_function(game, depth = d)
                    d = d + 1

        except Timeout:
            # Handle any actions required at timeout, if necessary
            if (m == (-1, -1)):
                m = legal_moves[random.randint(0, len(legal_moves) - 1)]
        return m

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        current_depth : int
            current_depth is an integer representing the number of plies

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        bestMove = (-1,-1)

        if maximizing_player == 1:
            player_to_maximize = game.active_player
        else:
            player_to_maximize = game.inactive_player

        if (depth == 0):
            return self.score(game, player_to_maximize), bestMove

        if game.utility(player_to_maximize) != 0.0:
            return game.utility(player_to_maximize), bestMove   

        legalMoves = game.get_legal_moves(game.active_player)
        bestMove = legalMoves[0]

        if maximizing_player:
            bestValue = float("-Inf")

            for m in legalMoves:
                v, _ = self.minimax(game.forecast_move(m), depth - 1, False)
                if v > bestValue:
                    bestValue = v
                    bestMove = m
            return bestValue, bestMove

        else:
            bestValue = float("Inf")

            for m in legalMoves:
                v, _ = self.minimax(game.forecast_move(m), depth - 1, True)
                if v < bestValue:
                    bestValue = v
                    bestMove = m
            return bestValue, bestMove

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player = True):
        if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

        bestMove = (-1,-1)

        if maximizing_player == 1:
            player_to_maximize = game.active_player
        else:
            player_to_maximize = game.inactive_player

        if (depth == 0):
            return self.score(game, player_to_maximize), bestMove

        if game.utility(player_to_maximize) != 0.0:
            return game.utility(player_to_maximize), bestMove   

        legal_moves = game.get_legal_moves(game.active_player)
        bestMove = legal_moves[0]

        if maximizing_player:
            v = float("-Inf")
            
            for m in legal_moves:
                newv, _ = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, False)
                if newv > v:
                    v = newv
                    bestMove = m
                if v > alpha:
                    alpha = v
                    bestMove = m
                if beta <= alpha:
                    break # beta cut-off
            return v, bestMove

        else:
            v = float("Inf")
            for m in legal_moves:
                newv, _ = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, True)
                if newv < v:
                    v = newv
                    bestMove = m
                if v < beta:
                    beta = v
                    bestMove = m
                if beta <= alpha:
                    break # beta cut-off
            return v, bestMove
    
    def negamax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        current_depth : int
            current_depth is an integer representing the number of plies

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        bestMove = (-1,-1)

        if maximizing_player == 1:
            player_to_maximize = game.active_player
        else:
            player_to_maximize = game.inactive_player

        if (game.utility(player_to_maximize) != 0):
            return maximizing_player * game.utility(player_to_maximize), bestMove

        if (depth == 0):
            return maximizing_player * self.score(game, player_to_maximize), bestMove

        bestValue = float("-Inf")
        legalMoves = game.get_legal_moves(game.active_player)
        bestMove = legalMoves[0]
        for m in legalMoves:
            v, _ = self.negamax(game.forecast_move(m), depth - 1, -maximizing_player)
            v = -v
            if v > bestValue:
                bestValue = v
                bestMove = m

        return bestValue, bestMove
