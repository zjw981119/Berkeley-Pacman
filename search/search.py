# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # initialize the initial-state node
    initial_node = (problem.getStartState(), [])

    # use stack and initialize frontier
    frontier = util.Stack()
    frontier.push(initial_node)

    # graph-search
    explored = []

    while not frontier.isEmpty():
        # curr_node[0] == state, curr_node[1] == [action]
        curr_node = frontier.pop()
        # only expand node that doesn't exist in explored set
        # to prune duplicated states
        if curr_node[0] not in explored:
            explored.append(curr_node[0])
            # find the goal state, return actions(path)
            if problem.isGoalState(curr_node[0]):
                return curr_node[1]
            # expand curr_node, getSuccessors(successor, action, stepCost)
            for next_state, action, stepCost in problem.getSuccessors(curr_node[0]):
                if next_state not in explored:
                    next_node = (next_state, curr_node[1] + [action])
                    # print(next_node)
                    frontier.push(next_node)

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # initialize the initial-state node
    initial_node = (problem.getStartState(), [])

    # use queue and initialize frontier
    frontier = util.Queue()
    frontier.push(initial_node)

    # graph-search
    explored = []

    while not frontier.isEmpty():
        # curr_node[0] == state, curr_node[1] == [action]
        curr_node = frontier.pop()
        # only expand node that doesn't exist in explored set
        # to prune duplicated states
        if curr_node[0] not in explored:
            # print("curr_node[0]: ", curr_node[0])
            explored.append(curr_node[0])
            # find the goal state, return actions(path)
            if problem.isGoalState(curr_node[0]):
                return curr_node[1]
            # expand curr_node, getSuccessors(successor, action, stepCost)
            for next_state, action, stepCost in problem.getSuccessors(curr_node[0]):
                if next_state not in explored:
                    next_node = (next_state, curr_node[1] + [action])
                    # print(next_node)
                    frontier.push(next_node)

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # initialize the initial-state node
    # node(position, path)
    initial_node = (problem.getStartState(), [])

    # use priority queue and initialize frontier
    frontier = util.PriorityQueue()
    frontier.push(initial_node, 0)

    # graph-search
    explored = []

    while not frontier.isEmpty():
        # curr_node[0] == state, curr_node[1] == [action], curr_node[2] == current_cost
        curr_node = frontier.pop()
        # find the goal state, return actions(path)
        if problem.isGoalState(curr_node[0]):
            return curr_node[1]
        # only expand node that doesn't exist in explored set
        # to prune duplicated states
        if curr_node[0] not in explored:
            explored.append(curr_node[0])
            # expand curr_node, getSuccessors(successor, action, stepCost)
            for next_state, action, stepCost in problem.getSuccessors(curr_node[0]):
                if next_state not in explored:
                    # getCostOfActions returns total cost of actions list
                    # calculate priority to nearby positions
                    priority = problem.getCostOfActions(curr_node[1]) + stepCost
                    next_node = (next_state, curr_node[1] + [action])
                    # print(next_node)
                    frontier.push(next_node, priority)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # initialize the initial-state node
    # node(position, path)
    initial_node = (problem.getStartState(), [])

    # use priority queue and initialize frontier
    frontier = util.PriorityQueue()
    # manhattanHeuristic(position, problem) returns the manhattan distance
    # heuristic(initial_node[0], problem)
    frontier.push(initial_node, 0)

    # graph-search
    explored = []

    while not frontier.isEmpty():
        # curr_node[0] == state, curr_node[1] == [action], curr_node[2] == current_cost
        curr_node = frontier.pop()
        # find the goal state, return actions(path)
        if problem.isGoalState(curr_node[0]):
            return curr_node[1]
        # only expand node that doesn't exist in explored set
        # to prune duplicated states
        if curr_node[0] not in explored:
            explored.append(curr_node[0])
            # expand curr_node, getSuccessors(successor, action, stepCost)
            for next_state, action, step_cost in problem.getSuccessors(curr_node[0]):
                if next_state not in explored:
                    # getCostOfActions returns total cost of actions list
                    # g(s) = curr_cost + step_cost
                    next_cost = problem.getCostOfActions(curr_node[1]) + step_cost

                    next_node = (next_state, curr_node[1] + [action])
                    # print(next_node)
                    # f(s) = g(s) + h(s), g(s) = next_cost
                    frontier.push(next_node, next_cost + heuristic(next_state, problem))

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
