# multiAgents.py
# --------------
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
import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        '''
        successor game state:  
        %%%%%%%%%%%%%%%%%%%%%%%%%
        %..              ....   %
        %..        G.  ...  ... %
        %          ..  ...  ... %
        %                ....   %
        %  v           ...  ... %
        %              ...  ... %
        %                ....  o%
        %%%%%%%%%%%%%%%%%%%%%%%%%
        '''
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates() # list of ghost states
        # getPosition() return the position tuple of an agent
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        """
        # evaluate capsule score
        capsuleList = successorGameState.getCapsules()
        capsuleScore = 0       
        closestCapsuleDist = sys.maxsize
        for capsule in capsuleList:
            if closestCapsuleDist > manhattanDistance(newPos, capsule):
                closestCapsuleDist = manhattanDistance(newPos, capsule)
        if closestCapsuleDist != sys.maxsize:
            if closestCapsuleDist == 0:
                capsuleScore = 10 # top priority to eat capsule
            else:
                capsuleScore = 1 / closestCapsuleDist
        """

        # evaluate basic food score
        newFoodList = newFood.asList()
        foodScore = 0
        closestFoodDist = sys.maxsize
        for foodPos in newFoodList:
            if closestFoodDist > manhattanDistance(newPos,foodPos):
                closestFoodDist = manhattanDistance(newPos,foodPos)
        if closestFoodDist != sys.maxsize:
            foodScore = 1 / closestFoodDist

        # evaluate action score
        actionScore = 0
        if action == Directions.STOP:
            actionScore = -sys.maxsize # avoid endless stop

        # evaluate ghost score
        ghostScore = 0
        for ghostState in newGhostStates:
            if(manhattanDistance(newPos,ghostState.getPosition()) == 0):
                ghostScore = -sys.maxsize # avoid encounter with ghost

        return foodScore + actionScore + ghostScore + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # check terminal state
        def terminalTest(gameState, curDepth):
            return curDepth == self.depth or gameState.isWin() or gameState.isLose()

        '''
        minimax decision: return the action with max value
        '''
        def minmaxDecision(gameState, curDepth):
            actionValList = []
            maxVal = -sys.maxsize
            bestAction = Directions.STOP #default value

            # evaluate each action of maximizer
            for action in gameState.getLegalActions(0):
                successorState = gameState.generateSuccessor(0, action)
                actionValList += [(action, minValue(successorState, curDepth, 1))]

            # select the action with max value
            for actionValPair in actionValList:
                # actionValPair = (action, val)
                if maxVal < actionValPair[1]:
                    maxVal = actionValPair[1]
                    bestAction = actionValPair[0]

            return bestAction

        # pacman agent: maximizer
        def maxValue(gameState, curDepth):
            # terminal test
            if terminalTest(gameState,curDepth):
                return self.evaluationFunction(gameState)

            value = -sys.maxsize # -inf
            for action in gameState.getLegalActions(0):
                # pass successor state to minimizer
                successorState = gameState.generateSuccessor(0, action)
                value = max(value, minValue(successorState, curDepth, 1))
            return value

        # ghost agents: minimizer (start from agent 1, may have multiple ghosts)
        def minValue(gameState, curDepth, agentindex):
            # terminal test
            if terminalTest(gameState,curDepth):
                return self.evaluationFunction(gameState)

            value = sys.maxsize  # inf
            for action in gameState.getLegalActions(agentindex):
                successorState = gameState.generateSuccessor(agentindex, action)
                if not agentindex == gameState.getNumAgents()-1:
                    # pass successor state to next ghost, same depth
                    value = min(value, minValue(successorState, curDepth, agentindex + 1))
                else:
                    # last ghost, pass successor state to maximizer, depth + 1
                    value = min(value, maxValue(successorState, curDepth + 1))

            return value

        return minmaxDecision(gameState, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        """
        "*** YOUR CODE HERE ***"

        # check terminal state
        def terminalTest(gameState, curDepth):
            return curDepth == self.depth or gameState.isWin() or gameState.isLose()

        '''
        minimax decision with alpha-beta pruning
        '''

        def minmaxDecision(gameState, intdepth):
            actionValList = []
            maxVal = -sys.maxsize
            bestAction = Directions.STOP  # default value

            alpha = -sys.maxsize
            beta = sys.maxsize
            # evaluate each action of maximizer
            for action in gameState.getLegalActions(0):
                successorState = gameState.generateSuccessor(0, action)
                value = minValue(successorState, intdepth, 1, alpha, beta)
                # pruning tree
                if value > beta:
                    return value
                # add (action, value) into list
                actionValList += [(action, value)]
                # update alpha
                alpha = max(alpha, value)

            # select the action with max value
            for actionValPair in actionValList:
                # actionValPair = (action, val)
                if maxVal < actionValPair[1]:
                    maxVal = actionValPair[1]
                    bestAction = actionValPair[0]

            return bestAction

        # pacman agent: maximizer
        def maxValue(gameState, curDepth, alpha, beta):
            # terminal test
            if terminalTest(gameState, curDepth):
                return self.evaluationFunction(gameState)

            value = -sys.maxsize  # -inf
            for action in gameState.getLegalActions(0):
                # pass successor state to minimizer
                successorState = gameState.generateSuccessor(0, action)
                value = max(value, minValue(successorState, curDepth, 1, alpha, beta))
                '''
                pruning tree, value == alpha can't be pruned 
                value == alpha, minimizer may choose either path, 
                but what if this maximizer's next state has greater value? 
                --> the minimizer won't choose this path any more, choice will change
                '''
                if value > beta:
                    return value
                # update alpha
                alpha = max(alpha, value)
            return value

        # ghost agents: minimizer (start from agent 1, may have multiple ghosts)
        def minValue(gameState, curDepth, agentindex, alpha, beta):
            # terminal test
            if terminalTest(gameState, curDepth):
                return self.evaluationFunction(gameState)

            value = sys.maxsize  # inf
            for action in gameState.getLegalActions(agentindex):
                successorState = gameState.generateSuccessor(agentindex, action)
                if not agentindex == gameState.getNumAgents() - 1:
                    # pass successor state to next ghost, same depth
                    value = min(value, minValue(successorState, curDepth, agentindex + 1, alpha, beta))
                else:
                    # last ghost, pass successor state to maximizer, depth + 1
                    value = min(value, maxValue(successorState, curDepth + 1, alpha, beta))
                '''
                pruning tree, value == alpha can't be pruned 
                value == alpha, maximizer may choose either path, 
                but what if this minimizer's next state has smaller value? 
                --> the maximizer won't choose this path any more, choice will change
                '''
                if value < alpha:
                    return value
                # update beta
                beta = min(beta, value)
            return value

        return minmaxDecision(gameState, 0)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # check terminal state
        def terminalTest(gameState, curDepth):
            return curDepth == self.depth or gameState.isWin() or gameState.isLose()

        '''
        expectiMax decision: basically same as minimax decision without pruning
        only need to change minValue function to expValue 
        '''

        def minmaxDecision(gameState, curDepth):
            actionValList = []
            maxVal = -sys.maxsize
            bestAction = Directions.STOP  # default value

            # evaluate each action of maximizer
            for action in gameState.getLegalActions(0):
                successorState = gameState.generateSuccessor(0, action)
                actionValList += [(action, expValue(successorState, curDepth, 1))]

            # select the action with max value
            for actionValPair in actionValList:
                # actionValPair = (action, val)
                if maxVal < actionValPair[1]:
                    maxVal = actionValPair[1]
                    bestAction = actionValPair[0]

            return bestAction

        # pacman agent: maximizer
        # remain same as minimaxAgent
        def maxValue(gameState, curDepth):
            # terminal test
            if terminalTest(gameState, curDepth):
                return self.evaluationFunction(gameState)

            value = -sys.maxsize  # -inf
            for action in gameState.getLegalActions(0):
                # pass successor state to minimizer
                successorState = gameState.generateSuccessor(0, action)
                value = max(value, expValue(successorState, curDepth, 1))
            return value

        # ghost agents: minimizer,(start from agent 1, may have multiple ghosts)
        # instead of choosing the minimum value, calculate the expect value(average value in this case)
        def expValue(gameState, curDepth, agentindex):
            # terminal test
            if terminalTest(gameState, curDepth):
                return self.evaluationFunction(gameState)

            avgValue = 0
            for action in gameState.getLegalActions(agentindex):

                successorState = gameState.generateSuccessor(agentindex, action)
                if not agentindex == gameState.getNumAgents() - 1:
                    # pass successor state to next ghost, same depth
                    value = expValue(successorState, curDepth, agentindex + 1)
                else:
                    # last ghost, pass successor state to maximizer, depth + 1
                    value = maxValue(successorState, curDepth + 1)

                # adversary which chooses among their getLegalActions uniformly at random.
                prob = 1 / len(gameState.getLegalActions(agentindex))
                avgValue += value * prob

            return avgValue

        return minmaxDecision(gameState, 0)

        # util.raiseNotDefined()

'''
    The evaluation function only evaluate states,
    rather than actions like the reflex agent evaluation function did
'''
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    evaluate scores based on minimum distance between pacman and food(including capsule),
    also takes consideration of eating ghost(after eating capsule), avoid ghost
    """

    "*** YOUR CODE HERE ***"

    curPos = currentGameState.getPacmanPosition()
    curFoodList = currentGameState.getFood().asList()
    curGhostStates = currentGameState.getGhostStates()  # list of ghost states
    # getPosition() return the position tuple of an agent

    # evaluate basic food score: find the closest food
    foodScore = 0
    closestFoodDist = sys.maxsize
    for foodPos in curFoodList:
        if closestFoodDist > manhattanDistance(curPos, foodPos):
            closestFoodDist = manhattanDistance(curPos, foodPos)
    if closestFoodDist != sys.maxsize:
        foodScore = 1 / closestFoodDist

    # evaluate capsule score: find the closest capsule
    capsuleList = currentGameState.getCapsules()
    capsuleScore = 0
    closestCapsuleDist = sys.maxsize
    for capsule in capsuleList:
        if closestCapsuleDist > manhattanDistance(curPos, capsule):
            closestCapsuleDist = manhattanDistance(curPos, capsule)
    if closestCapsuleDist != sys.maxsize:
        capsuleScore = 1 / closestCapsuleDist

    # evaluate ghost score
    ghostScore = 0
    for ghostState in curGhostStates:
        # ghost scared time
        # the closer the pac man is to the ghost, the higher the score
        if ghostState.scaredTimer:
            dis = manhattanDistance(curPos, ghostState.getPosition())
            ghostScore += 1 / dis
        else:
            # avoid encounter with ghost
            dis = manhattanDistance(curPos, ghostState.getPosition())
            if dis < 3:
                ghostScore -= 1
            else:
                ghostScore -= 0.2 # thrash around also gets penalty

    return foodScore + capsuleScore + ghostScore + currentGameState.getScore()


    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
