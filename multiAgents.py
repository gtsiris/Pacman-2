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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        def ghostStateTogState(ghostState):  # Turn the given form of ghostState to the actual state of the ghost
            ghostStateStr = str(ghostState)  # Turn the given form into a string
            parts = ghostStateStr.split("=")  # Split the string to parts
            subparts = parts[1].replace("(", "").replace(")", "").split(", ")  # Remove the parentheses and split again
            gPosition = (float(subparts[0]), float(subparts[1]))  # Position of ghost
            gDirection = subparts[2]  # The last action that ghost did to get there
            gState = (gPosition, gDirection)  # This pair is the actual state of the ghost
            return gState

        from util import manhattanDistance  # ManhattanDistance is needed to calculate distances
        eval = successorGameState.getScore()  # The evaluation of the given state. Initially it is just the score
        # About food
        newFood = list(newFood)  # Turn the given form into a list (which is iterable)
        distanceToNearestFood = "Food not detected yet"  # Initialize the distance to nearest foor using this string
        for x in range(1, len(newFood) - 1):  # For each row of the map (except walls)
            for y in range(1, len(newFood[0]) - 1):  # For each column of the map (except walls)
                if newFood[x][y] == True:  # If there is food there
                    if manhattanDistance((x, y), newPos) < distanceToNearestFood or distanceToNearestFood == "Food not detected yet":
                        # If this food is closer to pacman than the nearest detected food
                        distanceToNearestFood = manhattanDistance((x, y), newPos)  # Update the distance of the nearest
        if distanceToNearestFood != "Food not detected yet":  # If there was atleast one detected food
            eval += 10 / distanceToNearestFood  # The distance of the nearest food has effect on the evaluation
        # About ghosts
        for ghostState in newGhostStates:  # For each ghost in the new situation
            gState = ghostStateTogState(ghostState)  # Find its actual state
            gPosition = gState[0]  # Find its position
            if manhattanDistance(gPosition, newPos) <= 1:  # If it is really close to pacman
                eval = -1000  # Force pacman to avoid it
        return eval

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
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose():  # If the game is already won or lost
                return self.evaluationFunction(gameState)  # Return the evaluation (default evalFn is score)
            elif depth == self.depth:  # If the requested depth is reached (default depth is 2)
                return self.evaluationFunction(gameState)  # Return the evaluation
            elif agentIndex == 0:  # If it is pacman, perform max
                max = "max not found yet"  # Initialize max using this string
                for action in gameState.getLegalActions(0):  # For each legal action from agent index 0 (aka pacman)
                    if minimax(gameState.generateSuccessor(0, action), depth, 1) > max or max == "max not found yet":
                    # If the minimax value of this action is greater than current max
                        max = minimax(gameState.generateSuccessor(0, action), depth, 1)  # Update max
                return max  # Return max
            elif agentIndex + 1 < gameState.getNumAgents():  # If the agent index belongs to a ghost (except the last one)
                min = "min not found yet"  # Initialize min using this string
                for action in gameState.getLegalActions(agentIndex):  # For each legal action from this ghost
                    if minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1) < min or min == "min not found yet":
                    # If the minimax value of this action is less than current min
                        min = minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)  # Update min
                return min  # Return min
            else:  # If the agent index belongs to the last ghost (according to indexing)
                depth += 1  # Increase the depth
                min = "min not found yet"  # Initialize min using this string
                for action in gameState.getLegalActions(agentIndex):  # For each legal action from this ghost
                    if minimax(gameState.generateSuccessor(agentIndex, action), depth, 0) < min or min == "min not found yet":
                    # If the minimax value of this action is less than current min
                        min = minimax(gameState.generateSuccessor(agentIndex, action), depth, 0)  # Update min
                return min  # Return min

        max = "max not found yet"  # Initialize max using this string
        for action in gameState.getLegalActions(0):  # For each legal action from agent index 0 (aka pacman)
            if minimax(gameState.generateSuccessor(0, action), 0, 1) > max or max == "max not found yet":
            # If the minimax value of this action is greater (therefore better for pacman)
                bestAction = action  # Save it as the best action so far
                max = minimax(gameState.generateSuccessor(0, action), 0, 1)  # Update max
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBetaPruning(gameState, depth, agentIndex, a, b):
            if gameState.isWin() or gameState.isLose():  # If the game is already won or lost
                return self.evaluationFunction(gameState)  # Return the evaluation (default evalFn is score)
            elif depth == self.depth:  # If the requested depth is reached (default depth is 2)
                return self.evaluationFunction(gameState)  # Return the evaluation
            elif agentIndex == 0:  # If it is pacman, perform max
                max = "max not found yet"  # Initialize max using this string
                for action in gameState.getLegalActions(0):  # For each legal action from agent index 0 (aka pacman)
                    if alphaBetaPruning(gameState.generateSuccessor(0, action), depth, 1, a, b) > max or max == "max not found yet":
                    # If the minimax value of this action is greater than current max
                        max = alphaBetaPruning(gameState.generateSuccessor(0, action), depth, 1, a, b)  # Update max
                    if max > b:  # If max is greater than beta
                        return max  # There is no need to go any farther (pruning)
                    elif max > a:  # If max is greater than alpha
                        a = max  # Update alpha
                return max  # Return max
            elif agentIndex + 1 < gameState.getNumAgents():  # If the agent index belongs to a ghost (except the last one)
                min = "min not found yet"  # Initialize min using this string
                for action in gameState.getLegalActions(agentIndex):  # For each legal action from this ghost
                    if alphaBetaPruning(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, a, b) < min or min == "min not found yet":
                    # If the minimax value of this action is less than current min
                        min = alphaBetaPruning(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, a, b)  # Update min
                    if min < a:  # If min is less than alpha
                        return min  # There is no need to go any farther (pruning)
                    elif min < b:  # If min is less than beta
                        b = min  # Update beta
                return min  # Return min
            else:  # If the agent index belongs to the last ghost (according to indexing)
                depth += 1  # Increase the depth
                min = "min not found yet"  # Initialize min using this string
                for action in gameState.getLegalActions(agentIndex):  # For each legal action from this ghost
                    if alphaBetaPruning(gameState.generateSuccessor(agentIndex, action), depth, 0, a, b) < min or min == "min not found yet":
                    # If the minimax value of this action is less than current min
                        min = alphaBetaPruning(gameState.generateSuccessor(agentIndex, action), depth, 0, a, b)  # Update min
                    if min < a:  # If min is less than alpha
                        return min  # There is no need to go any farther (pruning)
                    elif min < b:  # If min is less than beta
                        b = min  # Update beta
                return min  # Return min

        max = "max not found yet"  # Initialize max using this string
        a = float("-inf")  # Initialize alpha as the smallest value
        b = float("inf")  # Initialize beta as the greatest value
        for action in gameState.getLegalActions(0):  # For each legal action from agent index 0 (aka pacman)
            if alphaBetaPruning(gameState.generateSuccessor(0, action), 0, 1, a, b) > max or max == "max not found yet":
                # If the minimax value of this action is greater (therefore better for pacman)
                bestAction = action  # Save it as the best action so far
                max = alphaBetaPruning(gameState.generateSuccessor(0, action), 0, 1, a, b)  # Update max
            if max > b:  # If max is greater than beta
                return max  # There is no need to go any farther (pruning)
            elif max > a:  # If max is greater than alpha
                a = max  # Update alpha
        return bestAction

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

        def expectimax(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose():  # If the game is already won or lost
                return self.evaluationFunction(gameState)  # Return the evaluation (default evalFn is score)
            elif depth == self.depth:  # If the requested depth is reached (default depth is 2)
                return self.evaluationFunction(gameState)  # Return the evaluation
            elif agentIndex == 0:  # If it is pacman, perform max
                max = "max not found yet"  # Initialize max using this string
                for action in gameState.getLegalActions(0):  # For each legal action from agent index 0 (aka pacman)
                    if expectimax(gameState.generateSuccessor(0, action), depth, 1) > max or max == "max not found yet":
                        # If the expectimax value of this action is greater than current max
                        max = expectimax(gameState.generateSuccessor(0, action), depth, 1)  # Update max
                return max  # Return max
            elif agentIndex + 1 < gameState.getNumAgents():  # If the agent index belongs to a ghost (except the last one)
                chance = 0  # Initialize chance
                for action in gameState.getLegalActions(agentIndex):  # For each legal action from this ghost
                    chance += expectimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
                    # Accumulate chance using recursion
                return chance  # Return sum
            else:  # If the agent index belongs to the last ghost (according to indexing)
                depth += 1  # Increase the depth
                chance = 0  # Initialize sum
                for action in gameState.getLegalActions(agentIndex):  # For each legal action from this ghost
                    chance += expectimax(gameState.generateSuccessor(agentIndex, action), depth, 0)
                    # Accumulate chance using recursion
                return chance  # Return chance

        max = "max not found yet"  # Initialize max using this string
        for action in gameState.getLegalActions(0):  # For each legal action from agent index 0 (aka pacman)
            if expectimax(gameState.generateSuccessor(0, action), 0, 1) > max or max == "max not found yet":
                # If the expectimax value of this action is greater (therefore better for pacman)
                bestAction = action  # Save it as the best action so far
                max = expectimax(gameState.generateSuccessor(0, action), 0, 1)  # Update max
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Takes under consideration the score of the given game state, the distance to the nearest food and the
      total distance to the ghosts.
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    def ghostStateTogState(ghostState):  # Turn the given form of ghostState to the actual state of the ghost
        ghostStateStr = str(ghostState)  # Turn the given form into a string
        parts = ghostStateStr.split("=")  # Split the string to parts
        subparts = parts[1].replace("(", "").replace(")", "").split(", ")  # Remove the parentheses and split again
        gPosition = (float(subparts[0]), float(subparts[1]))  # Position of ghost
        gDirection = subparts[2]  # The last action that ghost did to get there
        gState = (gPosition, gDirection)  # This pair is the actual state of the ghost
        return gState

    from util import manhattanDistance  # ManhattanDistance is needed to calculate distances
    eval = currentGameState.getScore()  # The evaluation of the given state. Initially it is just the score
    # About food
    food = list(food)  # Turn the given form into a list (which is iterable)
    distanceToNearestFood = "Food not detected yet"  # Initialize the distance to nearest foor using this string
    for x in range(1, len(food) - 1):  # For each row of the map (except walls)
        for y in range(1, len(food[0]) - 1):  # For each column of the map (except walls)
            if food[x][y] == True:  # If there is food there
                if manhattanDistance((x, y), pos) < distanceToNearestFood or distanceToNearestFood == "Food not detected yet":
                    # If this food is closer to pacman than the nearest detected food
                    distanceToNearestFood = manhattanDistance((x, y), pos)  # Update the distance of the nearest
    if distanceToNearestFood != "Food not detected yet":  # If there was atleast one detected food
            eval += 1 / distanceToNearestFood  # The distance of the nearest food has effect on the evaluation
    # About ghosts
    totalDistanceToGhosts = 0  # The sum of the distance to each ghost
    for ghostState in ghostStates:  # For each ghost in the new situation
        gState = ghostStateTogState(ghostState)  # Find its actual state
        gPosition = gState[0]  # Find its position
        totalDistanceToGhosts += manhattanDistance(gPosition, pos)  # Accumulate the distances to ghosts
    if totalDistanceToGhosts != 0:  # If the total distance is not zero
        eval += 1 / totalDistanceToGhosts  # The total distance has effect on the evaluation
    return eval

# Abbreviation
better = betterEvaluationFunction

