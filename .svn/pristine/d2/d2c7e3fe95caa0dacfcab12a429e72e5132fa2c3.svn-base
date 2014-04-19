# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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

    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood().asList() #Generate coordinates with asList()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
      
    "*** YOUR CODE HERE ***"
    from random import randint
    import math
    
    inf = 9999
    
    walls = currentGameState.getWalls().asList()
    oldFood = currentGameState.getFood().asList()
    ghostNumber = len(newGhostStates)
    
    dist_food = inf
    dist_ghost = inf
    
    for i in range(ghostNumber):
        if(newScaredTimes[i] != 0): continue;
        manhattan = util.manhattanDistance(newPos, newGhostStates[i].getPosition())
        dist_ghost = min(dist_ghost, manhattan)
    for food in newFood:
        manhattan = util.manhattanDistance(newPos, food)
        dist_food = min(dist_food, manhattan)
        
    if(dist_food == 0): dist_food = 1;
        
    if(dist_ghost < 3): return dist_ghost
    
    #print dist_food
    
    return successorGameState.getScore() + 10/float(dist_food);#might encounter tie


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
  
  def do_minimax(self, gameState, agent, agentValue, depth):
    
    inf = 999999
    legalMoves = gameState.getLegalActions(agent);
    if(Directions.STOP in legalMoves): legalMoves.remove(Directions.STOP)
    agentNumber = gameState.getNumAgents()
    #init
    if(gameState not in agentValue[depth]):
        if(agent == 0): agentValue[depth][gameState] = -inf
        else:   agentValue[depth][gameState] = inf

    for action in legalMoves:
        successorState = gameState.generateSuccessor(agent, action)
        if(agent + 1 >= agentNumber):
            #meet the bottom
            if(depth + 1 >= self.depth):
                agentValue[depth][successorState] = self.evaluationFunction(successorState)
                return agentValue[depth][successorState]
            else:
                ret = self.do_minimax(successorState, 0, agentValue, depth + 1)
        else:
                ret = self.do_minimax(successorState, agent + 1, agentValue, depth)
        if(ret == inf or ret == -inf): continue;
        if(agent != 0 and agentValue[depth][gameState] > ret) or (agent == 0 and agentValue[depth][gameState] < ret):
            agentValue[depth][gameState] = ret;
    
    if(agentValue[depth][gameState] == inf or agentValue[depth][gameState] == -inf):
    #no legal move
        agentValue[depth][gameState] = self.evaluationFunction(gameState);
        return agentValue[depth][gameState];
    return agentValue[depth][gameState]


  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    
    agentValue = {}
    for i in range(self.depth):
        agentValue[i] = {}
    
    legalMoves = gameState.getLegalActions(0);
    if(Directions.STOP in legalMoves): legalMoves.remove(Directions.STOP)
    result = self.do_minimax(gameState, 0, agentValue, 0)
    #print result
    #get action from result
    for action in legalMoves:
        successorState = gameState.generateSuccessor(0, action)
        if(agentValue[0][successorState] == result):
            return action
    return Directions.STOP

#Prune Alpha-Beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  
  def do_minimax_prune(self, gameState, agent, agentValue, alpha, beta, depth):
    
    inf = 999999
    legalMoves = gameState.getLegalActions(agent);
    if(Directions.STOP in legalMoves): legalMoves.remove(Directions.STOP)
    agentNumber = gameState.getNumAgents()
    #init
    if(gameState not in agentValue[depth]):
        if(agent == 0): agentValue[depth][gameState] = -inf
        else:   agentValue[depth][gameState] = inf
    try:
        tmp_alpha[0] = alpha[0];
        tmp_beta[0] = beta[0];
    except NameError:
        tmp_alpha = [alpha[0]]
        tmp_beta = [beta[0]]
    
    for action in legalMoves:
        if(agent != 0 and tmp_beta[0] <= alpha[0]): break;
        if(agent == 0 and tmp_alpha[0] >= beta[0]): break;
        successorState = gameState.generateSuccessor(agent, action)
        if(agent + 1 >= agentNumber):
            #meet the bottom
            if(depth + 1 >= self.depth):
                agentValue[depth][successorState] = self.evaluationFunction(successorState)
                return agentValue[depth][successorState]
            else:
                ret = self.do_minimax_prune(successorState, 0, agentValue, tmp_alpha, tmp_beta, depth + 1)
        else:
                ret = self.do_minimax_prune(successorState, agent + 1, agentValue, tmp_alpha, tmp_beta,depth)
        if(ret == inf or ret == -inf): continue;
        if(agent != 0 and agentValue[depth][gameState] > ret) or (agent == 0 and agentValue[depth][gameState] < ret):
            if(agent == 0): tmp_alpha[0] = ret;
            if(agent != 0): tmp_beta[0] = ret;
            agentValue[depth][gameState] = ret;
    #update alpha beta
    if(agent == 0) and (beta[0] > tmp_beta[0]): beta[0] = tmp_beta[0]
    if(agent != 0) and (alpha[0] < tmp_alpha[0]): alpha[0] = tmp_alpha[0]
    
    if(agentValue[depth][gameState] == inf or agentValue[depth][gameState] == -inf):
    #no legal move
        agentValue[depth][gameState] = self.evaluationFunction(gameState);
        return agentValue[depth][gameState];
    return agentValue[depth][gameState]

        
    return agentValue[depth][gameState]

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    inf = 999999

    agentValue = {}
    for i in range(self.depth):
        agentValue[i] = {}

    alpha = [-inf];
    beta = [inf];
    
    legalMoves = gameState.getLegalActions(0);
    if(Directions.STOP in legalMoves): legalMoves.remove(Directions.STOP)
    result = self.do_minimax_prune(gameState, 0, agentValue, alpha, beta, 0)
    #print result
    #get action from result
    for action in legalMoves:
        successorState = gameState.generateSuccessor(0, action)
        if(agentValue[0][successorState] == result):
            return action
    return Directions.STOP

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  
  def do_minimax(self, gameState, agent, agentValue, depth):
    
    inf = 999999
    legalMoves = gameState.getLegalActions(agent);
    if(Directions.STOP in legalMoves): legalMoves.remove(Directions.STOP)
    agentNumber = gameState.getNumAgents()
    #init
    if(gameState not in agentValue[depth]):
        if(agent == 0): agentValue[depth][gameState] = -inf
        else:   agentValue[depth][gameState] = inf
    
    for action in legalMoves:
        successorState = gameState.generateSuccessor(agent, action)
        if(agent + 1 >= agentNumber):
            #meet the bottom
            if(depth + 1 >= self.depth):
                agentValue[depth][successorState] = self.evaluationFunction(successorState)
                return agentValue[depth][successorState]
            else:
                ret = self.do_minimax(successorState, 0, agentValue, depth + 1)
        else:
                ret = self.do_minimax(successorState, agent + 1, agentValue, depth)
        #for exceptions
        if(ret == inf or ret == -inf): continue;
        if(agent == 0 and agentValue[depth][gameState] < ret):
            agentValue[depth][gameState] = ret
        elif(agent != 0):
            if(agentValue[depth][gameState] == inf):
                agentValue[depth][gameState] = 0;
            agentValue[depth][gameState] += ret*(1/float(len(legalMoves)));
    
    if(agentValue[depth][gameState] == inf or agentValue[depth][gameState] == -inf):
    #no legal move
        agentValue[depth][gameState] = self.evaluationFunction(gameState);
        return agentValue[depth][gameState];
    return agentValue[depth][gameState]

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    agentValue = {}
    for i in range(self.depth):
        agentValue[i] = {}
    
    legalMoves = gameState.getLegalActions(0);
    if(Directions.STOP in legalMoves): legalMoves.remove(Directions.STOP)
    result = self.do_minimax(gameState, 0, agentValue, 0)
    #print result
    #get action from result
    for action in legalMoves:
        successorState = gameState.generateSuccessor(0, action)
        if(agentValue[0][successorState] == result):
            return action
    return Directions.STOP


def food_bfs(currentGameState, pacmanPosition, limit):
    queue = []
    distCnt = [0 for i in range(limit + 1)]
    valueCnt = [0 for i in range(limit + 1)]
    hitBoard = []

    value = 0
    depth = 0
    startState = [pacmanPosition, value, depth]

    hitBoard.append(startState[0])
    queue.append(startState)
    
    foods = currentGameState.getFood()
    walls = currentGameState.getWalls()

    dirX = [1, -1, 0, 0]
    dirY = [0 , 0,-1, 1]

    while(1):
        if(len(queue) == 0): break;

        state = queue[0]
        queue.pop(0)
        distCnt[state[2]] += 1
        valueCnt[state[2]] += state[1]

        #generate successors
        for i in range(4):
            x, y = state[0]
            depth = state[2] + 1
            if(depth >= limit): break
            
            x0 = x + dirX[i]
            y0 = y + dirY[i]
            if(walls[x0][y0] == True): continue;
            if(foods[x0][y0] == True):
                value = state[1] + 20
            else:
                value -= 1
            if((x0, y0) in hitBoard): continue
            state0 = [(x0, y0), value, depth]
            hitBoard.append((x0, y0))
            queue.append(state0)

    expectation = 0
    for i in range(limit):
        probability = 1
        for j in range(i):
            if(distCnt[j] == 0): continue
            probability *= 1/float(distCnt[j])
        value = valueCnt[i]
        expectation += probability*value
        #print value, probability

    return expectation


#dfs ghost_bfs(self, currentGameState, pacmanPosition, ghostPosition, limit):
    
a = 1.0
b = 0.1
c = 15
d = 10
e = 10
f = 50

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #Useful information you can extract from a GameState (pacman.py)
    ghostStates = currentGameState.getGhostStates()
    foodStates = currentGameState.getFood().asList()
    ghostScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    walls = currentGameState.getWalls().asList()
    pacmanState = currentGameState.getPacmanState()

    currentScore = currentGameState.getScore()

    #compute ghost manhattan
    inf = 999999
    ghostManhattan = inf
    pacmanPosition = pacmanState.getPosition()
    nearestGhostPosition = None
    
    for ghost in ghostStates:
        ghostPosition = ghost.getPosition()
        manhattan = util.manhattanDistance(ghostPosition, pacmanPosition)
        index = ghostStates.index(ghost)
        if(ghostScaredTimes[index] == 0 and manhattan < ghostManhattan):
            ghostManhattan = manhattan
            nearestGhostPosition = ghostPosition

    ghostScore = 0
    foodScore = 0
    
    global a,b,c,d,e

    if(ghostManhattan >= d):
        ghostScore = d * 10
    else:
        if(ghostManhattan == inf): ghostManhattan = f
        ghostScore = ghostManhattan*e
        #cal ghostScore
        #ghostScore = self.ghost_bfs(currentGameState, pacmanPosition, ghostPosition, 10)
    
    #cal foodScore

    foodScore = food_bfs(currentGameState, pacmanPosition, c)
    return a*ghostScore + b*foodScore + currentScore
        

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """
  
  def do_minimax(self, gameState, agent, agentValue, depth):
    global a,b,c,d
    a = 1.0
    b = 0.001
    c = 15
    d = 15
    e = 15 #manhattan sometimes underestimates
    f = 100
    inf = 999999
    legalMoves = gameState.getLegalActions(agent);
    if(Directions.STOP in legalMoves): legalMoves.remove(Directions.STOP)
    agentNumber = gameState.getNumAgents()
    #init
    if(gameState not in agentValue[depth]):
        if(agent == 0): agentValue[depth][gameState] = -inf
        else:   agentValue[depth][gameState] = inf
    
    for action in legalMoves:
        successorState = gameState.generateSuccessor(agent, action)
        if(agent + 1 >= agentNumber):
            #meet the bottom
            if(depth + 1 >= self.depth):
                agentValue[depth][successorState] = betterEvaluationFunction(successorState)
                return agentValue[depth][successorState]
            else:
                ret = self.do_minimax(successorState, 0, agentValue, depth + 1)
        else:
                ret = self.do_minimax(successorState, agent + 1, agentValue, depth)
        #for exceptions
        if(ret == inf or ret == -inf): continue;
        if(agent == 0 and agentValue[depth][gameState] < ret):
            agentValue[depth][gameState] = ret
        elif(agent != 0):
            if(agentValue[depth][gameState] == inf):
                agentValue[depth][gameState] = 0;
            agentValue[depth][gameState] += ret*(1/float(len(legalMoves)));
    
    if(agentValue[depth][gameState] == inf or agentValue[depth][gameState] == -inf):
    #no legal move
        agentValue[depth][gameState] = betterEvaluationFunction(gameState);
        return agentValue[depth][gameState];
    return agentValue[depth][gameState]

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usuallym
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    agentValue = {}
    for i in range(self.depth):
        agentValue[i] = {}
    
    legalMoves = gameState.getLegalActions(0);
    if(Directions.STOP in legalMoves): legalMoves.remove(Directions.STOP)
    result = self.do_minimax(gameState, 0, agentValue, 0)
    #print result
    #get action from result
    for action in legalMoves:
        successorState = gameState.generateSuccessor(0, action)
        if(agentValue[0][successorState] == result):
            return action
    return Directions.STOP

