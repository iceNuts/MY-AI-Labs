################################
# Create by lizeng @9:26 NOV21 
# Very stupid agent....
################################

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    
    toTakeAction = random.choice(bestActions)
        
    return toTakeAction

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  
  def getFeatures(self, gameState, action):
         
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
        
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    
    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    

    # Considering capsule
    capsuleList = gameState.getCapsules()
    if len(capsuleList) > 0:
        for a in capsuleList:
            x, y = a
            if (gameState.isOnRedTeam(self.index) and x > 16) or (not gameState.isOnRedTeam(self.index) and x < 16):
                foodList.append(a)
                if(myPos == a): features['successorScore'] += 40
      
    if len(foodList) > 0:
      dist_set = []
      dists = [self.getMazeDistance(myPos, a) for a in foodList]
      for a in foodList:
          if self.getMazeDistance(myPos, a) < 8: dist_set.append(a)
      if dists: distance = min(dists)
      if distance < 8:
          dists = [getRealDistance(myPos, a, gameState, 250) for a in dist_set]
          if dists: distance = min(dists)
      features['distanceToFood'] = distance
      
    # Compute distance to the nearest opponent
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    defenders = [a for a in enemies if not a.isPacman and not a.scaredTimer and a.getPosition() != None]
    
    if len(defenders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
      if dists: distance = min(dists)
      if distance < 4:
          dists = [getRealDistance(myPos, a.getPosition(), gameState, 100) for a in defenders]
          if dists: distance = min(dists)
     
      features['distanceToOpponent'] = distance
      
    return features

  def getWeights(self, gameState, action):
    # Compute distance to the nearest opponent
    return {'successorScore': 30, 'distanceToFood': -1, 'distanceToOpponent': 1}    

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
      
  def getFeatures(self, gameState, action):
    
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    
    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    nonInvaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders) 
    
    isNear = False
    if enemies:
        nearDistance = [getRealDistance(myPos, a.getPosition(), gameState, 100) for a in enemies if a.getPosition() != None]
        x, y = myPos
        if not ((gameState.isOnRedTeam(self.index) and x > 16) or (not gameState.isOnRedTeam(self.index) and x < 16)):
            for a in nearDistance:
                if a < 3:
                    isNear = True
            
    if len(invaders) > 0 or isNear:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      if dists: distance = min(dists)
      if distance < 10:
          dists = [getRealDistance(myPos, a.getPosition(), gameState, 300) for a in invaders]
          if dists: distance = min(dists)
          features['invaderDistance'] = distance
          return features
    else:
       return self.getOffensiveFeatures(gameState, action)
      
    features['invaderDistance'] = distance
    if(myState.scaredTimer):
        features['invaderDistance'] *= -1
    
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    
    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'invaderDistance': -30, 'stop': -100, 'reverse': -2, 'successorScore': 50, 'distanceToFood': -1, 'distanceToOpponent': 1}


  def getOffensiveFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
        
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    
    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    

    # Considering capsule
    capsuleList = gameState.getCapsules()
    if len(capsuleList) > 0:
        for a in capsuleList:
            x, y = a
            if (gameState.isOnRedTeam(self.index) and x > 16) or (not gameState.isOnRedTeam(self.index) and x < 16):
                foodList.append(a)
                if(myPos == a): features['successorScore'] += 1000
      
    if len(foodList) > 0:
      dist_set = []
      dists = [self.getMazeDistance(myPos, a) for a in foodList]
      for a in foodList:
          if self.getMazeDistance(myPos, a) < 12: dist_set.append(a)
      if dists: distance = min(dists)
      if distance < 12:
          dists = [getRealDistance(myPos, a, gameState, 300) for a in dist_set]
          if dists: distance = min(dists)
      features['distanceToFood'] = distance
      
    # Compute distance to the nearest opponent
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    defenders = [a for a in enemies if not a.isPacman and not a.scaredTimer and a.getPosition() != None]
    
    if len(defenders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
      if dists: distance = min(dists)
      if distance < 4:
          dists = [getRealDistance(myPos, a.getPosition(), gameState, 100) for a in defenders]
          if dists: distance = min(dists)
     
      features['distanceToOpponent'] = distance
      
    return features
    
    
#Get real distance from location A to location B
     
def getRealDistance(pos1, pos2, gameState, nodeNumber):
    walls = gameState.getWalls().asList()
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    maxNodes = nodeNumber
    queue = list()
    visited = list()
    queue.append((pos1, 0))
    visited.append(pos1)
    while(1):
        if len(visited) > maxNodes: break
        position, distance = queue[0]
        if position == pos2: return distance
        visited.append(position)
        queue.pop(0)
        #generate successors
        x, y = position
        for mX, mY in directions:
            _x = x + mX
            _y = y + mY
            if not (_x, _y) in walls and not (_x, _y) in visited:
                queue.append(((_x, _y), distance + 1))
    x0, y0 = pos1
    x1, y1 = pos2
    return  abs(x0 - x1) + abs(y0 - y1)      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    