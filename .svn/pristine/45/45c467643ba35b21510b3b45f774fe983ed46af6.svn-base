# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def getDir(action):
    """Return Directions"""
    from game import Directions
    if action == 'North':
        return Directions.NORTH
    elif action == 'South':
        return Directions.SOUTH
    elif action == 'West':
        return Directions.WEST
    elif action == 'East':
        return Directions.EAST
    else:
        return ''

def checkMarked(visited,state):
    return state in visited

def depthFirstSearch(problem):
    
    _stack = util.Stack()
    actions = []
    visited = []
    _stack.push((problem.getStartState(), '', '0'))
    actions.append((problem.getStartState(), ''))
    
    """Loop for Searching"""
    while(1):
        if True == _stack.isEmpty():
            break
        else:
            currentState = _stack.pop()
            _stack.push(currentState)
            action = currentState[1]
            if currentState[0] != actions[-1][0]:
                actions.append((currentState[0] ,getDir(action)))
            
            if True == problem.isGoalState(currentState[0]):
                break
            else:
                successors = problem.getSuccessors(currentState[0])
                """Check if it's marked"""
                flag = 0
                for state in successors:
                    if False == checkMarked(visited,state):
                        _stack.push(state)
                        visited.append(state)
                        flag = 1
                if 0 == flag:
                    del actions[-1]
                    _stack.pop()
                    continue
    actionList = []
    for move in actions:
        if '' != move[1]:
            actionList.append(move[1])
    return actionList

def breadthFirstSearch(problem):
    _queue = util.Queue()
    visited = []
    startState = (problem.getStartState(), '', '0', [])
    _queue.push(startState)
    visited.append(startState)
    
    """Loop for Searching"""
    while(1):
        if True == _queue.isEmpty():
            break
        else:
            currentState = _queue.pop()
                    
            if True == problem.isGoalState(currentState[0]):
                return currentState[3]
            else:
                successors = problem.getSuccessors(currentState[0])
                for state in successors:
                    if False == checkMarked(visited, state[0]):
                        _tmp = list(currentState[3])
                        _tmp.append(getDir(state[1]))
                        _state = (state[0], state[1], state[2], _tmp)
                        if True == problem.isGoalState(state[0]):
                            return _tmp
                        _queue.push(_state)
                        visited.append(state[0])
    
    return []

def checkAdd(visited, _state):
    for oldState in visited:
        if oldState[0] == _state[0]:
            if _state[2] < oldState[1]:
                return True
            else:
                return False
    return True

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    _queue = util.PriorityQueue()
    visited = []
    startState = (problem.getStartState(), '', 0, [])
    _queue.push(startState, 0)
    visited.append([startState[0], 0])
    
    """Loop for Searching"""
    while(1):
        if True == _queue.isEmpty():
            break
        else:
            currentState = _queue.pop()
            flag = 0
            for _visit in visited:
                if _visit[0] == currentState[0]:
                    if currentState[2] < _visit[1]:
                        _visit[1] = currentState[2]
                    flag = 1
                    break
            if 0 == flag:
                visited.append([currentState[0], currentState[2]])
 
            if True == problem.isGoalState(currentState[0]):
                return currentState[3]
            else:
                successors = problem.getSuccessors(currentState[0])
                for state in successors:
                    _tmp = list(currentState[3])
                    _tmp.append(getDir(state[1]))
                    _state = (state[0], state[1], state[2] + currentState[2], _tmp)
                    if True == checkAdd(visited, _state):
                        """update or not"""
                        _queue.push(_state, _state[2])
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    _queue = util.PriorityQueue()
    visited = []
    startState = (problem.getStartState(), '', 0, [])
    _queue.push(startState, 0)
    visited.append([startState[0], 0])
    
    """Loop for Searching"""
    while(1):
        if True == _queue.isEmpty():
            break
        else:
            currentState = _queue.pop()
            flag = 0
            for _visit in visited:
                if _visit[0] == currentState[0]:
                    if currentState[2] < _visit[1]:
                        _visit[1] = currentState[2]
                    flag = 1
                    break
            if 0 == flag:
                visited.append([currentState[0], currentState[2]])
            
            if True == problem.isGoalState(currentState[0]):
                return currentState[3]
            else:
                successors = problem.getSuccessors(currentState[0])
                for state in successors:
                    _tmp = list(currentState[3])
                    _tmp.append(getDir(state[1]))
                    _state = (state[0], state[1], state[2] + currentState[2], _tmp)
                    if True == checkAdd(visited, _state):
                        """update or not"""
                        import searchAgents
                        """never met such strange thing, get int object"""
                        _queue.push(_state, _state[2] + heuristic(state[0], problem))
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
