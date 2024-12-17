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
from util import*
import tkinter



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


import time


def depthFirstSearch(problem):
    from util import Stack
    import time

    # Start the timer
    start_time = time.time()

    # Stack for the frontier
    frontier = Stack()
    frontier.push((problem.getStartState(), [], 0))  # (state, path_to_state, cost)

    # Set to track visited nodes
    explored = set()

    # Counter for visited nodes
    visited_count = 0

    # Start processing the nodes
    while not frontier.isEmpty():
        current_state, path, cost = frontier.pop()

        # Increment the visited node counter
        visited_count += 1

        # Check if this is the goal state
        if problem.isGoalState(current_state):
            elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            print(f"DFS Execution Time: {elapsed_time:.2f} ms")
            print(f"Total Visited Nodes: {visited_count}")
            print(f"Path to Goal: {path}")
            return path

        # If not visited yet, explore it
        if current_state not in explored:
            explored.add(current_state)  # Mark the node as visited

            # Add all successors to the stack
            for next_state, action, step_cost in problem.getSuccessors(current_state):
                if next_state not in explored:
                    new_path = path + [action]  # Append the action to the current path
                    frontier.push((next_state, new_path, cost + step_cost))

    # If no solution is found
    elapsed_time = (time.time() - start_time) * 1000
    print(f"DFS Execution Time: {elapsed_time:.2f} ms")
    print(f"Total Visited Nodes: {visited_count}")
    print("No solution found.")
    return []

def breadthFirstSearch(problem):
    from util import Queue
    import time

    start_time = time.time()  # Start the timer

    frontier = Queue()
    frontier.push((problem.getStartState(), []))  # (state, path)
    explored = set()
    visited_count = 0  # To count visited nodes

    while not frontier.isEmpty():
        current_state, path = frontier.pop()
        visited_count += 1  # Increment visited nodes



        if problem.isGoalState(current_state):  # If goal is found, return the path
            elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            print(f"BFS Execution Time: {elapsed_time:.2f} ms")
            print(f"Total Visited Nodes: {visited_count}")
            print(f"Path to Goal: {path}")
            return path

        if current_state not in explored:
            explored.add(current_state)

            # Add successors to the frontier
            for next_state, action, _ in problem.getSuccessors(current_state):
                if next_state not in explored:
                    frontier.push((next_state, path + [action]))

    elapsed_time = (time.time() - start_time) * 1000
    print(f"BFS Execution Time: {elapsed_time:.2f} ms")
    print(f"Total Visited Nodes: {visited_count}")
    print("No solution found.")
    return []


def uniformCostSearch(problem):
    from util import PriorityQueue
    import time

    start_time = time.time()  # Start the timer

    frontier = PriorityQueue()
    frontier.push((problem.getStartState(), [], 0), 0)  # (state, path, cost)
    explored = set()
    costSoFar = {problem.getStartState(): 0}
    visited_count = 0  # To count visited nodes

    while not frontier.isEmpty():
        state, path, cost = frontier.pop()
        visited_count += 1  # Increment visited nodes

        if problem.isGoalState(state):  # If the goal is reached
            elapsed_time = (time.time() - start_time) * 1000
            print(f"UCS Execution Time: {elapsed_time:.2f} ms")
            print(f"Total Visited Nodes: {visited_count}")
            print(f"Path to Goal: {path}")
            return path

        if state not in explored:
            explored.add(state)

            # Process all successors
            for successor, action, step_cost in problem.getSuccessors(state):
                new_cost = cost + step_cost
                if successor not in costSoFar or new_cost < costSoFar[successor]:
                    costSoFar[successor] = new_cost
                    frontier.push((successor, path + [action], new_cost), new_cost)

    elapsed_time = (time.time() - start_time) * 1000
    print(f"UCS Execution Time: {elapsed_time:.2f} ms")
    print(f"Total Visited Nodes: {visited_count}")
    print("No solution found.")
    return []
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


    import heapq

def aStarSearch(problem, heuristic):
    from util import PriorityQueue
    import time

    # Start the timer
    start_time = time.time()

    # Priority queue for the frontier
    frontier = PriorityQueue()
    start = problem.getStartState()

    # Push the start state onto the frontier with priority = heuristic(start)
    frontier.push((start, [], 0), heuristic(start, problem))  # (state, path, cost)

    # Dictionary to store the best cost to reach each state
    costSoFar = {start: 0}  # Initialize start cost

    # Set to track explored nodes
    explored = set()

    while not frontier.isEmpty():
        # Pop the lowest-priority node from the frontier
        state, path, cost = frontier.pop()

        # If the goal is reached, return the path
        if problem.isGoalState(state):
            elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            print(f"A* Execution Time: {elapsed_time:.2f} ms")
            return path

        # Add the state to explored after popping
        if state not in explored:
            explored.add(state)

            # Expand the current state
            for successor, action, stepCost in problem.getSuccessors(state):
                newCost = cost + stepCost

                # Only process successor if it improves the cost
                if successor not in costSoFar or newCost < costSoFar[successor]:
                    costSoFar[successor] = newCost
                    priority = newCost + heuristic(successor, problem)
                    frontier.push((successor, path + [action], newCost), priority)

    # If no solution is found, return an empty path
    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"A* Execution Time: {elapsed_time:.2f} ms")
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
