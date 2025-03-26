import cvxpy as cp
import numpy as np
#------------------------------------------------------------------------------------------------------------------#
# Matala 3 - Ex 3
# Id student: 328596978
#------------------------------------------------------------------------------------------------------------------#
def egalitarian_division(values):
    num_agents, num_resources = values.shape # number of agents and resources

    allocation = cp.Variable((num_agents, num_resources)) # resources div between the agents

    min_value = cp.Variable() # minimum of resources that each agent get

    constraints = [
        allocation >= 0,
        cp.sum(allocation, axis=0) == 1
    ]

    for i in range(num_agents):
        constraints.append(values[i] @ allocation[i] >= min_value)

    problem = cp.Problem(cp.Maximize(min_value), constraints) # egalitarian problem
    problem.solve()

# Print Results
    for i in range(num_agents):
        allocations = ", ".join(f"{allocation.value[i, j]:.2f} of resource #{j + 1}" for j in range(num_resources))
        print(f"Agent #{i + 1} gets {allocations}.")

#------------------------------------------------------------------------------------------------------------------#
# Example
values_matrix = np.array([
    [81, 19, 1],
    [70, 1, 29]
])
egalitarian_division(values_matrix)
# Solution
# Agent #1 gets 0.53 of resource #1, 1.00 of resource #2, 0.00 of resource #3.
# Agent #2 gets 0.47 of resource #1, 0.00 of resource #2, 1.00 of resource #3.
#------------------------------------------------------------------------------------------------------------------#
