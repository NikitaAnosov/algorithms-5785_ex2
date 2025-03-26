import cvxpy as cp
import numpy as np
#------------------------------------------------------------------------------------------------------------------#
# Matala 2 - Ex 3
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
# Examples
print("Example 1:")
values_matrix = np.array([
    [81, 19, 1],
    [70, 1, 29]
])
egalitarian_division(values_matrix)

print("Example 2:")
values_matrix2 = np.array([
    [50, 30, 20],
    [20, 40, 40],
    [30, 30, 40]
])
egalitarian_division(values_matrix2)

print("Example 3:")
values_matrix3 = np.array([
    [40, 20, 40, 0],
    [25, 25, 25, 25]
])
egalitarian_division(values_matrix3)
#------------------------------------------------------------------------------------------------------------------#
