Finally, let's implement the "Evolutive search" optimizer.
This optimizer will use a population of N points, and 3 different operators to generate new points: mutation, crossover and selection.
The points in the population must be marked in the plot to differentiate them.

# Mutation

The mutation will be a sigma value that will parametrized a normal distribution from each point, to generate randomly a new point. The user can select the sigma value in the settings.


# Crossover

The crossover will take 2 points from the population, and generate a new point that is a random weighted average of the 2 points, always being between both of them.
The user can select the temperature of the weighted values (lower the temperature, higher the probability to be close to the real average).

# Selection

The user can select between 3 methods:

- Best: select the N best points from the population and the new generated points.
- Roulette: select the points with a probability proportional to their utility value.
- Tournament: select 2 random points from the population and the new generated points, and select the best one. Repeat this process N times.

# Iteration

For each iteration, the program will mutate half of the population, and crossover between random pairs of points to generate N new points. Then, it will select the new population of N points based on the selection method selected.

Before each iteration, the program will show information regarding the next iteration:
- Show a circle of 1 sigma (66% probability) around each point that is going to be mutated.
- Show the pairs of points that are going to crossover, a line between them, and the new point generated.
- Show the points that are going to be selected, with a different color or shape.

# User interaction

The user can select the size of the population, the mutation sigma, the crossover temperature, and the selection method in the settings.

Include a "First population" button to generate randomly the first population of N points, and show them in the plot, along with the information of the following generation.

The user can click "Next generation" to calculate the new points and update the plot and the information accordingly.
Add also a "5 generations" button, and "10 generations" button.

# Evolution

Only if this tab is selected, show in the central panel the evolution of the population.
This is a plot with X axis the iteration, and Y axis the utility value. Each point in the plot is a point in the population, and it is colored with the same color as in the main plot.
Show also the best individual utility, and the mean utility of the population as a line that evolves with the iterations.
