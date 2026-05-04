Final feature is to develop the optimizers. These are algorithms that will suggest new points to calculate based on the points calculated so far.

Each optimizer will have a different tab inside the left column.
Each tab will have the buttons or elements to interact with the optimizer.
This is independent of the UI, so the UI can show points from different optimizers, or from different iterations of the same optimizer, etc.
Each tab will have a (i) icon with a tooltip to explain the optimizer and how to use it.

Start implementing the 2 basic ones:

# Manual
The user will be able to input the x1 and x2 values of the point to calculate. This is useful for testing the UI and for users who want to have full control over the points calculated.
Add number input fields and scroll bars to input the x1 and x2 values.

# Random
Just one button "New random point". When clicked, it will suggest a new random point to calculate. The random point should be generated within the x1 and x2 range defined in the settings, and always following the random seed.

# Local search
First, this optimizer will suggest an initial point to calculate, that will be set at random, but the user can modify it before starting the optimization.

Once an initial point is set, the optimizer will ask for a locality function `def locality(x1, x2) -> Iterator[Tuple[float, float]]` that will suggest new points to calculate based on the points calculated so far.
This will be written in a text field in python.
Give 2 pre-defined locality functions: manhattan distance with 4 cardinal points, and euclidean distance with 6 random points in 1 radious distance.
The user may be able to modify the locality function in text, but it shall not modify the pre-defined ones, so they can always be used as a reference or as a starting point.

This optimizer shall talk with the UI giving the current point as highlighted, the locality points as semi-highlighted, and connected with the current point.
When generating a new point, add a connection between the current point and the new point, and update the highlighted and semi-highlighted points accordingly.

Then, it will have these buttons:

- "1 step": when clicked, it will suggest the next point to calculate based on the locality function and the points calculated so far. The suggested point should be the one with the highest utility value among the points suggested by the locality function.
- "5 steps": like clicking the previous button 5 times.
- "20 steps": like clicking the previous button 20 times.
- "All steps": continue until reaching a local maximum, that is, until the suggested point is not better than the current point.
