Let's continue with more complex optimizers: gradient descent.

This new optimizer in its new tab will be similar to "Local search", but instead of asking for a locality function, it will use the derivative function set on function settings.

It shall show the current point selected, and plot an arrow showing the direction of the gradient, and the next point suggested by the optimizer. The arrow must be proportional size with the learning rate.

In the settings the user can select first point, or use a random one.
Also, the user can select the learning rate.

After this, use same buttons as in "Local search": "1 step", "5 steps", "20 steps", "All steps".

If not derivative function is set, show a warning in the tab and disable the buttons.
