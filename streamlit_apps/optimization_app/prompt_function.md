Next feature implementation is to build the whole function configuration pane.

It shall set the following variables:

- MIN: the minimum value for x1 x2 (default is -10)
- MAX: the maximum value for x1 x2 (default is 10)
- SEED: the random seed for generating the function (default is 0). This seed must affect all the project. Set a "reset seed" button to reset the whole project seed. This means, after resetting the seed, every random generation in the project should be affected by the new seed.
- INT: whether the function should allow only integers, or also floats for x1 x2 (default is False, meaning both integers and floats are allowed).

The function can be set by these 3 types:

- PREDEFINED: A set of pre-defined functions that users can choose from. For example, quadratic, sine/cosine, mount chair.
- CUSTOM: Users can input their own function in a text box in python syntax as `def utility(x1, x2): -> float`. The function should be validated to ensure it is a valid Python function and can be executed without errors or security risks.
- RANDOM: The system can generate a random function based on composition of simple mathematical operations (addition, multiplication, exponentiation) and basic functions (sine, cosine, exponential). The random function should be generated based on the current SEED value to ensure reproducibility, and have a parameters Nf to set the number of compositions to use (default 5, max 20).

It shall also be possible to set the derivative of the function. For predefined functions, the derivative must be pre-calculated. For custom and random functions, users can input the derivative in a text box in python syntax as `def derivative(x1, x2): -> (float, float)`, which should return the partial derivatives with respect to x1 and x2. This is not enforced.

The system should also show the function and derivative in mathematical notation using LaTeX rendering when possible (predefined functions for function and derivative, random functions only the function). For custom functions, if the user provides a LaTeX representation, it can be rendered as well.

Make the UI intuitive and not very populated, when one value is set, collapse the option so it reduces the amount of information shown. The latex representation must be visible only with this menu collapsed.
