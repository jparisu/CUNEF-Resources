I want to build a python streamlit application for the students to try and practice with optimization functions. The idea is to have a function F with 2 arguments x1,x2 in a limited range [MIN, MAX], and try to find the best pair x1,x2 that maximizes F.

The project will have several parts:

## Function Handler
In this part, it is possible to write the function F in python, select an already created one, or generate one symbolically.
It will also configure parameters as MIN, MAX, random SEED of the whole project, whether the points must be integers or float, etc.
It is possible also to have automatically generated or manually written the derivative of the function.

This will appear in the top bar of the app, in a collapse section.

## Solutions Handler
In this part, it will store the solutions (points [x1,x2]) generated so far.
Each point will have associated the F value, and can be connected with previous points to visualize the path of the optimization process.
It will also provide the possibility to change the visualization of specific points, to highlight some of them.

This will appear in the main part of the app as a plot.
It will provide options to show/hide the actual area and elevation curves of the function, to show/hide lines between points, to reset the points, etc.

## Optimization Handler
In this part, there will be different implementations to handle semi-automatic optimization algorithms.
It will generate points based on the previous ones.
First approach will be a manual optimization, where the user will select the next point manually by selecting x1 and x2, or by clicking in a square area to set the next point.
But the implementation must follow an interface that allows to easily add new optimization algorithms, such as gradient ascent, local search, etc.

This will appear in a left sidebar of the app, in a collapse column.

I need the skeleton of this app. The files required, with the classes and functions without inner implementation, just the signatures and docstrings.
Think on the best way to design it, with modularity and separation of concerns in mind.
If you have any doubts about the behavior, the design, or the future implementations, ask me.
