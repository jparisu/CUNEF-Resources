Next feature implementation is to build the whole ui functionality.

The UI should have the following components:

- Plot: A 2D plotly plot showing the x1 x2 range. Inside the plot:
    - Scatter: The points that have already been calculated, with the color representing the utility value. The points should be interactive, showing the utility value on hover.
    - Contour: A contour plot of the function calculated in a grid of 100x100 (this value can be changed in the settings). The contour plot should be updated whenever the function is changed.
    - Height curves: similar to contour. These 2 may be deactivated by a checkbox
    - Colorbar: A colorbar showing the current range of utility values, with the corresponding colors. The current range is determined by the minimum and maximum utility values calculated so far. The colorbar should be updated whenever new points are calculated or when the function is changed.
    - Path: some points may be connected with a line to show connections (child points, neighbor points, etc.).
- Information: a section below the plot with some information
    - Best point: The best point so far, with x1 and x2 values and the corresponding utility value. If several points have the same utility value, show them all.
    - Number of points: The number of points that have been calculated so far.
    - Statistical information: mean, median, standard deviation of the utility values calculated so far.
- Special points: some points must be shown differently:
    - best point: the point with the highest utility value, should be highlighted in a different color and shape. If several points have the same utility value, show them all with the same highlight.
    - highlighted points: those points set by the optimizer (commonly last points)
    - semi-highlighted points: those points set by the optimizer with a lower priority (e.g. neighbor points, etc.)

The UI must allow the following buttons:

- Reset points: A button to reset all the calculated points and start from scratch. This should also reset the best point and the statistical information.
- Show/Hide contour: A checkbox to show or hide the contour plot.
- Show/Hide height curves: A checkbox to show or hide the height curves.
- Show/Hide colorbar: A checkbox to show or hide the colorbar.
- Show/Hide path: A checkbox to show or hide the path of the points calculated so far.
- Show/Hide special points: A checkbox to show or hide the special points (best point, highlighted points, semi-highlighted points).
