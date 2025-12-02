# TSP Demo – "Bet on the Best Route"

This app is a small interactive demo of the **Traveling Salesman Problem (TSP)** and simple optimization methods.
You can think of it as a **betting game**: you “bet” on which route between the points is shorter, and the app tells you how good your bet was, and can also show you better routes.

---

## 1. The mathematical problem

We are given a set of points in the plane:

\[
\{A, B, C, \dots\}
\]

Each point has coordinates \((x_i, y_i)\) in the square \([0, 10] \times [0, 10]\).

A **path** is an ordered sequence of labels (for example, `ABCD`):

\[
A \to B \to C \to D
\]

The **length of a path** is the sum of the Euclidean distances between consecutive points:

\[
\text{length}(A \to B \to C \to D) = d(A,B) + d(B,C) + d(C,D)
\]

where

\[
d(P,Q) = \sqrt{(x_P - x_Q)^2 + (y_P - y_Q)^2}.
\]

In this app we consider a simplified TSP variant:

- We **start at A**.
- We want to **visit all other nodes once**.
- We **do not return to the start** (no cycle).

The objective is to find a **shortest possible route** (minimal total distance).

---

## 2. What the app lets you do

This app is designed so that you (or your students) can **experiment** with routes and see how good they are:

1. **Pick the number of points `N`** (between 2 and 26).
2. **Generate random coordinates** or manually edit the coordinates of each point.
3. **Type in a path** using the labels `A, B, C, ...`.
4. **See the total distance** of that path and whether it visits all nodes.
5. **Compare with:**
   - A **brute-force optimal path** (for small `N`, up to 10).
   - A **local search heuristic** that improves the path by swapping adjacent cities.

The main idea: you can **guess (bet) on a route**, then let the computer show you if there is a better one.

---

## 3. Interface & Controls

### 3.1. Sidebar – Parameters

- **Number of points (N)**
  Choose how many points you want: 2 ≤ N ≤ 26.
  Labels will be `A, B, C, …` up to the N-th letter.

- **Random seed**
  Controls how the random coordinates are generated.
  - The same seed + same N ⇒ same set of points.
  - Change the seed to get a different configuration.

- **Random points button**
  Generates a new random set of points using the current seed and N.

- **Coordinates editor**
  For each label (A, B, C, …), you can manually set:
  - `X_label` (x-coordinate in [0, 10])
  - `Y_label` (y-coordinate in [0, 10])

This lets you build your **own custom instances**.

---

### 3.2. Right column – Path & Distance

This section shows information about the path and the optimization tools.

- **Available points**
  A list like
  `A: (x_A, y_A)`
  `B: (x_B, y_B)`
  so you can see the exact coordinates.

- **Local search step time (seconds)**
  Controls the small delay between iterations of the automatic local search (for visualization / teaching).
  - 0.0 ⇒ fastest.
  - Larger value ⇒ slower, step-by-step feeling.

- **Buttons:**

  1. **"Find best path (brute force)"**
     - Only available for **N ≤ 10**.
     - Tries **all possible permutations** (starting at A and visiting each city once).
     - Guarantees the **globally optimal** path for that exact instance.
     - The path input is updated to this optimal route and the best distance is shown.

  2. **"Solve with local search (auto)"**
     - Starts from the current path.
     - Repeatedly applies local improvements (see below).
     - Stops when no improving adjacent swap is found (a **local optimum**).
     - Shows the current distance after each iteration.

  3. **"Local search – one step"**
     - Applies **one** local search step to the current path.
     - If there is an improving swap of two consecutive cities, it chooses the best such swap and updates the path.
     - If no swap improves the distance, it reports that a **local optimum** has been reached.

- **Path input** (`Path (e.g. ABCD)`)
  - Type any sequence of letters using the available labels.
  - Example: `ACBD`.
  - The app will:
    - Compute the total distance.
    - Check whether the path **visits all nodes at least once**.

- **Distance & validity indicator**
  - The distance is shown in large text:
    - Green if the path visits all nodes.
    - Red if some nodes are not visited.
  - A message indicates:
    - ✅ *“Path visits all nodes.”*
    - ❌ *“Path does NOT visit all nodes.”*

---

## 4. Local search: how it works

The **local search** operations ensure that the path is always interpreted as a **permutation** of the labels:

1. **Normalization**
   From whatever path you entered (e.g. `AACBDDA`), the app:
   - Keeps the **first occurrence** of each label in order.
   - Adds any missing labels at the end in natural order.

   Example with labels `[A, B, C, D]`:
   - Input: `AACBDDA`
   - First occurrences in order: `A C B D`
   - All labels present ⇒ permutation: `ACBD`.

2. **Neighborhood (adjacent swaps)**
   The neighbors of a path are all permutations obtained by **swapping two consecutive cities**, e.g.:

   - `ABCD` ⇒ neighbors: `BACD`, `ACBD`, `ABDC`.

3. **Improvement step**
   - The app evaluates all such **adjacent swaps**.
   - If one of them produces a **strictly shorter** path, it picks the best improving neighbor.
   - Otherwise, it declares a **local optimum** (no improving adjacent swap exists).

This is a simple example of a **local search heuristic**:
it often finds good solutions quickly, but **it does not guarantee the global optimum**.

---

## 5. Center column – Plot & Distance Matrix

### 5.1. Plot

- Shows the points labeled `A, B, C, …` in the plane.
- The current path (if valid) is drawn as a polyline connecting the points in the order of the path input.
- This allows you to **see the geometry** of your route:
  - Are you jumping back and forth?
  - Are there crossings that could be removed?

### 5.2. Distance matrix

Below the plot, the app displays the **distance matrix**:

- It is an N × N table.
- Entry `[i, j]` is the distance from point `i` to point `j`.
- Diagonal entries are 0.

This can be used:
- To verify the exact distances.
- As a **teaching tool** to connect the abstract matrix representation with the geometric picture.

---

## 6. Typical classroom / learning activities

Some ideas on how to use this app in a classroom or self-study:

1. **Betting on the route**
   - Students guess/“bet” on what they think is the best route.
   - They type it in and see the distance.
   - Then they run the local search or brute force and compare.

2. **Local versus global optimum**
   - Show examples where local search finds the global optimum.
   - Construct examples where local search gets stuck in a local optimum that is **not** global.

3. **Effect of problem size**
   - Show that brute force is feasible for small N (e.g. N = 6).
   - Show the warning for larger N and discuss **combinatorial explosion**.

4. **Tuning step time**
   - Increase the local search step time to watch the heuristic progress step by step.
   - Let students observe how the path evolves.

---

Enjoy exploring shortest routes and optimization methods!
