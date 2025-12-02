# Betting Strategy Visualizer – Help & Explanation

## 1. The betting problem

We imagine a random integer \( R \) chosen **uniformly at random** from the set:

\[
R \in \{1, 2, \dots, S\}.
\]

The system gives several *bets* on different positions \( b_i \) between 1 and \( S \).
For each bet you (the player) can choose an **action**:

- **high**
- **low**
- **none**

Each bet produces a number of points depending on the realized value of \( R \).

### Payoff from a single bet

For a given bet with position \( b \) and action:

- **high**:
  \[
  \text{points} = R - b
  \]
- **low**:
  \[
  \text{points} = b - R
  \]
- **none**:
  \[
  \text{points} = 0
  \]

Notice that the points can be **positive or negative**:

- A **high** bet at position \( b \) is good when \( R \) is **larger** than \( b \) (you gain points),
  and bad when \( R \) is smaller than \( b \) (you lose points).
- A **low** bet at position \( b \) is good when \( R \) is **smaller** than \( b \),
  and bad when \( R \) is larger than \( b \).

You can place multiple bets (each with its own \( b_i \) and action).
The **total number of points** for a given value of \( R \) is the sum of the points of all bets.

---

## 2. What the app shows

The app helps you explore how your choice of bets affects:

- the **total points** you get for each possible value of \( R \),
- the **expected value** (mean points over all \( R = 1,\dots,S \)),
- the **maximum** and **minimum** total points you could obtain.

It is meant as a small interactive tool for intuition and teaching.

---

## 3. Layout and controls

The app has three main areas:

1. **Right column – Parameters and bets**
2. **Left column – Summary and a specific \( R \)**
3. **Center column – Plots**

### 3.1 Right column – Parameters

Here you choose the overall settings:

- **Size of range \( S \)**
  Slider: choose the maximum value of \( R \).
  The random number will be uniformly selected from \( \{1, 2, \dots, S\} \).

- **Number of bets**
  How many bets you want to place. For each bet you will define a position and an action.

For each bet (Bet 1, Bet 2, …):

- **Position \( b_i \)**
  Slider from 1 to \( S \): where you “place” this bet on the number line.

- **Action**
  Radio buttons with three options:
  - `high`
  - `low`
  - `none`

Each bet is displayed with its own **color**, and this same color is used in the plots.

---

### 3.2 Left column – Outcome summary and specific \( R \)

This column has two main parts.

#### a) Outcome summary (over all possible \( R \))

It shows:

- **Mean points (expected value)**
  The average of the total points over all \( R = 1, \dots, S \).
  This is a simple, uniform expectation.

- **Maximum points over \( R \)**
  The largest total points you can get for some value of \( R \).

- **Minimum points over \( R \)**
  The smallest total points (worst-case outcome).

These numbers update automatically when you change the bets or \( S \).

#### b) Try a specific value of \( R \)

You can:

- Use **“Pick R randomly”** to choose a random value of \( R \) in \( 1,\dots,S \).
- Use the **slider** “Current selected value R” to choose \( R \) manually.

For that chosen \( R \), you see:

- **“Total points if R = …”** – the sum of the contributions of all bets.

You can also expand **“Show contribution of each bet for this R”** to see, for each bet:

- the individual points contributed by that bet,
- with the same color used in the plots.

---

### 3.3 Center column – Visualizations

There are three plots.

#### Plot 1 – Bet positions along the line \( 1..S \)

- Shows the number line from 1 to \( S \).
- Each bet appears as a colored marker at its position \( b_i \),
  labeled \( b_1, b_2, \dots \).
- This helps visualize where your bets sit relative to each other.

#### Plot 2 – Points from each bet as \( R \) varies

- One **line per bet**, same color as the bet.
- Horizontal axis: \( R = 1,\dots,S \).
- Vertical axis: **points from that single bet**.
- This shows how the payoff of each individual bet changes as \( R \) moves.

#### Plot 3 – Total points for each possible \( R \)

- A bar chart of **total points** (sum of all bets) for each \( R \).
- The bar corresponding to the currently selected \( R \) is highlighted.
- This lets you see the overall shape of your strategy:
  where it does well and where it does poorly.

---

## 4. How to use the app step by step

1. **Choose \( S \)**
   Use the slider to set the range \( \{1,\dots,S\} \) for the random number \( R \).

2. **Choose the number of bets**
   Decide how many bets you want to place.

3. **Configure each bet**
   For each bet:
   - Set the position \( b_i \) (an integer between 1 and \( S \)).
   - Choose the action: `high`, `low`, or `none`.

4. **Inspect the summary and graphs**
   Observe:
   - The expected value, maximum, and minimum total points.
   - How each individual bet’s payoff changes with \( R \).
   - The total points bar chart.

5. **Experiment**
   Vary the bets and see how:
   - The expected value changes,
   - The worst-case and best-case totals change,
   - The bar chart shape of total points changes.

6. **Check specific values of \( R \)**
   Pick some \( R \) (randomly or via the slider) and:
   - See the total points at that exact value of \( R \),
   - See how much each bet contributes.

---

## 5. Typical uses

- **Teaching tool**
  For illustrating randomness, expected value, and how linear payoffs can produce gains and losses.

- **Exploring intuition**
  Try different configurations of high/low/none bets and see how they shift the distribution of total points.

- **Experimenting with strategies**
  For example, try to design bets that:
  - maximize the expected value,
  - or reduce the worst-case loss,
  - or create a specific shape of total payoff across \( R \).

Enjoy exploring!
