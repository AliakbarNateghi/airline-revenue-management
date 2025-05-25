"""
REVENUE MAXIMIZATION MODEL FOR AIRLINE TICKET SALES
===================================================

PROBLEM DESCRIPTION:
This code implements a revenue maximization optimization model for airline ticket sales.
The goal is to determine the optimal number of tickets to sell each day across two fare
classes ($100 and $200 tickets) to maximize total revenue over a 90-day period.

APPROACH: Direct Revenue Maximization
- NOT using EMSR (Expected Marginal Seat Revenue) methods
- Using constraint programming to directly maximize: Total Revenue = Σ(Price × Quantity)
- Subject to daily demand limits and total capacity constraints
"""

import numpy as np
import matplotlib.pyplot as plt
# Google OR-Tools: Open-source optimization library from Google
# Used by major tech companies and airlines for constraint programming
# !pip install ortools
from ortools.sat.python import cp_model

# ============================================================================
# PROBLEM PARAMETERS
# ============================================================================

# TIME HORIZON: 90-day booking window
# Range from day 1 to day 90 (inclusive)
# This represents the advance booking period for the flight/route
days = list(range(1, 901))  # Creates [1, 2, 3, ..., 90]

# DEMAND PATTERN PARAMETERS FOR $100 TICKETS (Economy/Low-fare class)
# These parameters define when people want to buy $100 tickets over the 90-day period
mu_x = 300      # Mean (μ) = 45: Peak demand occurs on day 45
                 # This means most $100 ticket buyers book around day 45

var_x = 100      # Variance (σ²) = 250: Standard deviation = √250 ≈ 15.8 days
                 # This means demand is spread ±15.8 days around day 45
                 # High variance indicates $100 ticket demand is spread out over time

# DEMAND PATTERN PARAMETERS FOR $200 TICKETS (Premium/High-fare class)
# These parameters define when people want to buy $200 tickets over the 90-day period
mu_y = 700        # Mean (μ) = 75: Peak demand occurs on day 75
                 # This means most $200 ticket buyers book around day 75
                 # Later than $100 tickets (business travelers book closer to travel)

var_y = 100       # Variance (σ²) = 50: Standard deviation = √50 ≈ 7.1 days
                 # This means demand is spread ±7.1 days around day 75
                 # Lower variance indicates $200 ticket demand is more concentrated

# Comment explaining the domain of variable i
# i = {1,2,...,90} means i takes integer values from 1 to 90 (each represents a day)

# ============================================================================
# DEMAND CONSTRAINT FUNCTIONS (Normal Distribution Curves)
# ============================================================================

# MATHEMATICAL FORMULA: Normal Probability Density Function
# f(x) = A * (1/√(2πσ²)) * e^(-(x-μ)²/(2σ²))
# Where:
# - A = scaling factor (total market size)
# - μ = mean (peak demand day)
# - σ² = variance (spread of demand)
# - x = day number

# $100 TICKET DEMAND LIMITS PER DAY
# 200 = Total market size multiplier for $100 tickets
# This represents the total potential demand for $100 tickets across all 90 days
# The number 200 comes from market research/historical data analysis
normal_x = 150 * (1 / (np.sqrt(2 * np.pi * var_x))) * np.exp(-
    ((np.array(days) - mu_x) ** 2) / (2 * var_x))

# Breaking down the formula:
# - 200: Market size (total $100 ticket demand potential)
# - (1 / (np.sqrt(2 * np.pi * var_x))): Normalization factor for normal distribution
# - np.exp(-((np.array(days) - mu_x) ** 2) / (2 * var_x)): Exponential term
# - Result: normal_x[i] = maximum $100 tickets that can be sold on day i

# $200 TICKET DEMAND LIMITS PER DAY
# 75 = Total market size multiplier for $200 tickets
# This represents the total potential demand for $200 tickets across all 90 days
# The number 75 comes from market research/historical data analysis
# Note: 75 < 200 because fewer people buy premium tickets
normal_y = 100 * (1 / (np.sqrt(2 * np.pi * var_y))) * np.exp(-
    ((np.array(days) - mu_y) ** 2) / (2 * var_y))

# Breaking down the formula:
# - 75: Market size (total $200 ticket demand potential)
# - (1 / (np.sqrt(2 * np.pi * var_y))): Normalization factor for normal distribution
# - np.exp(-((np.array(days) - mu_y) ** 2) / (2 * var_y)): Exponential term
# - Result: normal_y[i] = maximum $200 tickets that can be sold on day i

# ============================================================================
# VISUALIZATION OF DEMAND CONSTRAINTS
# ============================================================================

# Create a plot to visualize the demand limits over time
plt.figure(figsize=(10, 5))  # Set figure size to 10x5 inches

# Plot $100 ticket demand curve
# Label shows the mathematical formula: 200 * N(i; μ=45, σ²=250)
# N(i; μ, σ²) represents normal distribution with mean μ and variance σ²
plt.plot(days, normal_x, label='200 * N(i; μ=45, σ²=250)', color='blue')

# Plot $200 ticket demand curve
# Label shows the mathematical formula: 75 * N(i; μ=75, σ²=50)
plt.plot(days, normal_y, label='75 * N(i; μ=75, σ²=50)', color='green')

# Add labels and formatting
plt.xlabel("Day (i)")                    # X-axis: day number from 1 to 90
plt.ylabel("Max Tickets")                # Y-axis: maximum tickets sellable per day
plt.title("Ticket Limits Over Days")     # Chart title
plt.legend()                            # Show legend with curve labels
plt.grid(True)                          # Add grid for easier reading
plt.show()                              # Display the plot

# ============================================================================
# OPTIMIZATION MODEL SETUP (Constraint Programming with CP-SAT)
# ============================================================================

# Initialize CP-SAT model
# CP-SAT = Constraint Programming - Boolean Satisfiability solver
# This is Google's state-of-the-art constraint programming solver
model = cp_model.CpModel()

# DECISION VARIABLES
# These are the variables the optimizer will determine optimal values for

# x[i] = number of $100 tickets to sell on day i
# Domain: [0, ceil(normal_x[i])] for each day i
# np.ceil() rounds up to ensure we have integer upper bounds
# f'x_{i}' creates variable names like 'x_0', 'x_1', ..., 'x_89'
x = [model.NewIntVar(0, int(np.ceil(normal_x[i])), f'x_{i}') for i in range(90)]

# y[i] = number of $200 tickets to sell on day i
# Domain: [0, ceil(normal_y[i])] for each day i
# np.ceil() rounds up to ensure we have integer upper bounds
# f'y_{i}' creates variable names like 'y_0', 'y_1', ..., 'y_89'
y = [model.NewIntVar(0, int(np.ceil(normal_y[i])), f'y_{i}') for i in range(90)]

# ============================================================================
# CONSTRAINT DEFINITION
# ============================================================================

# CONSTRAINT 1: DAILY DEMAND LIMITS
# Cannot sell more tickets than market demand allows on any given day
# np.floor() rounds down to ensure integer constraints

# Apply daily demand limits for each of the 90 days
for i in range(90):
    # $100 ticket daily limit: x[i] ≤ floor(normal_x[i])
    # floor() ensures we can't sell fractional tickets
    # normal_x[i] gives the maximum $100 tickets demandable on day i
    model.Add(x[i] <= int(np.floor(normal_x[i])))

    # $200 ticket daily limit: y[i] ≤ floor(normal_y[i])
    # floor() ensures we can't sell fractional tickets
    # normal_y[i] gives the maximum $200 tickets demandable on day i
    model.Add(y[i] <= int(np.floor(normal_y[i])))

# CONSTRAINT 2: TOTAL CAPACITY CONSTRAINT
# This is the key resource limitation in the problem
# 200 = Total number of tickets that can be sold across all 90 days
# This represents physical capacity limitation (e.g., aircraft seats, venue capacity)
# The number 200 comes from operational constraints (plane size, etc.)
model.Add(sum(x[i] + y[i] for i in range(90)) <= 200)

# Mathematical representation:
# Σ(x[i] + y[i]) ≤ 200, where i = 0,1,2,...,89 (representing days 1-90)

# ============================================================================
# OBJECTIVE FUNCTION: REVENUE MAXIMIZATION
# ============================================================================

# REVENUE CALCULATION:
# Revenue = (Price per $100 ticket × Number of $100 tickets) +
#          (Price per $200 ticket × Number of $200 tickets)
#
# Revenue = 100 × Σx[i] + 200 × Σy[i] for i = 0 to 89

# Price breakdown:
# - $100 per ticket for x[i] variables (economy/low-fare tickets)
# - $200 per ticket for y[i] variables (premium/high-fare tickets)
# These prices are given as fixed parameters in the problem

# Set optimization objective: maximize total revenue
model.Maximize(sum(100 * x[i] + 200 * y[i] for i in range(90)))

# Mathematical representation:
# max Σ(100*x[i] + 200*y[i]) for i = 0,1,2,...,89

# ============================================================================
# SOLVE THE OPTIMIZATION PROBLEM
# ============================================================================

# Create solver instance
# CpSolver() is the actual solving engine that will find the optimal solution
solver = cp_model.CpSolver()

# Solve the model
# This runs the constraint programming algorithm to find the optimal solution
# Returns status indicating whether solution was found
status = solver.Solve(model)

# ============================================================================
# EXTRACT AND DISPLAY RESULTS
# ============================================================================

# Check if a solution was found
# cp_model.OPTIMAL = best possible solution found
# cp_model.FEASIBLE = valid solution found, but may not be optimal
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solution found.")

    # Extract optimal values for decision variables
    # solver.Value(x[i]) returns the optimal number of $100 tickets for day i
    x_vals = [solver.Value(x[i]) for i in range(90)]

    # solver.Value(y[i]) returns the optimal number of $200 tickets for day i
    y_vals = [solver.Value(y[i]) for i in range(90)]

    # Calculate total revenue using optimal solution
    # Revenue = 100 × (total $100 tickets) + 200 × (total $200 tickets)
    total_revenue = sum(100 * x_vals[i] + 200 * y_vals[i] for i in range(90))

    # Display key results
    print(f"Total Revenue: ${total_revenue}")
    print(f"Total $100 tickets: {sum(x_vals)}")      # Sum of all x[i] values
    print(f"Total $200 tickets: {sum(y_vals)}")      # Sum of all y[i] values

    # ========================================================================
    # SOLUTION VISUALIZATION
    # ========================================================================

    # Create bar chart showing optimal ticket allocation by day
    plt.figure(figsize=(12, 6))  # Set figure size to 12x6 inches

    # Plot $100 tickets as blue bars
    # x_vals contains the optimal number of $100 tickets for each day
    plt.bar(days, x_vals, label='$100 tickets (x_i)', color='blue', alpha=0.6)

    # Plot $200 tickets as green bars stacked on top of $100 tickets
    # bottom=x_vals makes the green bars start where blue bars end
    # y_vals contains the optimal number of $200 tickets for each day
    plt.bar(days, y_vals, label='$200 tickets (y_i)', color='green',
            alpha=0.6, bottom=x_vals)

    # Add labels and formatting
    plt.xlabel("Day")                                    # X-axis: day number
    plt.ylabel("Tickets Sold")                          # Y-axis: number of tickets
    plt.title("Optimal Integer Ticket Sales Per Day")   # Chart title
    plt.legend()                                        # Show legend
    plt.grid(True)                                      # Add grid
    plt.tight_layout()                                  # Optimize spacing
    plt.show()

    # Calculate daily revenue
    revenue_x = [100 * x_vals[i] for i in range(90)]  # Revenue from $100 tickets
    revenue_y = [200 * y_vals[i] for i in range(90)]  # Revenue from $200 tickets
    revenue_total = [revenue_x[i] + revenue_y[i] for i in range(90)]  # Total daily revenue

    # Create revenue plot
    plt.figure(figsize=(12, 6))  # Set figure size to 12x6 inches

    # Plot revenue from $100 tickets
    plt.bar(days, revenue_x, label='Revenue from $100 tickets', color='blue', alpha=0.6)

    # Plot revenue from $200 tickets stacked on top
    plt.bar(days, revenue_y, label='Revenue from $200 tickets', color='green', alpha=0.6, bottom=revenue_x)

    # Add a line for total daily revenue
    plt.plot(days, revenue_total, label='Total Daily Revenue', color='red', linewidth=2)

    # Add labels and formatting
    plt.xlabel("Day")
    plt.ylabel("Revenue ($)")
    plt.title("Daily Revenue Breakdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Create cumulative revenue plot
    cumulative_revenue = np.cumsum(revenue_total)  # Calculate cumulative sum of daily revenue

    plt.figure(figsize=(12, 6))
    plt.plot(days, cumulative_revenue, label='Cumulative Revenue', color='red', linewidth=2)
    plt.xlabel("Day")
    plt.ylabel("Cumulative Revenue ($)")
    plt.title("Cumulative Revenue Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Display plot

else:
    # If no solution found, display error message
    print("No feasible solution found.")

# ============================================================================
# ALGORITHM EXPLANATION: DIRECT REVENUE MAXIMIZATION
# ============================================================================

"""
ALGORITHM APPROACH: Direct Revenue Maximization

This algorithm uses constraint programming to directly maximize revenue:

1. DECISION VARIABLES:
   - x[i] = $100 tickets sold on day i (for i = 1 to 90)
   - y[i] = $200 tickets sold on day i (for i = 1 to 90)

2. OBJECTIVE FUNCTION:
   - Maximize: Σ(100 × x[i] + 200 × y[i]) for i = 1 to 90
   - Direct revenue calculation: price × quantity for each fare class

3. CONSTRAINTS:
   - Daily demand limits: x[i] ≤ normal_x[i], y[i] ≤ normal_y[i]
   - Total capacity: Σ(x[i] + y[i]) ≤ 200

4. SOLUTION METHOD:
   - Constraint Programming with CP-SAT solver
   - Finds optimal integer solution that maximizes total revenue

KEY NUMBERS EXPLANATION:
- 45, 75: Peak demand days (from historical booking patterns)
- 250, 50: Demand variance (spread of booking behavior)  
- 200, 75: Market size multipliers (total demand potential)
- 100, 200: Ticket prices (given fare structure)
- 200: Capacity limit (operational constraint)
- 90: Time horizon (booking window length)

This is a pure revenue maximization approach, not EMSR (Expected Marginal 
Seat Revenue) which would involve probabilistic demand forecasting and 
protection levels for different fare classes.
"""