def step1_simple_revenue():
    """Basic capacity vs demand calculation"""
    # Constants
    capacity = 150
    fare = 200
    demand = 180

    # Algorithm: sell minimum of capacity and demand
    seats_sold = min(capacity, demand)
    revenue = seats_sold * fare

    print(f"Step 1 Results:")
    print(f"Capacity: {capacity}, Demand: {demand}, Fare: ${fare}")
    print(f"Seats sold: {seats_sold}")
    print(f"Revenue: ${revenue:,}")
    print(f"Load factor: {seats_sold / capacity:.1%}")

    return revenue


# Test Step 1
step1_revenue = step1_simple_revenue()
print("-" * 50)

import numpy as np


def step2_uncertain_demand():
    """Handle uncertain demand with probability"""
    # Constants
    capacity = 150
    fare = 200
    demand_mean = 180
    demand_std = 20

    # Simulate multiple scenarios
    np.random.seed(42)  # For reproducible results
    scenarios = 10000
    total_revenue = 0

    for _ in range(scenarios):
        # Generate random demand
        demand = max(0, np.random.normal(demand_mean, demand_std))
        seats_sold = min(capacity, demand)
        revenue = seats_sold * fare
        total_revenue += revenue

    avg_revenue = total_revenue / scenarios

    print(f"Step 2 Results (Uncertain Demand):")
    print(f"Average demand: {demand_mean} ± {demand_std}")
    print(f"Expected revenue: ${avg_revenue:,.0f}")
    print(f"Revenue increase vs Step 1: ${avg_revenue - step1_revenue:,.0f}")

    return avg_revenue


# Test Step 2
step2_revenue = step2_uncertain_demand()
print("-" * 50)


def step3_multiple_fares_deterministic():
    """Multiple fare classes with known demand"""
    # Constants
    capacity = 150

    # Fare classes: (fare, demand)
    fare_classes = [
        ("Business", 500, 30),
        ("Economy", 200, 140)
    ]

    # Greedy algorithm: serve highest fare first
    remaining_capacity = capacity
    total_revenue = 0
    allocation = {}

    # Sort by fare (highest first)
    sorted_classes = sorted(fare_classes, key=lambda x: x[1], reverse=True)

    for class_name, fare, demand in sorted_classes:
        seats_allocated = min(remaining_capacity, demand)
        allocation[class_name] = seats_allocated
        total_revenue += seats_allocated * fare
        remaining_capacity -= seats_allocated

        if remaining_capacity == 0:
            break

    print(f"Step 3 Results (Multiple Fare Classes):")
    for class_name, seats in allocation.items():
        fare = next(f[1] for f in fare_classes if f[0] == class_name)
        print(f"{class_name}: {seats} seats × ${fare} = ${seats * fare:,}")
    print(f"Total revenue: ${total_revenue:,}")
    print(f"Load factor: {sum(allocation.values()) / capacity:.1%}")

    return total_revenue


# Test Step 3
step3_revenue = step3_multiple_fares_deterministic()
print("-" * 50)

from scipy import stats


def step4_emsr_optimization():
    """EMSR model for probabilistic demand"""
    # Constants
    capacity = 150

    # Fare classes with probabilistic demand: (name, fare, mean_demand, std_demand)
    fare_classes = [
        ("Business", 500, 30, 8),
        ("Economy", 200, 140, 25)
    ]

    def calculate_protection_level(high_fare, low_fare, high_demand_mean, high_demand_std):
        """Calculate optimal seats to protect for high fare class"""
        # EMSR condition: P(demand > protection) = low_fare / high_fare
        critical_ratio = low_fare / high_fare

        # Find protection level where P(X > protection) = critical_ratio
        # This means P(X <= protection) = 1 - critical_ratio
        protection = stats.norm.ppf(1 - critical_ratio, high_demand_mean, high_demand_std)
        return max(0, min(capacity, protection))

    # Calculate optimal protection for business class
    business_protection = calculate_protection_level(
        high_fare=500, low_fare=200,
        high_demand_mean=30, high_demand_std=8
    )

    # Set booking limits
    business_limit = capacity
    economy_limit = capacity - business_protection

    # Simulate revenue
    np.random.seed(42)
    scenarios = 10000
    total_revenue = 0

    for _ in range(scenarios):
        business_demand = max(0, np.random.normal(30, 8))
        economy_demand = max(0, np.random.normal(140, 25))

        # Apply booking limits
        business_sold = min(business_demand, business_limit)
        remaining_capacity_after_business = capacity - business_sold
        economy_sold = min(economy_demand, economy_limit, remaining_capacity_after_business)

        revenue = business_sold * 500 + economy_sold * 200
        total_revenue += revenue

    avg_revenue = total_revenue / scenarios

    print(f"Step 4 Results (EMSR Optimization):")
    print(f"Business protection: {business_protection:.1f} seats")
    print(f"Economy limit: {economy_limit:.1f} seats")
    print(f"Expected revenue: ${avg_revenue:,.0f}")
    print(f"Revenue increase vs Step 3: ${avg_revenue - step3_revenue:,.0f}")

    return avg_revenue


# Test Step 4
step4_revenue = step4_emsr_optimization()
print("-" * 50)


def step5_overbooking():
    """Add overbooking to handle no-shows"""
    # Constants
    capacity = 150
    no_show_rate = 0.15  # 15% average
    no_show_std = 0.05  # 5% standard deviation
    denied_boarding_cost = 400

    # Fare classes (same as Step 4)
    fare_classes = [
        ("Business", 500, 30, 8),
        ("Economy", 200, 140, 25)
    ]

    def optimize_authorization_limit(capacity, no_show_rate, no_show_std, avg_fare):
        """Find optimal authorization limit using probabilistic model"""
        # Try different authorization limits
        best_revenue = 0
        best_auth_limit = capacity

        for auth_limit in range(capacity, int(capacity * 1.3)):
            expected_revenue = 0
            scenarios = 1000

            for _ in range(scenarios):
                # Simulate no-show rate
                actual_no_show = max(0, min(1, np.random.normal(no_show_rate, no_show_std)))

                # Calculate passengers who show up
                booked_passengers = auth_limit
                show_up = booked_passengers * (1 - actual_no_show)

                if show_up <= capacity:
                    # No denied boardings
                    revenue = booked_passengers * avg_fare
                    spoilage_cost = (capacity - show_up) * avg_fare
                    net_revenue = revenue - spoilage_cost
                else:
                    # Some denied boardings
                    denied = show_up - capacity
                    revenue = booked_passengers * avg_fare
                    denied_cost = denied * denied_boarding_cost
                    net_revenue = revenue - denied_cost

                expected_revenue += net_revenue

            avg_expected_revenue = expected_revenue / scenarios
            if avg_expected_revenue > best_revenue:
                best_revenue = avg_expected_revenue
                best_auth_limit = auth_limit

        return best_auth_limit, best_revenue

    # Calculate average fare from mix
    avg_fare = (30 * 500 + 120 * 200) / 150  # Approximate mix

    # Find optimal authorization limit
    optimal_auth_limit, expected_revenue = optimize_authorization_limit(
        capacity, no_show_rate, no_show_std, avg_fare
    )

    print(f"Step 5 Results (Overbooking):")
    print(f"Physical capacity: {capacity}")
    print(f"Optimal authorization limit: {optimal_auth_limit}")
    print(f"Overbooking rate: {(optimal_auth_limit - capacity) / capacity:.1%}")
    print(f"Expected revenue with overbooking: ${expected_revenue:,.0f}")
    print(f"Revenue increase vs Step 4: ${expected_revenue - step4_revenue:,.0f}")

    return expected_revenue


# Test Step 5
step5_revenue = step5_overbooking()
print("-" * 50)


def step6_dynamic_reoptimization():
    """Dynamic adjustment of booking limits over time"""
    # Constants
    capacity = 150
    days_before_departure = [60, 30, 14, 7, 1]

    # Initial demand forecasts (will be updated)
    initial_forecasts = {
        "Business": {"mean": 30, "std": 8},
        "Economy": {"mean": 140, "std": 25}
    }

    def update_forecast(initial_forecast, actual_bookings, days_remaining):
        """Update demand forecast based on actual bookings"""
        # Simple linear update (in practice, this would use time series analysis)
        booking_rate = actual_bookings / (60 - days_remaining + 1) if days_remaining < 60 else 0
        remaining_demand = initial_forecast["mean"] - actual_bookings

        # Adjust forecast based on booking pace
        adjustment_factor = 1.0 if days_remaining > 30 else 0.8
        new_mean = max(0, remaining_demand * adjustment_factor)
        new_std = initial_forecast["std"] * (days_remaining / 60)

        return {"mean": new_mean, "std": new_std}

    # Simulate booking process
    np.random.seed(42)
    current_bookings = {"Business": 0, "Economy": 0}
    total_revenue = 0

    print(f"Step 6 Results (Dynamic Reoptimization):")
    print("Day | Business | Economy | Business Limit | Economy Limit | Revenue")
    print("-" * 70)

    for day in days_before_departure:
        # Update forecasts based on current bookings
        business_forecast = update_forecast(
            initial_forecasts["Business"],
            current_bookings["Business"],
            day
        )
        economy_forecast = update_forecast(
            initial_forecasts["Economy"],
            current_bookings["Economy"],
            day
        )

        # Recalculate protection levels
        if business_forecast["mean"] > 0 and business_forecast["std"] > 0:
            critical_ratio = 200 / 500
            business_protection = stats.norm.ppf(
                1 - critical_ratio,
                business_forecast["mean"],
                business_forecast["std"]
            )
            business_protection = max(0, min(capacity - current_bookings["Business"], business_protection))
        else:
            business_protection = 0

        # Set new booking limits
        business_limit = capacity
        economy_limit = capacity - business_protection - current_bookings["Business"]

        # Simulate new bookings for this period
        new_business = min(
            max(0, np.random.poisson(business_forecast["mean"] / len(days_before_departure))),
            business_limit - current_bookings["Business"]
        )
        new_economy = min(
            max(0, np.random.poisson(economy_forecast["mean"] / len(days_before_departure))),
            economy_limit - current_bookings["Economy"]
        )

        current_bookings["Business"] += new_business
        current_bookings["Economy"] += new_economy

        current_revenue = current_bookings["Business"] * 500 + current_bookings["Economy"] * 200

        print(f"{day:3d} | {current_bookings['Business']:8d} | {current_bookings['Economy']:7d} | "
              f"{business_limit:14.0f} | {economy_limit:13.0f} | ${current_revenue:,}")

    final_revenue = current_bookings["Business"] * 500 + current_bookings["Economy"] * 200
    print(f"\nFinal revenue: ${final_revenue:,}")
    print(f"Revenue increase vs Step 5: ${final_revenue - step5_revenue:,.0f}")

    return final_revenue


# Test Step 6
step6_revenue = step6_dynamic_reoptimization()
print("-" * 50)


def step6_dynamic_reoptimization():
    """Dynamic adjustment of booking limits over time"""
    # Constants
    capacity = 150
    days_before_departure = [60, 30, 14, 7, 1]

    # Initial demand forecasts (will be updated)
    initial_forecasts = {
        "Business": {"mean": 30, "std": 8},
        "Economy": {"mean": 140, "std": 25}
    }

    def update_forecast(initial_forecast, actual_bookings, days_remaining):
        """Update demand forecast based on actual bookings"""
        # Simple linear update (in practice, this would use time series analysis)
        booking_rate = actual_bookings / (60 - days_remaining + 1) if days_remaining < 60 else 0
        remaining_demand = initial_forecast["mean"] - actual_bookings

        # Adjust forecast based on booking pace
        adjustment_factor = 1.0 if days_remaining > 30 else 0.8
        new_mean = max(0, remaining_demand * adjustment_factor)
        new_std = initial_forecast["std"] * (days_remaining / 60)

        return {"mean": new_mean, "std": new_std}

    # Simulate booking process
    np.random.seed(42)
    current_bookings = {"Business": 0, "Economy": 0}
    total_revenue = 0

    print(f"Step 6 Results (Dynamic Reoptimization):")
    print("Day | Business | Economy | Business Limit | Economy Limit | Revenue")
    print("-" * 70)

    for day in days_before_departure:
        # Update forecasts based on current bookings
        business_forecast = update_forecast(
            initial_forecasts["Business"],
            current_bookings["Business"],
            day
        )
        economy_forecast = update_forecast(
            initial_forecasts["Economy"],
            current_bookings["Economy"],
            day
        )

        # Recalculate protection levels
        if business_forecast["mean"] > 0 and business_forecast["std"] > 0:
            critical_ratio = 200 / 500
            business_protection = stats.norm.ppf(
                1 - critical_ratio,
                business_forecast["mean"],
                business_forecast["std"]
            )
            business_protection = max(0, min(capacity - current_bookings["Business"], business_protection))
        else:
            business_protection = 0

        # Set new booking limits
        business_limit = capacity
        economy_limit = capacity - business_protection - current_bookings["Business"]

        # Simulate new bookings for this period
        new_business = min(
            max(0, np.random.poisson(business_forecast["mean"] / len(days_before_departure))),
            business_limit - current_bookings["Business"]
        )
        new_economy = min(
            max(0, np.random.poisson(economy_forecast["mean"] / len(days_before_departure))),
            economy_limit - current_bookings["Economy"]
        )

        current_bookings["Business"] += new_business
        current_bookings["Economy"] += new_economy

        current_revenue = current_bookings["Business"] * 500 + current_bookings["Economy"] * 200

        print(f"{day:3d} | {current_bookings['Business']:8d} | {current_bookings['Economy']:7d} | "
              f"{business_limit:14.0f} | {economy_limit:13.0f} | ${current_revenue:,}")

    final_revenue = current_bookings["Business"] * 500 + current_bookings["Economy"] * 200
    print(f"\nFinal revenue: ${final_revenue:,}")
    print(f"Revenue increase vs Step 5: ${final_revenue - step5_revenue:,.0f}")

    return final_revenue


# Test Step 6
step6_revenue = step6_dynamic_reoptimization()
print("-" * 50)
