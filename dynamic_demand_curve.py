"""
SIMPLE DATA GENERATOR
====================

Generate realistic sales data for any number of days and run simulation.
As simple as possible!
"""

import numpy as np
import matplotlib.pyplot as plt

class SimpleSystem:
    def __init__(self, total_days):
        # Rational parameters based on total days
        self.mu_x = int(total_days * 0.4)      # Economy peak at 40%
        self.mu_y = int(total_days * 0.7)      # Business peak at 70%
        self.market_x = int(total_days * 2.2)  # Economy market size
        self.market_y = int(total_days * 0.8)  # Business market size
        self.var_x = (total_days * 0.3) ** 2   # Economy variance
        self.var_y = (total_days * 0.15) ** 2  # Business variance

        self.total_days = total_days
        self.current_day = 0
        self.actual_sales = []
        self.curve_history = []

        print(f"📊 System setup for {total_days} days:")
        print(f"   Economy peak: day {self.mu_x}")
        print(f"   Business peak: day {self.mu_y}")

        # Save initial curve
        self._save_curve("Initial")

    def _save_curve(self, label):
        """Save current curve state"""
        days = np.arange(1, self.total_days + 1)

        eco_curve = self.market_x * (1 / np.sqrt(2 * np.pi * self.var_x)) * \
                   np.exp(-((days - self.mu_x) ** 2) / (2 * self.var_x))

        bus_curve = self.market_y * (1 / np.sqrt(2 * np.pi * self.var_y)) * \
                   np.exp(-((days - self.mu_y) ** 2) / (2 * self.var_y))

        self.curve_history.append({
            'label': label,
            'days': days,
            'economy': eco_curve,
            'business': bus_curve,
            'mu_x': self.mu_x,
            'mu_y': self.mu_y
        })

    def process_day(self, economy_sold, business_sold):
        """Process one day of sales"""
        self.current_day += 1
        self.actual_sales.append((economy_sold, business_sold))

        # Compare with forecast
        if len(self.curve_history) > 0:
            last_curve = self.curve_history[-1]
            day_index = self.current_day - 1

            expected_eco = last_curve['economy'][day_index]
            expected_bus = last_curve['business'][day_index]

            eco_variance = economy_sold - expected_eco
            bus_variance = business_sold - expected_bus

            # Check if adaptation needed
            if self._needs_adaptation(eco_variance, bus_variance):
                self._adapt_parameters(eco_variance, bus_variance)
                self._save_curve(f"Day {self.current_day}")
                return True  # Adaptation made

        return False  # No adaptation

    def _needs_adaptation(self, eco_var, bus_var):
        """Simple check if adaptation is needed"""
        return abs(eco_var) > 1.5 or abs(bus_var) > 0.8 or abs(eco_var + bus_var) > 2

    def _adapt_parameters(self, eco_var, bus_var):
        """Simple parameter adaptation"""
        # Economy peak
        if abs(eco_var) > 1.5:
            if eco_var > 0:  # Higher than expected
                self.mu_x = max(1, self.mu_x - 2)
            else:  # Lower than expected
                self.mu_x = min(self.total_days - 5, self.mu_x + 2)

        # Business peak
        if abs(bus_var) > 0.8:
            if bus_var > 0:
                self.mu_y = max(self.mu_x + 5, self.mu_y - 2)
            else:
                self.mu_y = min(self.total_days - 2, self.mu_y + 2)

        # Market size
        total_var = eco_var + bus_var
        if abs(total_var) > 2:
            if total_var > 0:
                self.market_x *= 1.05
                self.market_y *= 1.05
            else:
                self.market_x *= 0.95
                self.market_y *= 0.95

    def show_results(self):
        """Show compact visualization and complete data"""
        # Smaller figure size
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.curve_history)))

        # Economy curves
        ax1.set_title('Economy Curves', fontsize=10)
        for i, curve in enumerate(self.curve_history):
            ax1.plot(curve['days'], curve['economy'], color=colors[i],
                    linewidth=1.5, alpha=0.8, label=f"{curve['label']} (pk:{curve['mu_x']})")

        # Actual sales
        if self.actual_sales:
            actual_days = list(range(1, len(self.actual_sales) + 1))
            actual_economy = [sale[0] for sale in self.actual_sales]
            ax1.scatter(actual_days, actual_economy, color='red', s=15,
                       alpha=0.8, label='Actual', zorder=10)

        ax1.set_xlabel('Day', fontsize=9)
        ax1.set_ylabel('Economy Tickets', fontsize=9)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)

        # Business curves
        ax2.set_title('Business Curves', fontsize=10)
        for i, curve in enumerate(self.curve_history):
            ax2.plot(curve['days'], curve['business'], color=colors[i],
                    linewidth=1.5, alpha=0.8, label=f"{curve['label']} (pk:{curve['mu_y']})")

        # Actual sales
        if self.actual_sales:
            actual_business = [sale[1] for sale in self.actual_sales]
            ax2.scatter(actual_days, actual_business, color='red', s=15,
                       alpha=0.8, label='Actual', zorder=10)

        ax2.set_xlabel('Day', fontsize=9)
        ax2.set_ylabel('Business Tickets', fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)

        plt.tight_layout()
        plt.show()

        # Print complete daily data
        self._print_complete_data()

    def _print_complete_data(self):
        """Print complete daily sales data and revenue"""
        print(f"\n📊 COMPLETE DAILY SALES DATA:")
        print("="*95)
        print(f"{'Day':<4} | {'Exp Eco':<7} | {'Act Eco':<7} | {'Exp Bus':<7} | {'Act Bus':<7} | {'Daily Rev':<10} | {'Cumulative':<12}")
        print("-"*95)

        # Get initial expectations
        initial_curve = self.curve_history[0]  # Initial curve before any adaptations

        cumulative_revenue = 0
        total_economy = 0
        total_business = 0
        total_expected_economy = 0
        total_expected_business = 0

        for day, (eco, bus) in enumerate(self.actual_sales, 1):
            daily_revenue = 100 * eco + 200 * bus
            cumulative_revenue += daily_revenue
            total_economy += eco
            total_business += bus

            # Get expected values from initial curve
            day_index = day - 1
            expected_eco = initial_curve['economy'][day_index]
            expected_bus = initial_curve['business'][day_index]
            total_expected_economy += expected_eco
            total_expected_business += expected_bus

            print(f"{day:<4} | {expected_eco:<7.1f} | {eco:<7} | {expected_bus:<7.1f} | {bus:<7} | ${daily_revenue:<9} | ${cumulative_revenue:<11}")

        print("-"*95)
        print(f"{'TOTAL':<4} | {total_expected_economy:<7.0f} | {total_economy:<7} | {total_expected_business:<7.0f} | {total_business:<7} | ${cumulative_revenue:<9} | ")
        print("="*95)

        # Variance analysis
        eco_variance = total_economy - total_expected_economy
        bus_variance = total_business - total_expected_business
        expected_revenue = total_expected_economy * 100 + total_expected_business * 200
        revenue_variance = cumulative_revenue - expected_revenue

        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"Total days processed: {len(self.actual_sales)}")
        print(f"Total economy tickets: {total_economy:,} (expected: {total_expected_economy:.0f}, variance: {eco_variance:+.0f})")
        print(f"Total business tickets: {total_business:,} (expected: {total_expected_business:.0f}, variance: {bus_variance:+.0f})")
        print(f"Total tickets sold: {total_economy + total_business:,}")
        print(f"Total revenue: ${cumulative_revenue:,} (expected: ${expected_revenue:.0f}, variance: ${revenue_variance:+.0f})")
        print(f"Average daily revenue: ${cumulative_revenue/len(self.actual_sales):.2f}")
        print(f"Economy revenue: ${total_economy * 100:,} ({total_economy * 100 / cumulative_revenue:.1%})")
        print(f"Business revenue: ${total_business * 200:,} ({total_business * 200 / cumulative_revenue:.1%})")

        # Performance vs expectations
        print(f"\n📊 PERFORMANCE vs INITIAL EXPECTATIONS:")
        print(f"Economy variance: {eco_variance:+.0f} tickets ({eco_variance/total_expected_economy*100:+.1f}%)")
        print(f"Business variance: {bus_variance:+.0f} tickets ({bus_variance/total_expected_business*100:+.1f}%)")
        print(f"Revenue variance: ${revenue_variance:+.0f} ({revenue_variance/expected_revenue*100:+.1f}%)")

        # Adaptation summary
        print(f"\n🔄 ALGORITHM ADAPTATIONS:")
        print(f"Curve adaptations made: {len(self.curve_history) - 1}")
        original_eco_peak = int(self.total_days * 0.4)
        original_bus_peak = int(self.total_days * 0.7)
        print(f"Economy peak: day {original_eco_peak} → day {self.mu_x} (change: {self.mu_x - original_eco_peak:+d})")
        print(f"Business peak: day {original_bus_peak} → day {self.mu_y} (change: {self.mu_y - original_bus_peak:+d})")

        return {
            'daily_data': self.actual_sales,
            'expected_data': [(initial_curve['economy'][i], initial_curve['business'][i]) for i in range(len(self.actual_sales))],
            'total_economy': total_economy,
            'total_business': total_business,
            'expected_economy': total_expected_economy,
            'expected_business': total_expected_business,
            'total_revenue': cumulative_revenue,
            'expected_revenue': expected_revenue,
            'adaptations': len(self.curve_history) - 1,
            'final_eco_peak': self.mu_x,
            'final_bus_peak': self.mu_y
        }

def generate_data(total_days):
    """Generate realistic sales data for given number of days"""
    print(f"🎲 Generating data for {total_days} days...")

    # Create scenario: Economy peaks earlier than expected
    actual_eco_peak = int(total_days * 0.25)  # Peak at 25% instead of 40%
    actual_bus_peak = int(total_days * 0.7)   # Business normal

    sales_data = []

    for day in range(1, total_days + 1):
        # Economy: peaks earlier than system expects
        eco_base = (total_days * 1.8) * (1 / np.sqrt(2 * np.pi * (total_days * 0.2) ** 2)) * \
                   np.exp(-((day - actual_eco_peak) ** 2) / (2 * (total_days * 0.2) ** 2))
        eco_actual = max(0, int(eco_base + np.random.normal(0, eco_base * 0.15)))

        # Business: follows expected pattern
        bus_base = (total_days * 0.7) * (1 / np.sqrt(2 * np.pi * (total_days * 0.1) ** 2)) * \
                   np.exp(-((day - actual_bus_peak) ** 2) / (2 * (total_days * 0.1) ** 2))
        bus_actual = max(0, int(bus_base + np.random.normal(0, bus_base * 0.1)))

        sales_data.append((eco_actual, bus_actual))

    print(f"✅ Data generated:")
    print(f"   Economy peaks around day {actual_eco_peak}")
    print(f"   Business peaks around day {actual_bus_peak}")
    print(f"   Total economy: {sum(e for e, b in sales_data)}")
    print(f"   Total business: {sum(b for e, b in sales_data)}")

    return sales_data

def run_simulation(total_days, sales_data):
    """Run the simulation"""
    print(f"\n🚀 Running simulation...")

    system = SimpleSystem(total_days)
    adaptations = 0

    for day, (eco, bus) in enumerate(sales_data, 1):
        adapted = system.process_day(eco, bus)
        if adapted:
            adaptations += 1
            print(f"🔄 Adaptation on day {day}")

    print(f"✅ Simulation complete!")
    print(f"   Adaptations made: {adaptations}")

    return system

def main():
    """Main function - as simple as possible"""
    print("🎯 SIMPLE DATA GENERATOR")
    print("="*30)

    # Get number of days from user
    try:
        total_days = int(input("How many days? "))
        if total_days < 10:
            total_days = 10
            print("Minimum 10 days, using 10")
        elif total_days > 365:
            total_days = 365
            print("Maximum 365 days, using 365")
    except:
        total_days = 90
        print("Invalid input, using 90 days")

    # Generate data
    sales_data = generate_data(total_days)

    # Run simulation
    system = run_simulation(total_days, sales_data)

    # Show results
    system.show_results()

    print(f"\n🎉 Done! The algorithm learned from {total_days} days of data.")

if __name__ == "__main__":
    main()