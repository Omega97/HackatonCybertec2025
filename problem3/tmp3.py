import pandas as pd
import numpy as np
import pulp
import sys
import os

def main(demand_file, capacity_file, production_cost_file, shipment_cost_file):
    # Load data
    capacity_df = pd.read_csv(capacity_file)
    demand_df = pd.read_csv(demand_file)
    prod_cost_df = pd.read_csv(production_cost_file)
    ship_cost_df = pd.read_csv(shipment_cost_file)

    # Data preparation
    countries = capacity_df['Country'].unique()
    products = demand_df['Product'].unique()
    months = demand_df['Month'].unique()

    # Create dictionaries for easier access
    capacity_dict = {row['Country']: row['Monthly Capacity'] for _, row in capacity_df.iterrows()}
    demand_dict = {(row['Country'], row['Product'], row['Month']): row['Quantity'] 
                  for _, row in demand_df.iterrows()}
    prod_cost_dict = {(row['Country'], row['Product']): row['Unit Cost'] 
                     for _, row in prod_cost_df.iterrows()}
    ship_cost_dict = {(row['Origin'], row['Destination']): row['Unit Cost'] 
                     for _, row in ship_cost_df.iterrows()}

    # Step 1: First optimize for balance only
    def solve_for_balance():
        prob = pulp.LpProblem("Production_Balance", pulp.LpMinimize)
        
        # Variables
        production = pulp.LpVariable.dicts("Production", 
                                         ((c, p, m) for c in countries for p in products for m in months),
                                         lowBound=0, cat=pulp.LpContinuous)
        
        shipments = pulp.LpVariable.dicts("Shipment", 
                                         ((o, d, p, m) for o in countries for d in countries if o != d 
                                          for p in products for m in months),
                                         lowBound=0, cat=pulp.LpContinuous)
        
        # Production capacity constraints
        for c in countries:
            for m in months:
                prob += pulp.lpSum(production[c, p, m] for p in products) <= capacity_dict[c]
        
        # Demand satisfaction constraints
        for d in countries:
            for p in products:
                for m in months:
                    demand = demand_dict.get((d, p, m), 0)
                    prod_in_country = production[d, p, m]
                    incoming = pulp.lpSum(shipments[o, d, p, m] for o in countries if o != d)
                    outgoing = pulp.lpSum(shipments[d, o, p, m] for o in countries if o != d)
                    
                    prob += prod_in_country + incoming - outgoing == demand
        
        # CRITICAL FIX: Add constraint that production must be sufficient for shipments out
        for i in countries:
            for p in products:
                for m in months:
                    outgoing = pulp.lpSum(shipments[i, j, p, m] for j in countries if i != j)
                    prob += production[i, p, m] >= outgoing
        
        # Calculate utilization rates
        utilization = {}
        for c in countries:
            for m in months:
                util_var = pulp.LpVariable(f"Util_{c}_{m}", 0, 1, pulp.LpContinuous)
                prob += util_var == pulp.lpSum(production[c, p, m] for p in products) / capacity_dict[c]
                utilization[c, m] = util_var
        
        # Calculate average utilization
        avg_util = pulp.LpVariable("Avg_Util", 0, 1, pulp.LpContinuous)
        prob += avg_util == pulp.lpSum(utilization[c, m] for c in countries for m in months) / (len(countries) * len(months))
        
        # Calculate squared deviations (approximated linearly)
        deviations = []
        for c in countries:
            for m in months:
                pos_dev = pulp.LpVariable(f"Pos_Dev_{c}_{m}", 0, None, pulp.LpContinuous)
                neg_dev = pulp.LpVariable(f"Neg_Dev_{c}_{m}", 0, None, pulp.LpContinuous)
                prob += utilization[c, m] - avg_util == pos_dev - neg_dev
                deviations.append(pos_dev + neg_dev)  # Linear approximation of absolute deviation
        
        # Objective: minimize sum of absolute deviations (proxy for RMSSE)
        prob += pulp.lpSum(deviations)
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Calculate actual RMSSE
        utils = [utilization[c, m].value() for c in countries for m in months]
        avg = avg_util.value()
        rmsse = np.sqrt(np.mean([(u - avg)**2 for u in utils]))
        
        return {
            'rmsse': rmsse,
            'production': {(c, p, m): production[c, p, m].value() for c in countries for p in products for m in months},
            'shipments': {(o, d, p, m): shipments[o, d, p, m].value() for o in countries for d in countries 
                          if o != d for p in products for m in months},
            'average_utilization': avg
        }

    # Step 2: Optimize for cost with a constraint on balance quality
    def solve_for_cost(optimal_rmsse, tolerance=0.05):
        prob = pulp.LpProblem("Production_Cost", pulp.LpMinimize)
        
        # Variables
        production = pulp.LpVariable.dicts("Production", 
                                         ((c, p, m) for c in countries for p in products for m in months),
                                         lowBound=0, cat=pulp.LpContinuous)
        
        shipments = pulp.LpVariable.dicts("Shipment", 
                                         ((o, d, p, m) for o in countries for d in countries if o != d 
                                          for p in products for m in months),
                                         lowBound=0, cat=pulp.LpContinuous)
        
        # Same constraints as before
        for c in countries:
            for m in months:
                prob += pulp.lpSum(production[c, p, m] for p in products) <= capacity_dict[c]
        
        for d in countries:
            for p in products:
                for m in months:
                    demand = demand_dict.get((d, p, m), 0)
                    prod_in_country = production[d, p, m]
                    incoming = pulp.lpSum(shipments[o, d, p, m] for o in countries if o != d)
                    outgoing = pulp.lpSum(shipments[d, o, p, m] for o in countries if o != d)
                    
                    prob += prod_in_country + incoming - outgoing == demand
        
        # CRITICAL FIX: Add constraint that production must be sufficient for shipments out
        for i in countries:
            for p in products:
                for m in months:
                    outgoing = pulp.lpSum(shipments[i, j, p, m] for j in countries if i != j)
                    prob += production[i, p, m] >= outgoing
        
        # Calculate utilization rates for RMSSE constraint
        utilization = {}
        for c in countries:
            for m in months:
                util_var = pulp.LpVariable(f"Util_{c}_{m}", 0, 1, pulp.LpContinuous)
                prob += util_var == pulp.lpSum(production[c, p, m] for p in products) / capacity_dict[c]
                utilization[c, m] = util_var
        
        # Calculate average utilization
        avg_util = pulp.LpVariable("Avg_Util", 0, 1, pulp.LpContinuous)
        prob += avg_util == pulp.lpSum(utilization[c, m] for c in countries for m in months) / (len(countries) * len(months))
        
        # Constrain RMSSE to be within tolerance of optimal
        # We use a linear approximation with absolute deviations
        deviations = []
        for c in countries:
            for m in months:
                pos_dev = pulp.LpVariable(f"Pos_Dev_{c}_{m}", 0, None, pulp.LpContinuous)
                neg_dev = pulp.LpVariable(f"Neg_Dev_{c}_{m}", 0, None, pulp.LpContinuous)
                prob += utilization[c, m] - avg_util == pos_dev - neg_dev
                deviations.append(pos_dev + neg_dev)
        
        # Approximation of RMSSE constraint using sum of absolute deviations
        max_allowed_dev_sum = pulp.LpVariable("Max_Dev_Sum", 0, None, pulp.LpContinuous)
        prob += max_allowed_dev_sum == optimal_rmsse * (1 + tolerance) * len(countries) * len(months)
        prob += pulp.lpSum(deviations) <= max_allowed_dev_sum
        
        # Calculate total cost
        total_prod_cost = pulp.lpSum(
            prod_cost_dict.get((c, p), 0) * production[c, p, m]
            for c in countries for p in products for m in months
            if (c, p) in prod_cost_dict
        )
        
        total_ship_cost = pulp.lpSum(
            ship_cost_dict.get((o, d), 0) * shipments[o, d, p, m]
            for o in countries for d in countries if o != d
            for p in products for m in months
            if (o, d) in ship_cost_dict
        )
        
        # Objective: minimize total cost
        prob += total_prod_cost + total_ship_cost
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Calculate actual RMSSE
        utils = [utilization[c, m].value() for c in countries for m in months]
        avg = avg_util.value()
        rmsse = np.sqrt(np.mean([(u - avg)**2 for u in utils]))
        
        # Calculate actual costs
        actual_prod_cost = sum(
            prod_cost_dict.get((c, p), 0) * production[c, p, m].value()
            for c in countries for p in products for m in months
            if (c, p) in prod_cost_dict and production[c, p, m].value() > 0
        )
        
        actual_ship_cost = sum(
            ship_cost_dict.get((o, d), 0) * shipments[o, d, p, m].value()
            for o in countries for d in countries if o != d
            for p in products for m in months
            if (o, d) in ship_cost_dict and shipments[o, d, p, m].value() > 0
        )
        
        return {
            'rmsse': rmsse,
            'production': {(c, p, m): production[c, p, m].value() for c in countries for p in products for m in months},
            'shipments': {(o, d, p, m): shipments[o, d, p, m].value() for o in countries for d in countries 
                          if o != d for p in products for m in months},
            'production_cost': actual_prod_cost,
            'shipping_cost': actual_ship_cost,
            'total_cost': actual_prod_cost + actual_ship_cost,
            'average_utilization': avg
        }

    # Step 3: Execute the optimization process
    # First optimize for balance
    balance_result = solve_for_balance()
    optimal_rmsse = balance_result['rmsse']
    print(f"Optimal RMSSE: {optimal_rmsse}")

    # Then optimize for cost with a constraint on balance
    # Try different tolerance levels
    tolerance_options = [0.01, 0.03, 0.05, 0.07, 0.10]
    best_result = None
    best_cost = float('inf')

    for tolerance in tolerance_options:
        cost_result = solve_for_cost(optimal_rmsse, tolerance)
        print(f"Tolerance: {tolerance}, RMSSE: {cost_result['rmsse']}, Total Cost: {cost_result['total_cost']}")
        
        if cost_result['total_cost'] < best_cost:
            best_cost = cost_result['total_cost']
            best_result = cost_result

    # Output the best solution
    print("\nBest Solution:")
    print(f"RMSSE: {best_result['rmsse']}")
    print(f"Production Cost: {best_result['production_cost']}")
    print(f"Shipping Cost: {best_result['shipping_cost']}")
    print(f"Total Cost: {best_result['total_cost']}")
    print(f"Average Utilization: {best_result['average_utilization']}")

    # Save the results to the specified directory
    file_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(file_dir, "../solutions")
    os.makedirs(output_dir, exist_ok=True)

    # Create production plan dataframe
    production_plan = []
    production_plan.extend(
        {'Country': c, 'Product': p, 'Month': m, 'Quantity': int(qty)}
        for (c, p, m), qty in best_result['production'].items()
        if qty > 0
    )
    
    # Create shipments dataframe
    shipments_plan = []
    shipments_plan.extend(
        {'Origin': o, 'Destination': d, 'Product': p, 'Month': m, 'Quantity': int(qty)}
        for (o, d, p, m), qty in best_result['shipments'].items()
        if qty > 0
    )
    
    # Save to CSV files
    pd.DataFrame(production_plan).to_csv(os.path.join(output_dir, '03_output_productionPlan_4095_v3.csv'), index=False)
    pd.DataFrame(shipments_plan).to_csv(os.path.join(output_dir, '03_output_shipments_4095_v3.csv'), index=False)
    
    print("Production plan and shipments files have been generated.")

if __name__ == "__main__":
    demand_file = sys.argv[1]
    capacity_file = sys.argv[2]
    production_cost_file = sys.argv[3]
    shipment_cost_file = sys.argv[4]
    main(demand_file, capacity_file, production_cost_file, shipment_cost_file)
