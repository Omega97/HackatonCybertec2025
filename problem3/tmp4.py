import pandas as pd
import time
import sys
import os
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus, value

def main(demand_file, capacity_file, production_cost_file, shipment_cost_file):
    # Start timing execution
    start_time = time.time()

    # Load data
    demand_df = pd.read_csv(demand_file)
    capacity_df = pd.read_csv(capacity_file)
    production_cost_df = pd.read_csv(production_cost_file)
    shipments_cost_df = pd.read_csv(shipment_cost_file)

    # Extract unique values
    countries = capacity_df['Country'].unique()
    products = demand_df['Product'].unique()
    months = demand_df['Month'].unique()
    destination_countries = demand_df['Country'].unique()

    # Create dictionaries for lookup
    capacity_dict = capacity_df.set_index('Country')['Monthly Capacity'].to_dict()

    production_cost_dict = {
        (row['Country'], row['Product']): row['Unit Cost']
        for _, row in production_cost_df.iterrows()
    }
    shipment_cost_dict = {
        (row['Origin'], row['Destination']): row['Unit Cost']
        for _, row in shipments_cost_df.iterrows()
    }

    demand_dict = {
        (row['Country'], row['Product'], row['Month']): row['Quantity']
        for _, row in demand_df.iterrows()
    }

    # Create optimization model
    model = LpProblem("Production_and_Transfer_Optimization", LpMinimize)

    # Decision Variables
    # production[country, product, month] = quantity produced
    production = LpVariable.dicts("Production", 
                                [(c, p, m) for c in countries for p in products for m in months], 
                                lowBound=0, 
                                cat='Integer')

    # shipment[origin, destination, product, month] = quantity shipped
    shipment = LpVariable.dicts("Shipment", 
                            [(o, d, p, m) for o in countries for d in destination_countries for p in products for m in months], 
                            lowBound=0, 
                            cat='Integer')

    # Slack variables for demand balance (allows small deviations for cost optimization)
    slack_plus = LpVariable.dicts("SlackPlus", 
                                [(d, p, m) for d in destination_countries for p in products for m in months], 
                                lowBound=0, 
                                cat='Integer')
    slack_minus = LpVariable.dicts("SlackMinus", 
                                [(d, p, m) for d in destination_countries for p in products for m in months], 
                                lowBound=0, 
                                cat='Integer')

    # Penalty for imbalance - make this high to prioritize balance over cost
    balance_penalty = 1000

    # Objective function: Minimize total cost + penalty for imbalance
    model += (
        lpSum([production_cost_dict.get((c, p), 0) * production[(c, p, m)] 
            for c in countries for p in products for m in months if (c, p) in production_cost_dict]) +
        lpSum([shipment_cost_dict.get((o, d), 0) * shipment[(o, d, p, m)]
            for o in countries for d in destination_countries for p in products for m in months if (o, d) in shipment_cost_dict]) +
        balance_penalty * lpSum([slack_plus[(d, p, m)] + slack_minus[(d, p, m)]
                                for d in destination_countries for p in products for m in months])
    )

    # Constraints

    # 1. Capacity constraint for each production facility per month
    for c in countries:
        for m in months:
            model += (
                lpSum([production[(c, p, m)] for p in products]) <= capacity_dict[c],
                f"Capacity_{c}_{m}"
            )

    # 2. Demand satisfaction for each destination, product, and month
    for d in destination_countries:
        for p in products:
            for m in months:
                # What arrives at destination = incoming shipments - outgoing shipments + local production
                model += (
                    lpSum([shipment[(o, d, p, m)] for o in countries if (o, d) in shipment_cost_dict]) - 
                    lpSum([shipment[(d, o, p, m)] for o in countries if (d, o) in shipment_cost_dict]) +
                    production[(d, p, m)] + slack_minus[(d, p, m)] - slack_plus[(d, p, m)] == 
                    demand_dict.get((d, p, m), 0),
                    f"Demand_{d}_{p}_{m}"
                )

    # Optional: Add a constraint to limit the total slack variables
    # This helps ensure balance quality stays high while optimizing costs
    max_total_slack = 1000  # Adjust this value based on your acceptable balance quality
    model += (
        lpSum([slack_plus[(d, p, m)] + slack_minus[(d, p, m)] 
            for d in destination_countries for p in products for m in months]) <= max_total_slack,
        "Total_Slack_Limit"
    )

    # Solve the model
    model.solve(PULP_CBC_CMD(msg=True, timeLimit=600))  # 10-minute time limit

    print(f"Status: {LpStatus[model.status]}")
    print(f"Total Cost: {value(model.objective)}")
    print(f"Execution time: {time.time() - start_time} seconds")

    # Extract results
    production_results = []
    for c in countries:
        for p in products:
            production_results.extend(
                {
                    'Country': c,
                    'Product': p,
                    'Month': m,
                    'Quantity': int(value(production[(c, p, m)])),
                }
                for m in months
                if value(production[(c, p, m)]) > 0
            )
    shipment_results = []
    for o in countries:
        for d in destination_countries:
            for p in products:
                shipment_results.extend(
                    {
                        'Origin': o,
                        'Destination': d,
                        'Product': p,
                        'Month': m,
                        'Quantity': int(value(shipment[(o, d, p, m)])),
                    }
                    for m in months
                    if (o, d) in shipment_cost_dict
                    and value(shipment[(o, d, p, m)]) > 0
                )
    # Set up output directory
    file_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(file_dir, "../solutions")
    os.makedirs(output_dir, exist_ok=True)

    # Save results to CSV
    pd.DataFrame(production_results).to_csv(os.path.join(output_dir, '03_output_productionPlan_4095_v4.csv'), index=False)
    pd.DataFrame(shipment_results).to_csv(os.path.join(output_dir, '03_output_shipments_4095_v4.csv'), index=False)

    # Calculate RMSSE
    total_squared_error = 0
    total_demand = 0
    imbalance_count = 0

    for d in destination_countries:
        for p in products:
            for m in months:
                demand = demand_dict.get((d, p, m), 0)
                total_demand += demand

                received = sum(value(shipment[(o, d, p, m)]) for o in countries if (o, d) in shipment_cost_dict) - \
                        sum(value(shipment[(d, o, p, m)]) for o in countries if (d, o) in shipment_cost_dict) + \
                        value(production[(d, p, m)])

                error = received - demand
                if error != 0:
                    imbalance_count += 1

                if demand != 0:
                    total_squared_error += (error ** 2) / demand
                else:
                    total_squared_error += (error ** 2) / (1 + demand)

    rmsse = (total_squared_error / len(demand_dict)) ** 0.5 if demand_dict else 0

    print("Production plan and shipments files have been generated.")
    print("RMSSE: ", rmsse)
    print("Imbalance count: ", imbalance_count)
    print("Total demand: ", total_demand)
    print("Imbalance count: ", imbalance_count)
    print("Total demand: ", total_demand)

if __name__ == "__main__":
    
    demand_file = sys.argv[1]
    capacity_file = sys.argv[2]
    production_cost_file = sys.argv[3]
    shipment_cost_file = sys.argv[4]
    
    main(demand_file, capacity_file, production_cost_file, shipment_cost_file)