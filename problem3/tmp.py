import pandas as pd
import pulp
import sys
import os

def main(demand_file, capacity_file, production_cost_file, shipment_cost_file):
    """
    Main function to solve the production balance problem with cost optimization.
    """

    # Read CSVs
    df_demand = pd.read_csv(demand_file)  # Columns: Country, Product, Month, Quantity (demand)
    df_capacity = pd.read_csv(capacity_file)  # Columns: Country, Monthly Capacity
    df_prod_cost = pd.read_csv(production_cost_file)  # Columns: Country, Product, Unit Cost
    df_ship_cost = pd.read_csv(shipment_cost_file)  # Columns: Origin, Destination, Unit Cost

    # -----------------------------
    # 2. Create Sets and Dictionaries
    # -----------------------------
    facilities = df_capacity["Country"].unique().tolist()  # Production locations
    markets = df_demand["Country"].unique().tolist()         # Destination markets
    products = df_demand["Product"].unique().tolist()          # Products
    months = df_demand["Month"].unique().tolist()              # Months

    capacity = df_capacity.set_index("Country")["Monthly Capacity"].to_dict()

    demand = {}
    for _, row in df_demand.iterrows():
        key = (row["Country"], row["Product"], row["Month"])
        demand[key] = row["Quantity"]

    prod_cost = df_prod_cost.set_index(["Country", "Product"])["Unit Cost"].to_dict()
    ship_cost = df_ship_cost.set_index(["Origin", "Destination"])["Unit Cost"].to_dict()

    # -----------------------------
    # 3. Define the Optimization Model
    # -----------------------------
    model = pulp.LpProblem("Production_Cost_Optimization", pulp.LpMinimize)

    # Decision Variables:
    x = {}
    for i in facilities:
        for p in products:
            for t in months:
                var_name = f"x_{i}_{p}_{t}"
                x[(i, p, t)] = pulp.LpVariable(var_name, lowBound=0, cat="Continuous")

    y = {}
    for i in facilities:
        for j in markets:
            for p in products:
                for t in months:
                    var_name = f"y_{i}_{j}_{p}_{t}"
                    y[(i, j, p, t)] = pulp.LpVariable(var_name, lowBound=0, cat="Continuous")

    # -----------------------------
    # 4. Define Adjusted Shipment Cost Function
    # -----------------------------
    # Penalty factor for non-local shipments
    local_penalty = -10  # Adjust this value as needed

    def adjusted_ship_cost(i, j):
        base_cost = ship_cost.get((i, j), 0)
        return base_cost + local_penalty if i != j else base_cost

    # Optional weights for production and shipment cost components
    production_weight = 1.0
    shipment_weight = 1.0

    # -----------------------------
    # 5. Objective Function: Minimize Total Cost
    # -----------------------------
    model += (
        production_weight * pulp.lpSum(
            prod_cost.get((i, p), 0) * x[(i, p, t)]
            for i in facilities for p in products for t in months
        )
        +
        shipment_weight * pulp.lpSum(
            adjusted_ship_cost(i, j) * y[(i, j, p, t)]
            for i in facilities for j in markets for p in products for t in months
        )
    ), "Total_Cost"

    # -----------------------------
    # 6. Constraints
    # -----------------------------
    for i in facilities:
        for t in months:
            model += (pulp.lpSum(x[(i, p, t)] for p in products) <= capacity[i],
                      f"Capacity_{i}_{t}")

    for j in markets:
        for p in products:
            for t in months:
                d = demand.get((j, p, t), 0)
                model += (pulp.lpSum(y[(i, j, p, t)] for i in facilities) == d,
                          f"Demand_{j}_{p}_{t}")

    for i in facilities:
        for p in products:
            for t in months:
                model += (x[(i, p, t)] == pulp.lpSum(y[(i, j, p, t)] for j in markets),
                          f"Flow_{i}_{p}_{t}")

    # -----------------------------
    # 7. Solve the Model
    # -----------------------------
    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    print("Status:", pulp.LpStatus[model.status])
    print("Objective (Total Cost):", pulp.value(model.objective))

    # -----------------------------
    # 8. Extract Results and Write Output Files
    # -----------------------------
    file_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(file_dir, "../solutions")
    os.makedirs(output_dir, exist_ok=True)

    prod_plan_data = []
    for i in facilities:
        for p in products:
            for t in months:
                qty = x[(i, p, t)].varValue
                if qty is None:
                    qty = 0
                prod_plan_data.append({"Country": i, "Product": p, "Month": t, "Quantity": qty})
    df_prod_plan = pd.DataFrame(prod_plan_data)
    df_prod_plan.to_csv(os.path.join(output_dir, "03_output_productionPlan_4095_v2.csv"), index=False)

    shipments_data = []
    for i in facilities:
        for j in markets:
            for p in products:
                for t in months:
                    qty = y[(i, j, p, t)].varValue
                    if qty is None:
                        qty = 0
                    if qty > 0:
                        shipments_data.append({
                            "Origin": i,
                            "Destination": j,
                            "Product": p,
                            "Month": t,
                            "Quantity": qty,
                        })
    df_shipments = pd.DataFrame(shipments_data)
    df_shipments.to_csv(os.path.join(output_dir, "03_output_shipments_4095_v2.csv"), index=False)

    print("Production plan and shipments files have been generated.")

if __name__ == "__main__":
    demand_file = sys.argv[1]
    capacity_file = sys.argv[2]
    production_cost_file = sys.argv[3]
    shipment_cost_file = sys.argv[4]
    main(demand_file, capacity_file, production_cost_file, shipment_cost_file)

