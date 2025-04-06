import pandas as pd
import pulp
import sys
import os


def main(demand_file, capacity_file):
    """
    Main function to solve the production balance problem.
    """

    # -----------------------------
    # 1. Read Input Data
    # -----------------------------

    # Read market demand: columns: Country, Product, Month, Quantity
    df_demand = pd.read_csv(demand_file)
    # Read facility capacities: columns: Country, Monthly Capacity
    df_capacity = pd.read_csv(capacity_file)

    # -----------------------------
    # 2. Create Sets and Parameters
    # -----------------------------
    facilities = df_capacity["Country"].unique().tolist()  # Production locations
    markets = df_demand["Country"].unique().tolist()         # Destination markets
    products = df_demand["Product"].unique().tolist()          # Products
    months = df_demand["Month"].unique().tolist()              # Months
    capacity = df_capacity.set_index("Country")["Monthly Capacity"].to_dict()
    demand = df_demand.set_index(["Country", "Product", "Month"])["Quantity"].to_dict()
    demand = {k: v for k, v in demand.items() if v > 0}

    # -----------------------------
    # 3. Define the Optimization Model
    # -----------------------------
    model = pulp.LpProblem("Production_Balance", pulp.LpMinimize)

    # Decision variables:
    # x[i, p, t]: production at facility i for product p in month t
    x = {}
    for i in facilities:
        for p in products:
            for t in months:
                x[(i, p, t)] = pulp.LpVariable(
                    f"x_{i}_{p}_{t}", lowBound=0, cat="Continuous"
                )

    # y[i, j, p, t]: shipments from facility i (origin) to market j (destination) for product p in month t
    y = {}
    for i in facilities:
        for j in markets:
            for p in products:
                for t in months:
                    y[(i, j, p, t)] = pulp.LpVariable(
                        f"y_{i}_{j}_{p}_{t}", lowBound=0, cat="Continuous"
                    )

    # -----------------------------
    # 4. Objective Function
    # -----------------------------
    # Minimize total shipment "cost" to encourage local fulfilment.
    model += (
        pulp.lpSum(
            y[(i, j, p, t)] * (1 if i != j else 0) # Local shipment costs 0
            for i in facilities
            for j in markets
            for p in products
            for t in months
        ),
        "Total_Shipment_Cost",
    )

    # -----------------------------
    # 5. Constraints
    # -----------------------------

    # (a) Facility capacity constraint for each facility and month
    for i in facilities:
        for t in months:
            model += (
                pulp.lpSum(x[(i, p, t)] for p in products) <= capacity[i],
                f"Capacity_{i}_{t}",
            )

    # (b) Demand satisfaction: for each market, product, and month, sum of shipments equals demand.
    for j in markets:
        for p in products:
            for t in months:
                model += (
                    pulp.lpSum(y[(i, j, p, t)] for i in facilities)
                    == demand.get((j, p, t), 0),
                    f"Demand_{j}_{p}_{t}",
                )

# (c) Shipment doesn't exceed production: for each facility, product, month.
    for i in facilities:
        for p in products:
            for t in months:
                model += (
                    pulp.lpSum(y[(i, j, p, t)] for j in markets) <= x[(i, p, t)],
                    f"Flow_{i}_{p}_{t}",
                )

    # -----------------------------
    # 6. Solve the Model
    # -----------------------------
    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    print("Status:", pulp.LpStatus[model.status])
    print("Objective (Total Shipment Cost):", pulp.value(model.objective))

    # -----------------------------
    # 7. Extract Results and Write Output Files
    # -----------------------------
    file_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(file_dir, "../solutions")
    os.makedirs(output_dir, exist_ok=True)

    # Create Production Plan DataFrame: columns [Country, Product, Month, Quantity]
    prod_plan = []
    for i in facilities:
        for p in products:
            for t in months:
                qty = x[(i, p, t)].varValue
                if qty > 0:
                    prod_plan.append(
                        {"Country": i, "Product": p, "Month": t, "Quantity": int(qty)}
                    )
    df_prod_plan = pd.DataFrame(prod_plan)
    # Write production plan to CSV.
    df_prod_plan.to_csv(
        os.path.join(output_dir, "02_output_productionPlan_4095.csv"), index=False
    )

    # Create Shipments DataFrame: columns [Origin, Destination, Product, Month, Quantity]
    shipments = []
    for i in facilities:
        for j in markets:
            for p in products:
                for t in months:
                    qty = y[(i, j, p, t)].varValue
                    if qty > 0:
                        shipments.append(
                            {
                                "Origin": i,
                                "Destination": j,
                                "Product": p,
                                "Month": t,
                                "Quantity": int(qty),
                            }
                        )
    df_shipments = pd.DataFrame(shipments)
    # Write shipments to CSV.
    df_shipments.to_csv(
        os.path.join(output_dir, "02_output_shipments_4095.csv"), index=False
    )

    print("Production plan and shipments files have been generated.")


if __name__ == "__main__":
    demand_file = sys.argv[1]
    capacity_file = sys.argv[2]
    main(demand_file, capacity_file)
