import pandas as pd
import pulp
import sys
import os


def main(demand_file, capacity_file, production_cost_file, shipment_cost_file):
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
    # Read production costs: columns: Country, Product, Month, Production Cost
    df_production_cost = pd.read_csv(production_cost_file)
    # Read shipment costs: columns: Origin, Destination, Product, Month, Shipment Cost
    df_shipment_cost = pd.read_csv(shipment_cost_file)

    # -----------------------------
    # 2. Create Sets and Parameters
    # -----------------------------
    # Production facilities: from capacity file (each country is a production facility)
    facilities = df_capacity["Country"].unique().tolist()

    # Markets (destination countries) from demand file.
    markets = df_demand["Country"].unique().tolist()

    # Products and months from demand file.
    products = df_demand["Product"].unique().tolist()
    months = df_demand["Month"].unique().tolist()

    # Create dictionaries for capacity and demand.
    capacity = df_capacity.set_index("Country")["Monthly Capacity"].to_dict()

    # Demand dictionary: key = (market, product, month)
    demand = {}
    for _, row in df_demand.iterrows():
        key = (row["Country"], row["Product"], row["Month"])
        demand[key] = row["Quantity"]

    # Constant monthly production cost dictionary: key = (facility, product)
    production_cost = {}
    for _, row in df_production_cost.iterrows():
        key = (row["Country"], row["Product"])
        production_cost[key] = row["Unit Cost"]

    # Shipment cost dictionary: key = (origin, destination)
    shipment_cost = {}
    for _, row in df_shipment_cost.iterrows():
        key_one_way = (row["Origin"], row["Destination"])
        shipment_cost[key_one_way] = row["Unit Cost"]
        key_reverse = (row["Destination"], row["Origin"])
        shipment_cost[key_reverse] = row["Unit Cost"]
    for i in facilities:
        key = (i, i)
        shipment_cost[key] = 0  # No cost to ship from one facility to itself

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
    # Minimize total cost (shipment + production).
    model += (
        pulp.lpSum(
            shipment_cost[(i, j)] * y[(i, j, p, t)]
            for i in facilities
            for j in markets
            for p in products
            for t in months
        )
        + pulp.lpSum(
            production_cost[(i, p)] * x[(i, p, t)]
            for i in facilities
            for p in products
            for t in months
        ),
        "Total_Cost",
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
                # If there is no demand entry for (j,p,t) we assume zero demand.
                d = demand.get((j, p, t), 0)
                model += (
                    pulp.lpSum(y[(i, j, p, t)] for i in facilities) == d,
                    f"Demand_{j}_{p}_{t}",
                )

    # (c) Production equals shipments out: for each facility, product, month.
    for i in facilities:
        for p in products:
            for t in months:
                model += (
                    x[(i, p, t)] == pulp.lpSum(y[(i, j, p, t)] for j in markets),
                    f"Flow_{i}_{p}_{t}",
                )

    # -----------------------------
    # 6. Solve the Model
    # -----------------------------
    solver = pulp.PULP_CBC_CMD(msg=True)
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
                # Optionally, round or convert to int if required.
                if qty is None:
                    qty = 0
                prod_plan.append(
                    {"Country": i, "Product": p, "Month": t, "Quantity": int(qty)}
                )
    df_prod_plan = pd.DataFrame(prod_plan)
    # Write production plan to CSV.
    df_prod_plan.to_csv(
        os.path.join(output_dir, "03_output_productionPlan_4095.csv"), index=False
    )

    # Create Shipments DataFrame: columns [Origin, Destination, Product, Month, Quantity]
    shipments = []
    for i in facilities:
        for j in markets:
            for p in products:
                for t in months:
                    qty = y[(i, j, p, t)].varValue
                    if qty is None:
                        qty = 0
                    # Only include positive shipments
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
        os.path.join(output_dir, "03_output_shipments_4095.csv"), index=False
    )

    print("Production plan and shipments files have been generated.")


if __name__ == "__main__":
    demand_file = sys.argv[1]
    capacity_file = sys.argv[2]
    production_cost_file = sys.argv[3]
    shipment_cost_file = sys.argv[4]
    main(demand_file, capacity_file, production_cost_file, shipment_cost_file)
