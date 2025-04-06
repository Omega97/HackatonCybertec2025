import pandas as pd
import pulp
import sys
import os


def main(demand_file, capacity_file, production_cost_file, shipment_cost_file):
    """
    Main function to solve the production balance problem.
    """

    # Input data
    df_demand = pd.read_csv(demand_file)
    df_capacity = pd.read_csv(capacity_file)
    df_production_cost = pd.read_csv(production_cost_file)
    df_shipment_cost = pd.read_csv(shipment_cost_file)

    # Input variables
    facilities = df_capacity["Country"].unique().tolist()  # Production locations
    markets = df_demand["Country"].unique().tolist()  # Destination markets
    products = df_demand["Product"].unique().tolist()  # Products
    months = df_demand["Month"].unique().tolist()  # Months
    capacity = df_capacity.set_index("Country")["Monthly Capacity"].to_dict()
    demand = df_demand.set_index(["Country", "Product", "Month"])["Quantity"].to_dict()
    demand = {k: v for k, v in demand.items() if v > 0}
    production_cost = df_production_cost.set_index(["Country", "Product"])[
        "Unit Cost"
    ].to_dict()

    shipment_cost = df_shipment_cost.set_index(["Origin", "Destination"])[
        "Unit Cost"
    ].to_dict()

    # Optimization model
    model = pulp.LpProblem("Production_Cost_Optimization", pulp.LpMinimize)

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

    # Objective function: Minimize Total Cost
    model += (
        (
            pulp.lpSum(
                production_cost.get((i, p), 0) * x[(i, p, t)]
                for i in facilities
                for p in products
                for t in months
            )
            + pulp.lpSum(
                shipment_cost.get((i, j), 0) * y[(i, j, p, t)]
                for i in facilities
                for j in markets
                for p in products
                for t in months
            )
        ),
        "Total_Cost",
    )

    # Constraints

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
                    pulp.lpSum(y[(i, j, p, t)] for j in markets) == x[(i, p, t)],
                    f"Flow_{i}_{p}_{t}",
                )

    # Solve the model
    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    print("Status:", pulp.LpStatus[model.status])
    print("Objective (Total Shipment Cost):", pulp.value(model.objective))

    # Results
    file_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(file_dir, "../solutions")
    os.makedirs(output_dir, exist_ok=True)

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
    df_prod_plan.to_csv(
        os.path.join(output_dir, "03_output_productionPlan_4095.csv"), index=False
    )

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
