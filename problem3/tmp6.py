import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# -----------------------------
# Data Loading Function
# -----------------------------
def load_data(demand_file, capacity_file, production_cost_file, shipment_cost_file):
    df_demand = pd.read_csv(demand_file)       # Columns: Country, Product, Month, Quantity (demand)
    df_capacity = pd.read_csv(capacity_file)     # Columns: Country, Monthly Capacity
    df_prod_cost = pd.read_csv(production_cost_file)  # Columns: Country, Product, Unit Cost
    df_ship_cost = pd.read_csv(shipment_cost_file)    # Columns: Origin, Destination, Unit Cost

    facilities = df_capacity["Country"].unique().tolist()
    markets = df_demand["Country"].unique().tolist()
    products = df_demand["Product"].unique().tolist()
    months = df_demand["Month"].unique().tolist()

    capacity = df_capacity.set_index("Country")["Monthly Capacity"].to_dict()
    
    demand = {}
    for _, row in df_demand.iterrows():
        key = (row["Country"], row["Product"], row["Month"])
        demand[key] = row["Quantity"]
    
    prod_cost = df_prod_cost.set_index(["Country", "Product"])["Unit Cost"].to_dict()
    ship_cost = df_ship_cost.set_index(["Origin", "Destination"])["Unit Cost"].to_dict()
    
    return facilities, markets, products, months, capacity, demand, prod_cost, ship_cost

# -----------------------------
# Differentiable Optimization Model
# -----------------------------
class ProductionShipmentModel(nn.Module):
    def __init__(self, facilities, markets, products, months):
        super(ProductionShipmentModel, self).__init__()
        self.facilities = facilities
        self.markets = markets
        self.products = products
        self.months = months

        # Decision variables as free parameters (later transformed to be nonnegative)
        self.x_params = nn.ParameterDict()  # Production variables x[i,p,t]
        self.y_params = nn.ParameterDict()  # Shipment variables y[i,j,p,t]

        for i in facilities:
            for p in products:
                for t in months:
                    key = f"x_{i}_{p}_{t}"
                    self.x_params[key] = nn.Parameter(torch.tensor(0.0))
                    
        for i in facilities:
            for j in markets:
                for p in products:
                    for t in months:
                        key = f"y_{i}_{j}_{p}_{t}"
                        self.y_params[key] = nn.Parameter(torch.tensor(0.0))
                        
        # Softplus transformation ensures non-negativity.
        self.softplus = nn.Softplus()

    def forward(self):
        # Apply softplus so that x and y are nonnegative.
        x = { key: self.softplus(param) for key, param in self.x_params.items() }
        y = { key: self.softplus(param) for key, param in self.y_params.items() }
        return x, y

# -----------------------------
# Loss Function: Business Cost + Prediction Error
# -----------------------------
def compute_loss(x, y, facilities, markets, products, months, capacity, demand, prod_cost, ship_cost,
                 penalty_weight=1000.0, error_weight=1.0):
    """
    Total loss is:
      total_cost + error_weight * total_error 
      + penalty_weight * (capacity violations + flow conservation deviations)
    """
    cost = 0.0
    penalty = 0.0
    error_term = 0.0

    # Business cost: production cost
    for i in facilities:
        for p in products:
            for t in months:
                key = f"x_{i}_{p}_{t}"
                cost += prod_cost.get((i, p), 0) * x[key]
                
    # Business cost: shipment cost
    for i in facilities:
        for j in markets:
            for p in products:
                for t in months:
                    key = f"y_{i}_{j}_{p}_{t}"
                    cost += ship_cost.get((i, j), 0) * y[key]
                    
    # Penalty: Capacity constraint (per facility and month)
    for i in facilities:
        for t in months:
            prod_sum = sum(x[f"x_{i}_{p}_{t}"] for p in products)
            cap = capacity[i]
            penalty += penalty_weight * torch.relu(prod_sum - cap)
            
    # Penalty: Flow conservation (production equals shipments out from a facility for each product, month)
    for i in facilities:
        for p in products:
            for t in months:
                prod_val = x[f"x_{i}_{p}_{t}"]
                shipped = sum(y[f"y_{i}_{j}_{p}_{t}"] for j in markets)
                penalty += penalty_weight * torch.abs(prod_val - shipped)
                
    # Error: Demand satisfaction error for each market, product, month
    # Here, we compute the error according to the evaluation metric.
    for j in markets:
        for p in products:
            for t in months:
                # Total shipped to market j for product p in month t.
                shipment_sum = sum(y[f"y_{i}_{j}_{p}_{t}"] for i in facilities)
                d = demand.get((j, p, t), 0)
                if d != 0:
                    error_term += ((shipment_sum - d)**2) / d
                else:
                    error_term += ((shipment_sum - d)**2)  # denominator 1 when demand is 0

    total_loss = cost + penalty + error_weight * error_term
    return total_loss

# -----------------------------
# Main Optimization Routine
# -----------------------------
def optimize_model(demand_file, capacity_file, production_cost_file, shipment_cost_file,
                   num_iterations=5000, lr=0.01, error_weight=1.0):
    facilities, markets, products, months, capacity, demand, prod_cost, ship_cost = load_data(
        demand_file, capacity_file, production_cost_file, shipment_cost_file
    )
    
    model = ProductionShipmentModel(facilities, markets, products, months)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        x, y = model()
        loss = compute_loss(x, y, facilities, markets, products, months,
                            capacity, demand, prod_cost, ship_cost,
                            penalty_weight=1000.0, error_weight=error_weight)
        loss.backward()
        optimizer.step()
        
        if iteration % 500 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")
    
    x_final, y_final = model()
    return x_final, y_final, facilities, markets, products, months

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Expected command-line arguments:
    #   1. Demand file, 2. Capacity file, 3. Production cost file, 4. Shipment cost file
    import sys
    demand_file = sys.argv[1]
    capacity_file = sys.argv[2]
    production_cost_file = sys.argv[3]
    shipment_cost_file = sys.argv[4]
    
    # error_weight is a tunable parameter to balance cost minimization and error minimization.
    x_sol, y_sol, facilities, markets, products, months = optimize_model(
        demand_file, capacity_file, production_cost_file, shipment_cost_file,
        num_iterations=5000, lr=0.01, error_weight=1.0
    )
    
    # Save production plan as CSV.
    prod_plan = []
    for i in facilities:
        for p in products:
            for t in months:
                key = f"x_{i}_{p}_{t}"
                qty = x_sol[key].detach().cpu().item()
                prod_plan.append({"Country": i, "Product": p, "Month": t, "Quantity": qty})
    df_prod_plan = pd.DataFrame(prod_plan)
    df_prod_plan.to_csv("03_output_productionPlan_4095_v6.csv", index=False)
    
    # Save shipments plan as CSV.
    shipments_plan = []
    for i in facilities:
        for j in markets:
            for p in products:
                for t in months:
                    key = f"y_{i}_{j}_{p}_{t}"
                    qty = y_sol[key].detach().cpu().item()
                    if qty > 1e-6:
                        shipments_plan.append({
                            "Origin": i, "Destination": j, "Product": p, "Month": t, "Quantity": qty
                        })
    df_shipments = pd.DataFrame(shipments_plan)
    df_shipments.to_csv("03_output_shipments_4095_v6.csv", index=False)
    
    print("Optimized production and shipment plans (with cost and error minimization) have been saved.")
