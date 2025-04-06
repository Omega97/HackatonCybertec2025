################################################################################
# ALL-IN-ONE PYTHON CODE FOR EXERCISE 3 PRODUCTION & TRANSFER COST OPTIMIZATION
# ----------------------------------------------------------------------------
# This script illustrates a Mixed Integer Linear Programming (MILP) approach
# using the PuLP library. It reads:
#   1) capacity.csv        (02_input_capacity.csv)
#   2) demand forecast     (02_input_target.csv)  [from Exercise 2]
#   3) production cost     (03_input_productionCost.csv)
#   4) shipping cost       (02_03_input_shipmentsCost.csv)
# and outputs:
#   - productionPlan CSV   (03_output_productionPlan_CODE.csv)
#   - shipments CSV        (03_output_shipments_CODE.csv)
#
# It also includes optional "slack" (underdelivery and overdelivery) to allow 
# balancing the trade-off between meeting exact demand (RMSSE ~ 0) and minimizing 
# total production + shipping costs. Tweak penalty coefficients to achieve 
# your desired balance.
#
# DISCLAIMER: You must adapt file names, directory paths, or variable names 
# to match your actual data structure. This code is a template to demonstrate 
# the approach and can be refined or extended for your final hackathon solution.
################################################################################

###############################################################################
# 0) INSTALL & IMPORT
###############################################################################
# If running in a fresh environment, you might need to install PuLP:
# %pip install pulp

import pulp
import csv
import sys
import os

###############################################################################
# 1) READ INPUT DATA FROM CSV FILES
###############################################################################

def main(demand_file, capacity_file, production_cost_file, shipment_cost_file):
    # -- 1.1: Read Capacity (Country -> Monthly Capacity)
    capacity_data = {}
    with open(capacity_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            country_name = row['Country']
            cap = float(row['Monthly Capacity'])
            # If capacity is uniform for each month, store it in a dictionary
            capacity_data[country_name] = cap

    # -- 1.2: Read Demand (Country, Product, Month -> Quantity)
    demand_data = {}
    countries_set = set()
    products_set = set()
    months_set = set()

    with open(demand_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = row['Country']
            p = row['Product']
            m = row['Month']
            q = float(row['Quantity'])
            countries_set.add(c)
            products_set.add(p)
            months_set.add(m)
            demand_data[(c,p,m)] = q

    # Turn these sets into sorted lists for convenience
    countries = sorted(list(countries_set))
    products = sorted(list(products_set))
    months = sorted(list(months_set))

    # -- 1.3: Read Production Cost (Country, Product -> UnitCost)
    production_cost_data = {}
    with open(production_cost_file,'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = row['Country']
            p = row['Product']
            cost = float(row['Unit Cost'])
            production_cost_data[(c, p)] = cost

    # -- 1.4: Read Shipment Cost (Origin, Destination -> UnitCost)
    shipment_cost_data = {}
    with open(shipment_cost_file,'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            o = row['Origin']
            d = row['Destination']
            cost = float(row['Unit Cost'])
            shipment_cost_data[(o, d)] = cost

    ###############################################################################
    # 2) SET UP THE MILP MODEL
    ###############################################################################
    model = pulp.LpProblem("MultiFacilityProduction_Optimization", pulp.LpMinimize)

    ###############################################################################
    # 3) DECISION VARIABLES
    ###############################################################################

    # 3.1: Production: x[f, p, m] >= 0
    x_vars = {}
    for f in countries:
        for p in products:
            for m in months:
                var_name = f"x_{f}_{p}_{m}"
                # If you must ensure integer solutions, use cat=pulp.LpInteger
                x_vars[(f,p,m)] = pulp.LpVariable(var_name, lowBound=0, cat=pulp.LpContinuous)

    # 3.2: Shipments: s[o, d, p, m] >= 0
    s_vars = {}
    for o in countries:
        for d in countries:
            for p in products:
                for m in months:
                    var_name = f"s_{o}_{d}_{p}_{m}"
                    s_vars[(o,d,p,m)] = pulp.LpVariable(var_name, lowBound=0, cat=pulp.LpContinuous)

    # 3.3: (Optional) Slack Variables for Underdelivery / Overdelivery
    #      If you want to allow demand mismatch to reduce costs. 
    #      Set BIG_SLACK=False if you want strict equality to demand with zero mismatch.
    BIG_SLACK = True
    under_vars = {}
    over_vars = {}

    if BIG_SLACK:
        for c in countries:
            for p in products:
                for m in months:
                    under_name = f"under_{c}_{p}_{m}"
                    over_name  = f"over_{c}_{p}_{m}"
                    under_vars[(c,p,m)] = pulp.LpVariable(under_name, lowBound=0, cat=pulp.LpContinuous)
                    over_vars[(c,p,m)]  = pulp.LpVariable(over_name,  lowBound=0, cat=pulp.LpContinuous)

    ###############################################################################
    # 4) CONSTRAINTS
    ###############################################################################

    # 4.1: Capacity Constraint: sum of all product production <= capacity
    #      For each facility f, each month m
    for f in countries:
        capacity_f = capacity_data.get(f, 0.0)
        for m in months:
            model += (
                pulp.lpSum(x_vars[(f,p,m)] for p in products) <= capacity_f,
                f"CapacityConstraint_{f}_{m}"
            )

    # 4.2: Flow / Production-Balance Constraint: 
    #      Production in f for p in m must equal total shipped out from f
    for f in countries:
        for p in products:
            for m in months:
                model += (
                    pulp.lpSum(s_vars[(f,d,p,m)] for d in countries) == x_vars[(f,p,m)],
                    f"FlowBalance_{f}_{p}_{m}"
                )

    # 4.3: Demand Constraint: sum of inbound shipments to c == demand +/- slack
    if BIG_SLACK:
        for c in countries:
            for p in products:
                for m in months:
                    demand_cpm = demand_data.get((c,p,m), 0.0)
                    model += (
                        pulp.lpSum(s_vars[(o,c,p,m)] for o in countries)
                        == demand_cpm - under_vars[(c,p,m)] + over_vars[(c,p,m)],
                        f"Demand_{c}_{p}_{m}"
                    )
    else:
        # If you want perfect coverage:
        for c in countries:
            for p in products:
                for m in months:
                    demand_cpm = demand_data.get((c,p,m), 0.0)
                    model += (
                        pulp.lpSum(s_vars[(o,c,p,m)] for o in countries)
                        == demand_cpm,
                        f"Demand_{c}_{p}_{m}"
                    )

    ###############################################################################
    # 5) OBJECTIVE FUNCTION
    ###############################################################################
    # We minimize:
    #   Sum of (Production cost + Shipping cost) + Slack Penalties (if BIG_SLACK = True)
    #
    # If the Hackathon demands near-perfect coverage, we put a very large penalty
    # on underdelivery so we rarely allow shortfalls.

    UNDER_PENALTY = 1000.0
    OVER_PENALTY = 10.0

    obj_terms = []

    # 5.1: Production cost
    for f in countries:
        for p in products:
            for m in months:
                if (f,p) in production_cost_data:
                    prod_cost = production_cost_data[(f,p)]
                    obj_terms.append(prod_cost * x_vars[(f,p,m)])

    # 5.2: Shipping cost
    for o in countries:
        for d in countries:
            if (o,d) in shipment_cost_data:
                cost_od = shipment_cost_data[(o,d)]
                for p in products:
                    for m in months:
                        obj_terms.append(cost_od * s_vars[(o,d,p,m)])

    # 5.3: Slack penalties (only if we allow mismatch)
    if BIG_SLACK:
        for c in countries:
            for p in products:
                for m in months:
                    obj_terms.append(UNDER_PENALTY * under_vars[(c,p,m)])
                    obj_terms.append(OVER_PENALTY  * over_vars[(c,p,m)])

    model.setObjective(pulp.lpSum(obj_terms))

    ###############################################################################
    # 6) SOLVE MODEL
    ###############################################################################
    solution_status = model.solve(pulp.PULP_CBC_CMD(msg=0))  # CBC solver by default
    print("Solver Status:", pulp.LpStatus[solution_status])
    print("Objective Value:", pulp.value(model.objective))

    ###############################################################################
    # 7) EXPORT RESULTS
    #    Write to solutions directory
    ###############################################################################
    
    # Set up output directory
    file_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(file_dir, "../solutions")
    os.makedirs(output_dir, exist_ok=True)
    
    # 7.1: Production Plan
    production_file = os.path.join(output_dir, '03_output_productionPlan_4095_v5.csv')
    with open(production_file, 'w', newline='', encoding='utf-8') as out_prod:
        writer = csv.writer(out_prod)
        writer.writerow(["Country","Product","Month","Quantity"])
        for f_country in countries:
            for p in products:
                for m in months:
                    val = x_vars[(f_country,p,m)].varValue
                    if val is not None and val > 0:
                        # If you need integer solutions, round them:
                        qty_int = int(round(val))
                        writer.writerow([f_country, p, m, qty_int])

    # 7.2: Shipments
    shipments_file = os.path.join(output_dir, '03_output_shipments_4095_v5.csv')
    with open(shipments_file, 'w', newline='', encoding='utf-8') as out_ship:
        writer = csv.writer(out_ship)
        writer.writerow(["Origin","Destination","Product","Month","Quantity"])
        for o in countries:
            for d in countries:
                for p in products:
                    for m in months:
                        val = s_vars[(o,d,p,m)].varValue
                        if val is not None and val > 0:
                            qty_int = int(round(val))
                            writer.writerow([o, d, p, m, qty_int])

    print("Done! The solution has been written to CSV files in the solutions directory.")

if __name__ == "__main__":
    demand_file = sys.argv[1]
    capacity_file = sys.argv[2]
    production_cost_file = sys.argv[3]
    shipment_cost_file = sys.argv[4]
    
    main(demand_file, capacity_file, production_cost_file, shipment_cost_file)
################################################################################
# END OF ALL-IN-ONE SCRIPT
################################################################################
