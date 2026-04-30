import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Hardcoded values from model
# -----------------------------

sqft = 1000

cost_3d_per_sqft = 159.10
cost_conv_per_sqft = 136.74

initial_3d = cost_3d_per_sqft * sqft
initial_conv = cost_conv_per_sqft * sqft

extra_cost = initial_3d - initial_conv

conv_maintenance = 0.006
conv_repair = 0.25

disaster_probability = 0.15
discount_rate = 0.05
years = 30

# Present worth factor
pw_factor = (1 - (1 + discount_rate)**(-years)) / discount_rate

# Conventional annual costs
maint_conv_cost = initial_conv * conv_maintenance
repair_conv_cost = initial_conv * conv_repair * disaster_probability

# -----------------------------
# 1. Vary repair % for 3D
# -----------------------------

repair_rates = np.linspace(0.05, 0.25, 100)

bcr_repair = []
pw_repair = []

maint_3d_fixed = 0.004  # assumed maintenance rate for 3D

for r in repair_rates:

    maint_3d_cost = initial_3d * maint_3d_fixed
    repair_3d_cost = initial_3d * r * disaster_probability

    annual_savings = (maint_conv_cost + repair_conv_cost) - (maint_3d_cost + repair_3d_cost)

    pw_savings = annual_savings * pw_factor
    bcr = pw_savings / extra_cost

    pw_repair.append(pw_savings)
    bcr_repair.append(bcr)

# -----------------------------
# Plot BCR vs Repair %
# -----------------------------

plt.figure()
plt.plot(repair_rates * 100, bcr_repair)
plt.axhline(1, linestyle="--")
plt.xlabel("3D Printed Repair Cost (%)")
plt.ylabel("BCR")
plt.title("BCR vs 3D Printed Repair Cost")
plt.grid(True)
plt.show()

# -----------------------------
# Plot PW vs Repair %
# -----------------------------

plt.figure()
plt.plot(repair_rates * 100, pw_repair)
plt.axhline(0, linestyle="--")
plt.xlabel("3D Printed Repair Cost (%)")
plt.ylabel("PW of Savings ($)")
plt.title("Present Worth of Savings vs 3D Printed Repair Cost")
plt.grid(True)
plt.show()

# -----------------------------
# 2. Vary maintenance rate
# -----------------------------

maintenance_rates = np.linspace(0.001, 0.006, 100)

bcr_maint = []
pw_maint = []

repair_3d_fixed = 0.15

for m in maintenance_rates:

    maint_3d_cost = initial_3d * m
    repair_3d_cost = initial_3d * repair_3d_fixed * disaster_probability

    annual_savings = (maint_conv_cost + repair_conv_cost) - (maint_3d_cost + repair_3d_cost)

    pw_savings = annual_savings * pw_factor
    bcr = pw_savings / extra_cost

    pw_maint.append(pw_savings)
    bcr_maint.append(bcr)

# -----------------------------
# Plot BCR vs Maintenance %
# -----------------------------

plt.figure()
plt.plot(maintenance_rates * 100, bcr_maint)
plt.axhline(1, linestyle="--")
plt.xlabel("3D Printed Maintenance Rate (%)")
plt.ylabel("BCR")
plt.title("BCR vs 3D Printed Maintenance Rate")
plt.grid(True)
plt.show()

# -----------------------------
# Plot PW vs Maintenance %
# -----------------------------

plt.figure()
plt.plot(maintenance_rates * 100, pw_maint)
plt.axhline(0, linestyle="--")
plt.xlabel("3D Printed Maintenance Rate (%)")
plt.ylabel("PW of Savings ($)")
plt.title("Present Worth of Savings vs 3D Printed Maintenance Rate")
plt.grid(True)
plt.show()