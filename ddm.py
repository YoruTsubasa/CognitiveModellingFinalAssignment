import numpy as np
import matplotlib.pyplot as plt
import hddm as hddm_df
import pandas as pd

# Parameters for the Drift Diffusion Model
a = 0.2  # Boundary separation
t_0 = 0.3  # Non-decision time
z = a / 2  # Starting point (middle)

# Drift rates for each of the five images
# Higher drift rates for clear images (woman and lion), lower for ambiguous ones
drift_rates = [0.08, 0.02, 0.01, -0.01, -0.08]  # Example drift rate
cafeine_drift_rates = []

# Function to simulate the decision-making process
def simulate_ddm(v, a, z, t_0, n_trials=1000, dt=0.01):
    rt = []  # Reaction times
    decision = []  # Decisions (1 for upper boundary, 0 for lower)

    for _ in range(n_trials):
        evidence = z
        time = 0

        while 0 < evidence < a:
            # Update evidence with drift and noise
            evidence += v * dt + np.random.randn() * np.sqrt(dt)
            time += dt

            if evidence <= 0 or evidence >= a:
                break

        rt.append(time + t_0)  # Add non-decision time
        decision.append(1 if evidence >= a else 0)  # Decision made

    return np.array(rt), np.array(decision)

# Simulating the decision-making process for each image
def simulate_spectrum(ddm_params, n_trials=1000, t_0=t_0):
    results = []
    for v in ddm_params:
        rt, decision = simulate_ddm(v, a, z, t_0, n_trials)
        results.append((rt, decision))
    return results

# Simulating data for the spectrum of images
spectrum_results = simulate_spectrum(drift_rates)


# Here we simulate the data but with a cafeine boost
# Increased drift rates and reduced non-decision time to simulate caffeine boost
drift_rates_boosted = [rate * 1.5 for rate in drift_rates]  
t_0_boosted = 0.2 

# Simulating data with caffeine boost
spectrum_results_boosted = simulate_spectrum(drift_rates_boosted, n_trials=1000, t_0=t_0_boosted)



# Plotting the reaction times for each image in the spectrum with and without caffeine boost
plt.figure(figsize=(12, 8))

# Without caffeine
plt.subplot(2, 1, 1)
for i, (rt, decision) in enumerate(spectrum_results):
    plt.hist(rt, bins=30, alpha=0.5, label=f'Image {i+1} (No Caffeine)')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Count')
plt.title('Simulated DDM Data without Caffeine')
plt.legend()

# With caffeine
plt.subplot(2, 1, 2)
for i, (rt, decision) in enumerate(spectrum_results_boosted):
    plt.hist(rt, bins=30, alpha=0.5, label=f'Image {i+1} (With Caffeine)')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Count')
plt.title('Simulated DDM Data with Caffeine Boost')
plt.legend()

plt.tight_layout()
plt.show()
