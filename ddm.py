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

# Correct decisions for each image (assuming 0 for woman, 1 for lion)
correct_decisions = [0, 0, 1, 1, 1]  # Example mapping

def simulate_ddm(v, a, z, t_0, correct_decision, n_trials=1000, dt=0.01):
    rt = []  # Reaction times
    decision = []  # Decisions (1 for upper boundary, 0 for lower)
    correct = []  # Correctness of each decision

    for _ in range(n_trials):
        evidence = z
        time = 0

        while 0 < evidence < a:
            evidence += v * dt + np.random.randn() * np.sqrt(dt)
            time += dt

            if evidence <= 0 or evidence >= a:
                break

        rt.append(time + t_0)
        decision_made = 1 if evidence >= a else 0
        decision.append(decision_made)
        correct.append(decision_made == correct_decision)

    return np.array(rt), np.array(decision), np.array(correct)

def simulate_spectrum(ddm_params, correct_decisions, n_trials=1000, t_0=t_0):
    results = []
    for v, correct_decision in zip(ddm_params, correct_decisions):
        rt, decision, correct = simulate_ddm(v, a, z, t_0, correct_decision, n_trials)
        results.append((rt, decision, correct))
    return results

# Simulating data for the spectrum of images
spectrum_results = simulate_spectrum(drift_rates, correct_decisions)


# Here we simulate the data but with a cafeine boost
# Increased drift rates and reduced non-decision time to simulate caffeine boost
drift_rates_boosted = [rate * 1.5 for rate in drift_rates]  
t_0_boosted = 0.2 

# Simulating data with caffeine boost
spectrum_results_boosted = simulate_spectrum(drift_rates_boosted, correct_decisions, n_trials=1000, t_0=t_0_boosted)




# Plotting the reaction times for each image in the spectrum with and without caffeine boost
plt.figure(figsize=(12, 8))

# Without caffeine
plt.subplot(2, 1, 1)
for i, (rt, decision, _) in enumerate(spectrum_results):
    plt.hist(rt, bins=30, alpha=0.5, label=f'Image {i+1} (No Caffeine)')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Count')
plt.title('Simulated DDM Data without Caffeine')
plt.legend()

# With caffeine
plt.subplot(2, 1, 2)
for i, (rt, decision, _) in enumerate(spectrum_results_boosted):
    plt.hist(rt, bins=30, alpha=0.5, label=f'Image {i+1} (With Caffeine)')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Count')
plt.title('Simulated DDM Data with Caffeine Boost')
plt.legend()

plt.tight_layout()
plt.show()

# Computing total accuracy for each image in both conditions (with and without caffeine)
accuracies_no_caffeine = [np.mean(correct) for _, _, correct in spectrum_results]
accuracies_with_caffeine = [np.mean(correct) for _, _, correct in spectrum_results_boosted]


plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), accuracies_no_caffeine, marker='o', label='No Caffeine')
plt.plot(range(1, 6), accuracies_with_caffeine, marker='o', label='With Caffeine')
plt.xlabel('Image Number')
plt.ylabel('Accuracy')
plt.title('Accuracy of Decision-Making with and without Caffeine Boost')
plt.xticks(range(1, 6))
plt.legend()
plt.show()