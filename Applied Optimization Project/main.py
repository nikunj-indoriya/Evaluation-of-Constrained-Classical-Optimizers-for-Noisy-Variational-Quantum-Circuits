import numpy as np
import os
import csv
import time
from qiskit import transpile
from qiskit_aer import Aer
from quantum_circuit import create_hamiltonian, ansatz
from optimizers import (
    gradient_descent_optimizer,
    cobyla_optimizer,
    spsa_optimizer,
    adam_optimizer,
    simulated_annealing_optimizer,
    momentum_gradient_descent_optimizer,
)
from report import generate_report

# Backend for quantum simulation
backend = Aer.get_backend('aer_simulator')

# Function to compute expectation value
def compute_expectation(params, backend, shots=1024):
    hamiltonian = create_hamiltonian()
    qc = ansatz(params)
    qc.save_expectation_value(hamiltonian, range(4))
    qobj = transpile(qc, backend)
    result = backend.run(qobj, shots=shots).result()
    return np.real(result.data(0)["expectation_value"])

# Run optimizer comparison and generate report
def run_comparison_and_report(initial_params):
    optimizers = {
        "Gradient Descent": gradient_descent_optimizer,
        "COBYLA": cobyla_optimizer,
        "SPSA": spsa_optimizer,
        "Adam": adam_optimizer,
        "Simulated Annealing": simulated_annealing_optimizer,
        "Momentum GD": momentum_gradient_descent_optimizer,
    }

    optimizer_results = []
    optimized_params = []
    hyperparameters = {
        "Gradient Descent": {"learning_rate": 0.01},
        "COBYLA": {"maxiter": 100},
        "SPSA": {"maxiter": 100},
        "Adam": {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999},
        "Simulated Annealing": {"initial_temp": 10.0, "cooling_rate": 0.95},
        "Momentum GD": {"learning_rate": 0.01, "momentum": 0.9},
    }

    for name, optimizer in optimizers.items():
        print(f"Running {name} Optimizer...")
        start_time = time.time()
        params = optimizer(initial_params, compute_expectation, backend)
        duration = time.time() - start_time
        cost = compute_expectation(params, backend)
        optimizer_results.append({
            "Optimizer": name,
            "Final Cost": cost,
            "Time Taken": duration
        })
        optimized_params.append(params)

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Save raw results as CSV
    with open("results/optimizer_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Optimizer", "Final Cost", "Time Taken (s)"])
        for result in optimizer_results:
            writer.writerow([result["Optimizer"], result["Final Cost"], result["Time Taken"]])

    # Generate PDF report
    generate_report(optimizer_results, num_qubits=4, optimized_params=optimized_params, hyperparameters=hyperparameters)

    return optimizer_results

if __name__ == "__main__":
    initial_params = np.random.uniform(-np.pi, np.pi, size=4)
    results = run_comparison_and_report(initial_params)

    print("\nFinal Optimizer Results:")
    for res in results:
        print(f"{res['Optimizer']}: Final Cost = {res['Final Cost']:.6f}, Time = {res['Time Taken']:.2f}s")
