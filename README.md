# Evaluation of Constrained Classical Optimizers for Noisy Variational Quantum Circuits

This project investigates the performance of various classical optimization algorithms applied to noisy variational quantum circuits. It focuses on the integration of realistic constraints such as circuit depth, hardware noise, and parameter variance — factors that are crucial in the NISQ (Noisy Intermediate-Scale Quantum) era.

---

##  Project Highlights

- Custom 4-qubit variational quantum circuit ansatz
- Expectation value computations using Qiskit simulators
- Multiple classical optimizers implemented:
  - Gradient Descent
  - Adam
  - Momentum-based GD
  - Simulated Annealing
  - SPSA (Simultaneous Perturbation Stochastic Approximation)
  - COBYLA
- Constraints enforced on:
  - Circuit depth
  - Variance of parameters
  - Fidelity & noise tolerance
- Comparative analysis with visual plots and metrics
- PDF report generation with flowchart and results

---

##  Installation

Before running this project, ensure you have Python 3.10+ installed.

Install all required packages using:

```bash
pip install qiskit numpy scipy matplotlib fpdf
```
---

###  Getting Started

Clone the repository:

```bash
git clone https://github.com/nikunj-indoriya/Evaluation-of-Constrained-Classical-Optimizers-for-Noisy-Variational-Quantum-Circuits.git
cd Evaluation-of-Constrained-Classical-Optimizers-for-Noisy-Variational-Quantum-Circuits
cd Applied Optimization Project
```
Run the main file:
```bash
python main.py
```
