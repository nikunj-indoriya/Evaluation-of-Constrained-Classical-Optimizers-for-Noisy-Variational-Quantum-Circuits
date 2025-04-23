import matplotlib.pyplot as plt
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from fpdf import FPDF
import numpy as np
import matplotlib.pyplot as plt
import time
from graphviz import Digraph

def generate_report(optimizers_data, num_qubits, optimized_params, hyperparameters=None):
    """
    Generate a detailed report comparing the optimization algorithms.

    Arguments:
        optimizers_data: A list of dictionaries containing optimizer names and final costs.
        num_qubits: The number of qubits used in the quantum circuit.
        optimized_params: A list of optimized parameters for each optimizer.
        hyperparameters: A dictionary with hyperparameters for each optimizer (optional).
    """
    # Generate a DataFrame from the optimizer comparison data
    results = pd.DataFrame(optimizers_data)

    # Create a PDF for the report
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Optimizer Comparison Report", ln=True, align="C")
    pdf.ln(10)  # Add line break

    # Number of Qubits
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"Number of Qubits Used: {num_qubits}", ln=True)

    # Hyperparameters section (if provided)
    if hyperparameters:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Optimizer Hyperparameters", ln=True)
        pdf.set_font("Arial", "", 10)
        for optimizer, params in hyperparameters.items():
            pdf.cell(200, 10, f"{optimizer}: {params}", ln=True)
        pdf.ln(5)  # Line break after hyperparameters section

    # Add a line showing the structure of the results table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Optimizer Comparison Table", ln=True)
    pdf.ln(5)

    # Table of Results
    pdf.set_font("Arial", "", 10)
    pdf.cell(60, 10, 'Optimizer', border=1)
    pdf.cell(60, 10, 'Final Cost', border=1)
    pdf.cell(60, 10, 'Optimized Parameters', border=1)
    pdf.cell(40, 10, 'Time Taken (s)', border=1)
    pdf.ln(10)

    # Iterate over each optimizer's data and display it in the table
    for idx, row in results.iterrows():
        pdf.cell(60, 10, row['Optimizer'], border=1)
        pdf.cell(60, 10, f"{row['Final Cost']:.6f}", border=1)
        pdf.cell(60, 10, ', '.join([f"{param:.3f}" for param in optimized_params[idx]]), border=1)
        pdf.cell(40, 10, f"{row['Time Taken']:.2f}", border=1)
        pdf.ln(10)

    # Plot the comparison bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(results['Optimizer'], results['Final Cost'], color=['red', 'green', 'blue'])
    plt.xlabel('Optimizer')
    plt.ylabel('Final Cost')
    plt.title('Optimizer Comparison')
    chart_path = 'optimizer_comparison.png'
    plt.savefig(chart_path)
    plt.close()

    # Add the chart to the PDF
    pdf.ln(10)
    pdf.cell(200, 10, "Optimizer Comparison Bar Chart", ln=True, align="C")
    pdf.image(chart_path, x=10, w=180)

    # Generate the Quantum Circuit Diagram
    circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        circuit.ry(optimized_params[0][i], i)  # Assuming optimized_params is a list of parameter arrays
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)
    circuit_path = "quantum_circuit.png"
    circuit.draw(output='mpl').savefig(circuit_path)

    # Add the quantum circuit diagram to the PDF
    pdf.ln(10)
    pdf.cell(200, 10, "Quantum Circuit Diagram", ln=True, align="C")
    pdf.image(circuit_path, x=10, w=180)

    # Generate the Process Flowchart dynamically using Graphviz
    flowchart = Digraph('process_flow')
    flowchart.node('A', 'Start')
    flowchart.node('B', 'Initialize Parameters')
    flowchart.node('C', 'Run Optimizer')
    flowchart.node('D', 'Compute Cost')
    flowchart.node('E', 'Check Convergence')
    flowchart.node('F', 'End')

    flowchart.edge('A', 'B')
    flowchart.edge('B', 'C')
    flowchart.edge('C', 'D')
    flowchart.edge('D', 'E')
    flowchart.edge('E', 'C', label='Not Converged')
    flowchart.edge('E', 'F', label='Converged')

    # Render the flowchart to a PNG file
    flowchart.render("process_flowchart", format='png')
    flowchart_path = "process_flowchart.png"

    # Add the process flowchart to the PDF
    pdf.ln(10)
    pdf.cell(200, 10, "Optimization Process Flowchart", ln=True, align="C")
    pdf.image(flowchart_path, x=10, w=180)

    # Save the PDF
    pdf_output_path = "optimizer_comparison_report.pdf"
    pdf.output(pdf_output_path)

    print(f"Report Generated: {pdf_output_path}")

# Example Usage:
optimizers_data = [
    {'Optimizer': 'Gradient Descent', 'Final Cost': 0.05, 'Time Taken': 1.25},
    {'Optimizer': 'Momentum GD', 'Final Cost': 0.03, 'Time Taken': 1.75},
    {'Optimizer': 'Adam', 'Final Cost': 0.02, 'Time Taken': 2.50}
]
num_qubits = 4
optimized_params = [
    [0.5, 0.3, 0.2, 0.1],  # Optimized parameters for Gradient Descent
    [0.6, 0.4, 0.25, 0.15],  # Optimized parameters for Momentum GD
    [0.55, 0.35, 0.22, 0.12]  # Optimized parameters for Adam
]

# Example Hyperparameters
hyperparameters = {
    'Gradient Descent': {'learning_rate': 0.01, 'momentum': 0.9},
    'Momentum GD': {'learning_rate': 0.01, 'momentum': 0.95},
    'Adam': {'learning_rate': 0.001, 'beta_1': 0.9, 'beta_2': 0.999}
}

generate_report(optimizers_data, num_qubits, optimized_params, hyperparameters)
