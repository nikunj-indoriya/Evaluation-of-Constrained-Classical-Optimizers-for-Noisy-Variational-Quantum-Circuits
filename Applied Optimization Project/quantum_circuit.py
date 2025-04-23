from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
import numpy as np

def create_hamiltonian():
    hamiltonian = SparsePauliOp.from_list([
        ("ZZII", -1),
        ("IIZZ", -1),
        ("XXII", 1),
        ("IIXX", 1)
    ])
    return hamiltonian

def ansatz(params, num_qubits=4):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc

def get_user_params(num_qubits):
    params = []
    for i in range(num_qubits):
        param = float(input(f"Enter the RY angle for qubit {i + 1}: "))
        params.append(param)
    return np.array(params)

def create_custom_hamiltonian(operators, coefficients):
    hamiltonian = SparsePauliOp.from_list(zip(operators, coefficients))
    return hamiltonian

# Example usage
num_qubits = int(input("Enter the number of qubits for the ansatz: "))
params = get_user_params(num_qubits)

operators = ["ZZII", "IIZZ", "XXII", "IIXX"]
coefficients = [-1, -1, 1, 1]
hamiltonian = create_custom_hamiltonian(operators, coefficients)

qc = ansatz(params, num_qubits)
print(qc.draw())
