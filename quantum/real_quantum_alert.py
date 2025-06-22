from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
import numpy as np
import os

def real_quantum_decision(confidence: float) -> str:
    theta = confidence * (np.pi / 2)
    qc = QuantumCircuit(2, 1)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.measure(1, 0)

    backend = Aer.get_backend("qasm_simulator")
    transpiled = transpile(qc, backend)
    job = backend.run(transpiled, shots=100)
    result = job.result()
    counts = result.get_counts()

     # Plot and save histogram for judges to see
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)  # create folder if not exists
    plot_histogram(counts).savefig(os.path.join(output_dir, "quantum_decision_histogram.png"))
    print(f"Quantum decision histogram saved to {output_dir}/quantum_decision_histogram.png")
    

    prob_danger = counts.get('1', 0) / 100.0

    if prob_danger > 0.7:
        return "CRITICAL THREAT"
    elif prob_danger > 0.4:
        return "MODERATE THREAT"
    else:
        return "LOW/NO THREAT"

    #------------tldr description on what it actually does (im too lazy but whatever) ------------
    # This function uses a quantum circuit to make a decision based on the confidence level.
    # It uses entanglement and actual qubit measurements to decide danger level.
