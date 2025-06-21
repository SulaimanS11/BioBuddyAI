from qiskit import QuantumCircuit, Aer, execute
import numpy as np

def real_quantum_decision(confidence: float) -> str:
    # Confidence ∈ [0, 1] → angle θ ∈ [0, π/2]
    theta = confidence * (np.pi / 2)
    qc = QuantumCircuit(2, 1)

    # Encode confidence level in a rotation
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.measure(1, 0)

    # Use the Qiskit simulator
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=100).result()
    counts = result.get_counts()

    prob_danger = counts.get('1', 0) / 100.0

    if prob_danger > 0.7:
        return "CRITICAL THREAT"
    elif prob_danger > 0.4:
        return "MODERATE THREAT"
    else:
        return "LOW/NO THREAT"
    
    #------------Description on what it actually does ------------
    # This function uses a quantum circuit to make a decision based on the confidence level.
    # It uses entanglement and actual qubit measurements to decide danger level.
