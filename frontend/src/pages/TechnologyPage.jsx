import '../App.css'
import { FaBrain, FaWifi, FaBatteryFull, FaShieldAlt, FaEye, FaMicrochip, FaCloud, FaMobile, FaCode, FaCogs } from 'react-icons/fa'

const TechnologyPage = () => {
  const quantumCode = `# Quantum Decision Tree for Wildlife Threat Assessment
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np

def create_threat_assessment_circuit(environmental_data):
    # Create quantum circuit with 4 qubits for threat assessment
    qc = QuantumCircuit(4, 4)
    
    # Encode environmental factors as quantum states
    # Qubit 0: Proximity to wildlife
    # Qubit 1: Time of day (night vs day)
    # Qubit 2: Weather conditions
    # Qubit 3: Terrain type
    
    # Apply quantum gates based on environmental data
    if environmental_data['proximity'] > 0.7:
        qc.x(0)  # High proximity = danger
    
    if environmental_data['time'] == 'night':
        qc.x(1)  # Night time = increased risk
    
    if environmental_data['weather'] == 'poor':
        qc.x(2)  # Poor weather = higher risk
    
    if environmental_data['terrain'] == 'dense':
        qc.x(3)  # Dense terrain = limited escape
    
    # Apply quantum entanglement for threat correlation
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    
    # Measure all qubits
    qc.measure_all()
    
    return qc

def assess_threat_level(environmental_data):
    qc = create_threat_assessment_circuit(environmental_data)
    
    # Execute on quantum simulator
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000)
    result = job.result()
    
    # Analyze results for threat assessment
    counts = result.get_counts(qc)
    threat_score = analyze_quantum_results(counts)
    
    return threat_score

def analyze_quantum_results(counts):
    # Convert quantum measurements to threat score
    total_shots = sum(counts.values())
    high_threat_states = 0
    
    for state, count in counts.items():
        if state.count('1') >= 3:  # 3 or more danger indicators
            high_threat_states += count
    
    threat_probability = high_threat_states / total_shots
    return min(threat_probability * 100, 100)  # Scale to 0-100

# Example usage
env_data = {
    'proximity': 0.8,
    'time': 'night',
    'weather': 'clear',
    'terrain': 'open'
}

threat_level = assess_threat_level(env_data)
print(f"Threat Level: {threat_level:.1f}%")`

  return (
    <div className="page">
      <div className="technology-content" style={{ background: 'white', padding: '4rem 2rem' }}>
        <h1 style={{ color: 'var(--forest-green)', marginBottom: '2rem', textAlign: 'center' }}>Our Technology</h1>
        
        <div style={{ 
          background: 'var(--gray-100)', 
          padding: '3rem 2rem', 
          borderRadius: '12px',
          marginBottom: '3rem'
        }}>
          <h2 style={{ color: 'var(--forest-green)', marginBottom: '1.5rem' }}>AI-Powered Detection</h2>
          <p style={{ fontSize: '1.1rem', lineHeight: '1.6', color: 'var(--gray-700)' }}>
            BioBuddy uses advanced machine learning algorithms trained on millions of wildlife images and sounds. 
            Our AI can identify multiple species of animals, birds, and reptiles with high accuracy, even in 
            challenging conditions like low light, rain, or dense vegetation. The system provides 48-hour 
            continuous detection monitoring, ensuring round-the-clock wilderness safety.
          </p>
        </div>

        <div style={{ 
          background: 'white', 
          padding: '3rem 2rem', 
          borderRadius: '12px',
          marginBottom: '3rem'
        }}>
          <h2 style={{ color: 'var(--forest-green)', marginBottom: '1.5rem' }}>48-Hour Continuous Detection</h2>
          <p style={{ fontSize: '1.1rem', lineHeight: '1.6', color: 'var(--gray-700)' }}>
            BioBuddy's advanced power management and efficient AI processing enable 48-hour continuous 
            wildlife detection without interruption. This extended monitoring period ensures comprehensive 
            coverage for multi-day outdoor activities, providing peace of mind during extended wilderness 
            adventures. The system maintains peak performance throughout the entire detection period while 
            conserving battery life through intelligent power cycling and adaptive processing.
          </p>
        </div>

        <div style={{ 
          background: 'var(--gray-100)', 
          padding: '3rem 2rem', 
          borderRadius: '12px',
          marginBottom: '3rem'
        }}>
          <h2 style={{ color: 'var(--forest-green)', marginBottom: '1.5rem' }}>Quantum Computing Integration</h2>
          <p style={{ fontSize: '1.1rem', lineHeight: '1.6', color: 'var(--gray-700)', marginBottom: '2rem' }}>
            Our backend decision tree leverages quantum computing principles to process complex environmental 
            data and make split-second decisions about threat levels. This quantum-enhanced processing allows 
            BioBuddy to analyze multiple data streams simultaneously and provide more accurate risk assessments.
          </p>
          
          <div className="quantum-code-preview">
            <h3>Quantum Decision Tree Implementation</h3>
            <div className="code-block">
              <pre><code>{quantumCode}</code></pre>
            </div>
          </div>
        </div>

        <div style={{ 
          background: 'white', 
          padding: '3rem 2rem', 
          borderRadius: '12px',
          marginBottom: '3rem'
        }}>
          <h2 style={{ color: 'var(--forest-green)', marginBottom: '1.5rem' }}>Rugged Hardware Design</h2>
          <p style={{ fontSize: '1.1rem', lineHeight: '1.6', color: 'var(--gray-700)' }}>
            Built to withstand the harshest outdoor conditions, BioBuddy devices feature:
          </p>
          <ul style={{ fontSize: '1.1rem', lineHeight: '1.6', color: 'var(--gray-700)', marginTop: '1rem' }}>
            <li> Highly responsive to voice commands+voice isolation.</li>
            <li>Operating temperature range in large temmpurature ranges.</li>
            <li>High Level of hock resistance</li>
            <li>Extended battery life supporting 48-hour continuous detection</li>
            <li>100% offline operation - no internet required</li>
            <li>Advanced sensor array including motion detectio and audio analysis</li>
          </ul>
        </div>

        <div style={{ 
          background: 'var(--gray-100)', 
          padding: '3rem 2rem', 
          borderRadius: '12px'
        }}>
          <h2 style={{ color: 'var(--forest-green)', marginBottom: '1.5rem' }}>Real-Time Processing</h2>
          <p style={{ fontSize: '1.1rem', lineHeight: '1.6', color: 'var(--gray-700)' }}>
            Every BioBuddy device processes data locally using our proprietary edge computing technology. 
            This means instant alerts and zero latency, even in remote areas without cellular coverage. 
            The device continuously learns and adapts to its environment, improving accuracy over time.
          </p>
        </div>
      </div>
    </div>
  )
}

export default TechnologyPage