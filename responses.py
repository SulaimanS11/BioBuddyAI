# responses.py
from voice.text_to_speech import speak

def plant_response(threat_level):
    if threat_level > 0.8:
        danger_type = "HIGHLY POISONOUS PLANT"
        action = "Do not touch! Keep away from skin! Wash hands immediately if contact!"
    elif threat_level > 0.6:
        danger_type = "POISONOUS PLANT"
        action = "Avoid contact! Do not consume! Keep children and pets away!"
    elif threat_level > 0.4:
        danger_type = "POTENTIALLY HARMFUL PLANT"
        action = "Do not eat! Avoid unnecessary handling!"
    else:
        danger_type = "SAFE PLANT"
        action = "Appears safe but avoid consumption!"
    
    message = f"{danger_type} detected. {action}"
    speak(message)
    return danger_type, action

def snake_response(threat_level):
    if threat_level > 0.8:
        danger_type = "VENOMOUS SNAKE"
        action = "Stop! Back away slowly! Do not run! Call emergency services!"
    elif threat_level > 0.6:
        danger_type = "DANGEROUS SNAKE"
        action = "Keep distance! Do not approach! Let it move away!"
    elif threat_level > 0.4:
        danger_type = "UNKNOWN SNAKE"
        action = "Stay back! Treat as dangerous! Observe from distance!"
    else:
        danger_type = "NON-VENOMOUS SNAKE"
        action = "Keep respectful distance! Do not handle!"
    
    message = f"{danger_type} detected. {action}"
    speak(message)
    return danger_type, action

def bear_response(threat_level):
    if threat_level > 0.8:
        danger_type = "AGGRESSIVE BEAR"
        action = "Do not run! Make yourself large! Back away slowly! Make noise!"
    elif threat_level > 0.6:
        danger_type = "BEAR SPOTTED"
        action = "Stay calm! Keep distance! Make noise! Do not approach!"
    elif threat_level > 0.4:
        danger_type = "POSSIBLE BEAR"
        action = "Be alert! Make noise! Prepare to retreat!"
    else:
        danger_type = "BEAR ACTIVITY"
        action = "Stay vigilant! Make noise while moving!"
    
    message = f"{danger_type} detected. {action}"
    speak(message)
    return danger_type, action

def other_danger_response(threat_level):
    if threat_level > 0.8:
        danger_type = "HIGH THREAT"
        action = "Immediate danger! Move away quickly! Seek safety!"
    elif threat_level > 0.6:
        danger_type = "MODERATE THREAT"
        action = "Caution required! Keep safe distance! Stay alert!"
    elif threat_level > 0.4:
        danger_type = "POTENTIAL RISK"
        action = "Exercise caution! Observe from distance!"
    else:
        danger_type = "LOW RISK"
        action = "Stay aware! Continue with caution!"
    
    message = f"{danger_type} detected. {action}"
    speak(message)
    return danger_type, action