"""Prompt profiles retain the archived and current notebook wording.

Both protocol labels execute telemetry replacement, NOT physical isolation.
The raw prompt wording is retained to make the reproduction behavior inspectable.
"""

def forensic_prompt(building, sensors):
    return f'''
    You are an AI Forensics Analyst. The community network shows an anomaly at Building {building}.
    Mean Squared Error deviations at peak impact:
    - Pricing: {sensors["Electricity_Pricing"]:.4f}
    - Load Meter: {sensors["Non_Shiftable_Load"]:.4f}
    - Solar Inverter: {sensors["Solar_Generation"]:.4f}
    - Clock Sync: {sensors["Clock_Sync"]:.4f}

    Respond ONLY in valid JSON format:
    {{
        "diagnosis": "Attack Name (e.g. Market Hack, Meter Hack, Time Spoofing, Inverter Hack)",
        "reasoning": "Why this specific sensor indicates this attack."
    }}
    '''

def archived_defense_prompt(building, diagnosis):
    return f'''
    You are the Community Defense Agent. The Forensics Node diagnosed a '{diagnosis}' on Building {building}.
    You must issue a defense command to stabilize the community's energy profile.
    Select ONE protocol:
    1. "DIGITAL_TWIN_OVERRIDE" (Replaces corrupted sensor data with known-good model predictions)
    2. "ASSET_ISOLATION" (Disconnects the building from the community network)

    Respond ONLY in valid JSON format:
    {{
        "command": "The exact protocol name",
        "justification": "Why this protocol is safe."
    }}
    '''



def current_defense_prompt(building, diagnosis):
    bldg_id = building
    return f"""\n    You are the Community Defense Agent. The Forensics Node diagnosed a '{diagnosis}' on Building {bldg_id}.\n    You must issue a defense command to stabilize the community's energy profile. \n    \n    1. Select a technical protocol: "DIGITAL_TWIN_OVERRIDE" or "ASSET_ISOLATION".\n    2. Define an actual physical defense strategy (e.g., 'Implement immediate power cycling on Building {bldg_id}s main electrical panel' or 'Isolate the power grid and re-route energy').\n    3. Provide a 'justification' that gives a deep, technical reason explaining EXACTLY how this physical strategy and protocol will physically and electrically resolve the cyber-physical anomaly and restore grid stability.\n    \n    Respond ONLY in strictly valid JSON format exactly like this. Do not add markdown formatting or conversational text outside the JSON:\n    {{\n        "protocol": "The technical protocol name",\n        "strategy": "The actionable physical strategy.",\n        "justification": "The detailed technical reason why the strategy will work."\n    }}\n    """


def defense_prompt(building, diagnosis, profile="current"):
    if profile == "current":
        return current_defense_prompt(building, diagnosis)
    if profile == "archived":
        return archived_defense_prompt(building, diagnosis)
    raise ValueError("Unknown prompt profile.")
