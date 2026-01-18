
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from src.standards import calculate_rotorbar_severity
from Iec_style_severity_Engine_fixed import evaluate_rotor_bar

def test_case(name, db_level, load_pct):
    print(f"\n--- Test Case: {name} (Sidebands: {db_level} dB, Load: {load_pct}%) ---")
    
    # 1. Test Existing Standard (src/standards.py)
    # Note: src/standards expects negative dB values usually
    params_std = {
        'Upper Sideband': db_level,
        'Lower Sideband': db_level,
        'Rotorbar Health': None, # Let it derive from sidebands
        'Se Fund': 100,
        'Se Harm': 0, # Ignored for this test
        'Rotorbar Level %': None
    }
    try:
        res_std = calculate_rotorbar_severity(params_std)
        print(f"[Current Standard] Level: {res_std['Level']} ({res_std['Status']})")
    except Exception as e:
        print(f"[Current Standard] Error: {e}")

    # 2. Test IEC Engine (Iec_style_severity_Engine.py)
    try:
        res_iec = evaluate_rotor_bar(
            sb_lower_db=db_level,
            sb_upper_db=db_level,
            load_pct=load_pct,
            currents=(100, 100, 100),
            voltages=(400, 400, 400)
        )
        print(f"[IEC Engine] Severity: {res_iec['Severity_Level']} ({res_iec['Status']}) | RBI_corrected: {res_iec['RBI_corrected']}")
    except Exception as e:
        print(f"[IEC Engine] Error: {e}")

if __name__ == "__main__":
    # Case A: Healthy Motor (-60 dB)
    test_case("Healthy Motor", -60.0, 100.0)
    
    # Case B: Faulty Motor (-35 dB)
    test_case("Faulty Motor", -35.0, 100.0)

    # Case C: Healthy Motor at Low Load (-60 dB, 50% Load)
    test_case("Healthy Low Load", -60.0, 50.0)
