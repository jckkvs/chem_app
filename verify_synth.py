
from core.services.features.synthesizability import SynthesizabilityAssessor

def test_synthesizability():
    assessor = SynthesizabilityAssessor()
    
    # Test case: Molecule with chiral center and alkyne
    # [C@@H] is chiral, C#C is alkyne
    # Note: RDKit might canonicalize SMILES differently, so HasSubstructMatch is essential.
    smiles = "C#C[C@@H](C)O" 
    
    print(f"--- Synthesizability Test ---")
    print(f"SMILES: {smiles}")
    
    result = assessor.assess(smiles)
    
    if result:
        print(f"SA Score: {result.sa_score}")
        print(f"Alerts: {result.structural_alerts}")
        
        # Check if alerts are correctly detected
        expected_alerts = ['キラル中心', 'アルキン']
        detected = 0
        for alert in expected_alerts:
            if alert in result.structural_alerts:
                print(f"[PASS] Alert detected: {alert}")
                detected += 1
            else:
                print(f"[FAIL] Alert NOT detected: {alert}")
                
        if detected == len(expected_alerts):
            print("All expected alerts detected.")
    else:
        print("[FAIL] Assessment failed.")

if __name__ == "__main__":
    test_synthesizability()
