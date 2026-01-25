
from core.services.features.reaction import ReactionPredictor

def test_reaction_prediction():
    predictor = ReactionPredictor()
    
    # Test case 1: Ester formation
    # Ethanoyl chloride + Ethanol -> Ethyl acetate
    reactants_ester = "CC(=O)Cl.CCO" 
    result_ester = predictor.predict_product(reactants_ester)
    
    print(f"--- Ester Formation Test ---")
    print(f"Reactants: {reactants_ester}")
    if result_ester:
        print(f"Products: {result_ester.products}")
        print(f"Reaction Type: {result_ester.reaction_type}")
        print(f"Reaction SMILES: {result_ester.reaction_smiles}")
        
        # Validate that product is NOT just concatenated reactants
        if result_ester.products == result_ester.reactants:
            print("[FAIL] Product matches reactants. Dummy implementation detected!")
        elif "CC(=O)OCC" in result_ester.products[0] or "CCOC(C)=O" in result_ester.products[0]:
             print("[PASS] Ethyl acetate detected.")
        else:
             print(f"[WARN] Unexpected product: {result_ester.products}")
    else:
        print("[FAIL] No reaction predicted.")

    # Test case 2: Amide formation
    # Benzoyl chloride + Methylamine -> N-Methylbenzamide
    reactants_amide = "c1ccccc1C(=O)Cl.CN"
    result_amide = predictor.predict_product(reactants_amide)
    
    print(f"\n--- Amide Formation Test ---")
    print(f"Reactants: {reactants_amide}")
    if result_amide:
        print(f"Products: {result_amide.products}")
        
        # Expected: c1ccccc1C(=O)NC
        if "CNC(=O)c1ccccc1" in result_amide.products[0] or "c1ccccc1C(=O)NC" in result_amide.products[0]:
             print("[PASS] Amide detected.")
        else:
             print(f"[WARN] Unexpected product: {result_amide.products}")
    else:
        print("[FAIL] No reaction predicted.")

if __name__ == "__main__":
    test_reaction_prediction()
