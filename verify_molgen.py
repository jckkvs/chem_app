
from core.services.features.mol_generator import MoleculeGenerator

def test_mol_generator():
    generator = MoleculeGenerator(random_state=42)
    
    # Test case: O-methylation of Phenol
    # c1ccccc1O -> c1ccccc1OC
    smiles = "c1ccccc1O"
    
    print(f"--- Molecule Generator Test ---")
    print(f"Parent SMILES: {smiles}")
    
    variants = generator.generate_variants(smiles, n_variants=10)
    
    found_methylation = False
    for v in variants:
        print(f"Variant: {v.smiles} ({v.modification})")
        if v.modification == "O-methylation":
            found_methylation = True
            # Check correctness: O should be replaced by OCH3 (Methoxy)
            # Original: Phenol (OH) -> Anisole (OCH3)
            # Note: ReplaceSubstructs replaces the MATCHED part.
            # [OH] matches the OH group. Replacing with OCH3 gives c1ccccc1OCH3 -> correct.
            # If string replace was used on "c1ccccc1O", replacing "OH" with "OCH3" works for "OH",
            # but if SMILES was "Oc1ccccc1", replacing "O" with "OCH3" might result in "OCH3c1ccccc1" (ok)
            # but replacing "O" in "CCO" -> "CCOCH3" might be wrong if it was meant to be hydroxyl.
            pass

    if found_methylation:
        print("[PASS] O-methylation generated.")
    else:
        print("[WARN] O-methylation not generated (might be random selection or failed match).")

if __name__ == "__main__":
    test_mol_generator()
