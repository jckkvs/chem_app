import requests
import time
import sys

API_URL = "http://127.0.0.1:8000/api"

def run_verification():
    print("1. Uploading Dataset...")
    content = "SMILES,target\nC,1.0\nCC,2.0\nCCC,3.0\n"
    files = {'file': ('test_uma.csv', content, 'text/csv')}
    res = requests.post(f"{API_URL}/datasets", files=files, data={'name': 'UMA Test', 'smiles_col': 'SMILES', 'target_col': 'target'})
    if res.status_code != 200:
        print(f"Failed upload: {res.text}")
        sys.exit(1)
    ds_id = res.json()['id']
    print(f"Dataset ID: {ds_id}")

    print("2. Starting UMA Experiment...")
    res = requests.post(f"{API_URL}/experiments", json={
        "dataset_id": ds_id,
        "name": "UMA Exp",
        "features": ["uma"], # Using UMA
        "model_type": "lgbm"
    })
    if res.status_code != 200:
        print(f"Failed create: {res.text}")
        sys.exit(1)
    exp_id = res.json()['id']
    print(f"Experiment ID: {exp_id}")

    print("3. Waiting for Training...")
    # NOTE: user must have huey_consumer running! 
    # If not running, this will hang forever or fail if we don't handle it.
    # For this script, we assume worker is running OR we manually trigger for testing?
    # We can't manually trigger easily if it's async. 
    # BUT, we can use 'huey.immediate = True' in settings for testing?
    # Or just start a consumer in background in the tool usage?
    # Let's try waiting 10s.
    
    for i in range(10):
        res = requests.get(f"{API_URL}/experiments/{exp_id}")
        status = res.json()['status']
        print(f"Status: {status}")
        if status == 'COMPLETED':
            break
        if status == 'FAILED':
            print("Experiment Failed!")
            sys.exit(1)
        time.sleep(2)
        
    if status != 'COMPLETED':
        print("Timeout waiting for experiment.")
        sys.exit(1)

    print("4. Testing Interactive Prediction with UMA...")
    res = requests.post(f"{API_URL}/experiments/{exp_id}/predict", json={"smiles": "CCCC"})
    if res.status_code == 200:
        print(f"Prediction Success: {res.json()['prediction']}")
    else:
        print(f"Prediction Failed: {res.text}")
        sys.exit(1)

if __name__ == "__main__":
    run_verification()
