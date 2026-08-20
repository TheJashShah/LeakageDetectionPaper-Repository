from water_benchmark_hub.leakdb import LeakDB
import os, shutil

db = LeakDB()

download_dir = ""

for i in range(1, 1000):
    try:
        data = db.load_data(scenarios_id=[i], use_net1=False, download_dir="")
    
    except (FileNotFoundError, RuntimeError) as e:
        source  = f"{download_dir}\Scenario-{i}\Scenario-{i}"
        target  = f"{download_dir}\Hanoi\Scenario-{i}"
        
        files = os.listdir(source)
        
        for file in files:
            shutil.move(os.path.join(source, file), target)
        
        print(f"Scenario-{i} moved")
        
    except Exception as e:
        print(f"Scenario-{i} cannot be downloaded.")