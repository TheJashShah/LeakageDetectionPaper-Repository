import pandas as pd
import numpy as np
import os

import csv

for i in range(0, 1000):

    path = f"download_dir\Hanoi\Scenario-{i}\Hanoi_CMH_Scenario-{i}.inp"

    csv_path = os.path.join(os.getcwd(), "..", "csv_of_inp", f"Scenario-{i}-structure.csv")

    if (os.path.exists(path)):
        in_pipes_section = False
        data = []
        
        with open(path, "r") as file:
            for line in file:
                line = line.strip()
                
                if not line or line.startswith(";"):
                    continue
                
                if line.startswith("[PIPES]"):
                    in_pipes_section = True
                    continue
                
                elif line.startswith("[") and in_pipes_section:
                    break
                
                if in_pipes_section:
                    parts = line.split()
                    
                    if len(parts) >= 6:
                        pipe_id, node1, node2, length, diameter, roughness = parts[:6]
                        data.append([pipe_id, node1, node2, length, diameter, roughness])
                        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pipe_id", "node1", "node2", "length", "diameter", "roughness"])
            writer.writerows(data)
            
        print(f"Extracted inp data of Scenario-{i} to csv.")

    else:
        print(f"Scenario-{i} could not be found.")
