
import os

for ModelName in ['src','tgt']:
    for NIdx in range(6):
        for ExpertIdx in range(8):
            os.system(f".\\venv\\Scripts\\python.exe .\\SwitchTransfomer\\main.py experiment --Greedy True --Beam False --num 100000 --GpuNum 1 --batch_size 16 --Occlusion_ModelName {ModelName} --Occlusion_NIdx {NIdx} --Occlusion_ExpertIdx {ExpertIdx}")    
