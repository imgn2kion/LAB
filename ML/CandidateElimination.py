import numpy as np
import pandas as pd

data = pd.read_csv('trainingdata_candidate_elimination.csv')
print(data)

concepts = np.array(data.iloc[:, 0:-1])
print("\nConcepts:")
print(concepts)

target = np.array(data.iloc[:, -1])
print("\nTarget:")
print(target)

def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("\nInitialization of specific_h and general_h")
    print("Specific Hypothesis:", specific_h)

    general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    print("General Hypothesis:", general_h)

    for i, h in enumerate(concepts):
        print(f"\nInstance {i+1}: {h}")

        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
            general_h[i] = '?'
        
        elif target[i] == "No":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]

        print("Specific Hypothesis after instance", i+1, ":", specific_h)
        print("General Hypothesis after instance", i+1, ":", general_h)

    general_h = [h for h in general_h if h != ['?']*len(specific_h)]
    print("\nFinal General Hypothesis:", general_h)

learn(concepts, target)
