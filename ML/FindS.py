import csv

csv_file = open('DATA/FindS.csv', 'r')
readcsv = csv.reader(csv_file)
data = []

print("\nThe given training examples are:")
for row in readcsv:
    print(row)
    if row[-1].strip().lower() == "yes":
        data.append(row)

print("\nThe positive examples are:")
for x in data:
    print(x)

if len(data) > 0:
    hypothesis = data[0][:-1]
    print("\nThe steps of the Find-S algorithm are:")
    print(hypothesis)

    for i in range(1, len(data)):
        for k in range(len(hypothesis)):
            if hypothesis[k] != data[i][k]:
                hypothesis[k] = '?'
        print(f"After training example {i + 1}, hypothesis is: {hypothesis}")

    print("\nThe maximally specific Find-S hypothesis for the given training examples is:")
    print(hypothesis)
else:
    print("\nNo positive examples found.")
