PATH = "app_v7.py"
with open(PATH, "rb") as f:
    lines = f.readlines()
for idx in range(1314, 1360):
    print(idx + 1, repr(lines[idx]))
