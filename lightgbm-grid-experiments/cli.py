import os
print("[Search] Preparing experiments")
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
filename = os.path.join(dname, "grid.json")
print(filename)