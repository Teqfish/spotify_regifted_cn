import os, re, subprocess, sys
from pathlib import Path

# 1. Gather all .py files
py_files = list(Path(".").rglob("*.py"))

# 2. Extract all top-level imports
import_pattern = re.compile(r'^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)')
imported = set()

for file in py_files:
    for line in open(file, encoding="utf-8", errors="ignore"):
        m = import_pattern.match(line)
        if m:
            root = m.group(1).split(".")[0]
            imported.add(root.lower())

# 3. Get installed packages
installed = {
    line.split("==")[0].lower()
    for line in subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    if "==" in line
}

# 4. Safe “mapping” between import names and pip packages
# (some packages import under different names)
ALIASES = {
    "pil": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "dateutil": "python-dateutil",
    "bcrypt": "python-bcrypt",
    "streamlit_authenticator": "streamlit-authenticator",
    "country_converter": "country-converter",
}

resolved_imports = set(imported)
for k, v in ALIASES.items():
    if k in imported:
        resolved_imports.add(v)

# 5. Compare sets
unused = sorted(installed - resolved_imports)

print("\n Detected imports in project:")
print(", ".join(sorted(resolved_imports)))

print("\n Installed packages:")
print(", ".join(sorted(installed)))

print("\n Unused packages (safe to remove):")
print(", ".join(unused) if unused else "None ")

# Print clean machine-readable line for piping
if unused:
    print("\nUNUSED:" + " ".join(unused))
