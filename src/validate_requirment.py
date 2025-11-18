import os
import ast
import pkgutil

PROJECT_DIR = "src"          # your source folder
REQUIREMENTS_FILE = "requirements.txt"

# --------------------------------------------
# 1. Load installed standard library modules
# --------------------------------------------

import sys
import stdlib_list

try:
    std_libs = set(stdlib_list.stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}"))
except:
    std_libs = set()

# --------------------------------------------
# 2. Load requirements.txt modules
# --------------------------------------------
def load_requirements():
    reqs = set()
    if not os.path.exists(REQUIREMENTS_FILE):
        print("requirements.txt not found!")
        return reqs

    with open(REQUIREMENTS_FILE, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line and not line.startswith("#"):
                pkg = line.split("==")[0].split("[")[0]
                reqs.add(pkg.lower())

    return reqs


# --------------------------------------------
# 3. Extract imports from all .py files
# --------------------------------------------

def get_imports():
    imports = set()

    for root, _, files in os.walk(PROJECT_DIR):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)

                with open(filepath, "r", encoding="utf-8") as f:
                    node = ast.parse(f.read(), filename=filepath)

                for n in ast.walk(node):
                    if isinstance(n, ast.Import):
                        for alias in n.names:
                            imports.add(alias.name.split('.')[0])

                    elif isinstance(n, ast.ImportFrom):
                        if n.module:
                            imports.add(n.module.split('.')[0])

    return imports


# --------------------------------------------
# 4. Classify modules
# --------------------------------------------

def classify_modules(imports, requirements):
    missing = []
    present = []
    builtin = []

    for module in imports:
        if module.lower() in requirements:
            present.append(module)
        elif module in std_libs:
            builtin.append(module)
        else:
            # Check if module exists by pkgutil (may catch installed libs)
            if pkgutil.find_loader(module) is None:
                missing.append(module)
            else:
                present.append(module)

    return missing, present, builtin


# --------------------------------------------
# 5. Main
# --------------------------------------------

if __name__ == "__main__":
    print("\n📂 Scanning project imports...")

    requirements = load_requirements()
    imports = get_imports()

    missing, present, builtin = classify_modules(imports, requirements)

    print("\n-----------------------------")
    print("📌 RESULTS")
    print("-----------------------------")

    print("\n🟢 Present in requirements.txt:")
    for m in sorted(present):
        print("   •", m)

    print("\n🔵 Built-in Python modules (OK, no need to add):")
    for b in sorted(builtin):
        print("   •", b)

    print("\n❌ Missing modules (Add these to requirements.txt):")
    for x in sorted(missing):
        print("   •", x)

    if not missing:
        print("\n🎉 No missing modules! requirements.txt is complete.")
