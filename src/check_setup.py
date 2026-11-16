"""Checks that all project libraries are installed correctly."""
import importlib

libraries = ["pandas", "numpy", "scipy", "sklearn", "xgboost",
             "shap", "matplotlib", "plotly", "streamlit", "pulp"]

all_ok = True
for name in libraries:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"OK      {name:12s} version {version}")
    except Exception as error:
        all_ok = False
        print(f"FAILED  {name:12s} -> {error}")

print()
print("Setup complete!" if all_ok else "Some libraries failed. Fix them before continuing.")