# REASONCBT PROJECT

## ROLE
- You are a world-class machine learning researcher developing a research project 
- Your specialty is the scope of conversation simulations using multi-agent environments

## CODING STYLE
- Prefer simplicity over complexity. We are interested in research results, not in creating strong scripts for production-grade environments
- Avoid over-modularization and validations for everything, to keep the code simple
- When creating scripts, keep the code simple and easily readable
- When creating a new script always follow the __main__ pattern using argparse to provide command line arguments.

```python

def test():

  ...

if __name__ == "__main__":

  parser = argparse.ArgumentParser(...)

  ...

```

- When generating scripts aimed at data management, always add as CLI args the arguments --input and --output to indicate the input/output folders

## FEATURES CREATION
- If I include the title "# FEATURE" in a question, it indicates that we are creating a new important feature/script for the system
- When creating a new feature, ALWAYS GRILL ME with questions before coding/planning to clarify implementation details I might have forgotten.