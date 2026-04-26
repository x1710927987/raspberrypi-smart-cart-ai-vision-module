# Python Environment Setup Guide

This guide explains how to recreate the project environment after pulling the repository from GitHub.

Use PowerShell on Windows unless your team has agreed on a different shell.

Important notes:
- Use only one setup method for the same local environment: either `requirements.txt` or `environment.yml`.
- Do not use absolute paths from another teammate's computer. Always run the commands from your own local clone of the repository.
- This repository targets Python `3.14`.

## 1. Move to the Repository Root

After cloning or pulling the repository, open PowerShell and go to the project root:

```powershell
cd "YOUR_LOCAL_PATH\raspberrypi-smart-cart-ai-vision-module"
```

Example:

```powershell
cd "D:\Repository\internship_and_research\raspberrypi-smart-cart-ai-vision-module"
```

## 2. Method A: Create the Environment from `requirements.txt`

Use this method if you want a clean Conda environment and install the project packages with `pip`.

### Step A1. Create the Conda environment

```powershell
conda create -n smartcart-ai python=3.14 -y
```

### Step A2. Activate the environment

```powershell
conda activate smartcart-ai
```

If `conda activate` does not work in PowerShell, run this once, close PowerShell, and open it again:

```powershell
conda init powershell
```

### Step A3. Upgrade `pip`

```powershell
python -m pip install --upgrade pip
```

### Step A4. Install dependencies from `requirements.txt`

```powershell
python -m pip install -r .\virtual_environment_configuration\requirements.txt
```

## 3. Method B: Create the Environment from `environment.yml`

Use this method if you want Conda to recreate the environment from the exported environment file.

### Step B1. Check `environment.yml` for a user-specific `prefix:` line

Before creating the environment, open:

```text
virtual_environment_configuration\environment.yml
```

If the file contains a line like this at the end:

```yaml
prefix: C:\Users\some_user\.conda\envs\smartcart-ai
```

delete that `prefix:` line before continuing. It is machine-specific and should not be reused on another teammate's computer.

### Step B2. Create the environment

```powershell
conda env create -f .\virtual_environment_configuration\environment.yml
```

### Step B3. Activate the environment

```powershell
conda activate smartcart-ai
```

If `conda activate` does not work in PowerShell, run this once, close PowerShell, and open it again:

```powershell
conda init powershell
```

## 4. Verify the Environment

After using either method, run these checks:

```powershell
python --version
python -c "import sys; print(sys.executable)"
python -m pip list
```

You should confirm:
- the Python version is `3.14.x`
- the executable path points to your `smartcart-ai` environment
- the required packages are installed

You can also check the main packages directly:

```powershell
python -m pip show numpy
python -m pip show opencv-python
python -m pip show PyYAML
python -m pip show pyserial
python -m pip show pytest
```

## 5. Optional Test Command

If you want to do a quick project check after installation:

```powershell
python -m unittest
```

Note: if tests fail, that does not always mean the environment setup is wrong. It may also mean the project code still needs changes.

## 6. Useful Maintenance Commands

### Show all Conda environments

```powershell
conda env list
```

### Remove the environment completely

```powershell
conda remove -n smartcart-ai --all -y
```

### Reinstall dependencies with the `requirements.txt` method

```powershell
conda activate smartcart-ai
python -m pip install --upgrade pip
python -m pip install -r .\virtual_environment_configuration\requirements.txt
```

## 7. Recommended Team Workflow

- Keep `virtual_environment_configuration/requirements.txt` updated when project dependencies change.
- If `environment.yml` is re-exported, remove the machine-specific `prefix:` line before committing.
- When sharing setup instructions in the team, refer to files by relative path instead of absolute local paths.
