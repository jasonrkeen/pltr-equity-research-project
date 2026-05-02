import os
import shutil
from datetime import datetime


OUTPUT_DIRS = [
    "outputs/charts",
    "outputs/tables",
    "data/raw",
    "data/processed"
]

CLEAN_DIRS = [
    "outputs/charts",
    "outputs/tables"
]


def setup_project_folders():
    for folder in OUTPUT_DIRS:
        os.makedirs(folder, exist_ok=True)

        gitkeep_path = os.path.join(folder, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, "w"):
                pass


def clean_outputs(enabled=True, verbose=True):
    if not enabled:
        if verbose:
            print("Skipping output cleanup")
        return

    for folder in CLEAN_DIRS:
        if not os.path.exists(folder):
            continue

        if verbose:
            print(f"Cleaning: {folder}")

        for filename in os.listdir(folder):
            if filename == ".gitkeep":
                continue

            file_path = os.path.join(folder, filename)

            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")


def create_timestamped_run_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"outputs/runs/run_{timestamp}"

    os.makedirs(f"{run_folder}/charts", exist_ok=True)
    os.makedirs(f"{run_folder}/tables", exist_ok=True)

    return run_folder


def copy_outputs_to_run_folder(run_folder):
    chart_source = "outputs/charts"
    table_source = "outputs/tables"

    chart_target = f"{run_folder}/charts"
    table_target = f"{run_folder}/tables"

    for folder, target in [(chart_source, chart_target), (table_source, table_target)]:
        if not os.path.exists(folder):
            continue

        for filename in os.listdir(folder):
            if filename == ".gitkeep":
                continue

            source_path = os.path.join(folder, filename)
            target_path = os.path.join(target, filename)

            if os.path.isfile(source_path):
                shutil.copy2(source_path, target_path)