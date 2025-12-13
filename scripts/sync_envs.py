import shutil
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENVS = ROOT / "envs"
PROJECTS = ROOT / "projects"

def sync(project_name: str):
    env_dir = ENVS / project_name
    proj_dir = PROJECTS / project_name

    if not env_dir.is_dir():
        print(f"Skipping {project_name}: no env directory.")
        return
    if not proj_dir.is_dir():
        print(f"Skipping {project_name}: no project directory.")
        return

    print(f"Syncing env for {project_name}...")

    shutil.copy(env_dir / "pyproject.toml", proj_dir / "pyproject.toml")
    lock_file = env_dir / "uv.lock"
    if lock_file.exists():
        shutil.copy(lock_file, proj_dir / "uv.lock")
    else:
        print(f"Warning: {lock_file} not found, run uv lock first.")

def main():
    for env in ENVS.iterdir():
        if env.is_dir() and env.name != "base":
            sync(env.name)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3 -- envs/projectA/.venv/bin/python
#python scripts/sync_envs.py