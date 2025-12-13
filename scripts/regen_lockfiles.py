import subprocess
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENVS = ROOT / "envs"

def regen(path: pathlib.Path):
    print(f"Regenerating lockfile for {path}...")
    subprocess.run(["uv", "lock"], cwd=path, check=True)

def main():
    for env in ENVS.iterdir():
        if env.is_dir():
            regen(env)

if __name__ == "__main__":
    main()
# This script regenerates lockfiles for all virtual environments
# located in the 'envs' directory relative to the script's location.