import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENVS = ROOT / "envs"
PROJECTS = ROOT / "projects"
BASE_ENV = ENVS / "base"

def create_project(name: str):
    print(f"Creating project {name}...")

    # Create env dir
    env_dir = ENVS / name
    env_dir.mkdir(exist_ok=True)

    base_pyproject = BASE_ENV / "pyproject.toml"
    if base_pyproject.exists():
        shutil.copy(base_pyproject, env_dir / "pyproject.toml")
    else:
        print("Warning: base pyproject.toml not found, env will be empty.")

    # Create project dir
    proj_dir = PROJECTS / name
    proj_dir.mkdir(exist_ok=True)
    (proj_dir / "notebooks").mkdir(exist_ok=True)
    (proj_dir / "src").mkdir(exist_ok=True)
    (proj_dir / "app").mkdir(exist_ok=True)

    readme = proj_dir / "README.md"
    if not readme.exists():
        readme.write_text(f"# {name}\n\nProject scaffold created.\n")

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_project.py <project_name>")
        sys.exit(1)
    create_project(sys.argv[1])