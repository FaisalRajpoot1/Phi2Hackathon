"""Upload the Gradio app to its Hugging Face Space.

The Space gets space/app.py, space/requirements.txt, space/README.md, the
graph_detective package and the sample data, laid out the way the Space expects.
Log in first with `hf auth login`.

Usage, from the repo root:

    python space/deploy.py Faisal87/graph-detective
    python space/deploy.py Faisal87/graph-detective --stage-only some/folder
"""
import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stage(folder):
    """Copy the Space's files into `folder`, in the Space's layout."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy(ROOT / "space" / name, folder / name)
    shutil.copytree(ROOT / "graph_detective", folder / "graph_detective",
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    (folder / "data").mkdir(exist_ok=True)
    for sample in ("fraud.json", "fraud2.json"):
        shutil.copy(ROOT / "data" / sample, folder / "data" / sample)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("space_id", help="for example Faisal87/graph-detective")
    parser.add_argument("--stage-only", metavar="FOLDER", help="only lay the files out in FOLDER, for a local test")
    args = parser.parse_args()

    if args.stage_only:
        print(f"Staged in {stage(args.stage_only)}")
        return

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as folder:
        HfApi().upload_folder(folder_path=str(stage(folder)), repo_id=args.space_id, repo_type="space",
                              commit_message="Deploy from the GitHub fork")
    print(f"Uploaded. Open https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
