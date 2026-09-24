"""Size of an environment's installed packages (its site-packages folder).

    python benchmarks/bench_install_size.py path/to/env

Prints one JSON line: total bytes, number of files, and number of installed packages.
"""
import json
import sys
from pathlib import Path


def site_packages(env):
    for folder in [env / "Lib" / "site-packages", *env.glob("lib/python*/site-packages")]:
        if folder.is_dir():
            return folder
    raise SystemExit(f"No site-packages folder in {env}")


def main():
    env = Path(sys.argv[1]).resolve()
    folder = site_packages(env)
    files = [path for path in folder.rglob("*") if path.is_file() and not path.is_symlink()]
    print(json.dumps({
        "env": env.name,
        "bytes": sum(path.stat().st_size for path in files),
        "files": len(files),
        "packages": len(list(folder.glob("*.dist-info"))),
    }))


if __name__ == "__main__":
    main()
