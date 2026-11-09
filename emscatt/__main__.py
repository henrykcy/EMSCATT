"""Command line: python -m emscatt {fit,assemble,run} project.json [--only NAME ...] [-v]"""
import argparse

from .project import assemble, fit_components, load_project


def main(argv=None):
    ap = argparse.ArgumentParser(prog="emscatt", description=__doc__)
    ap.add_argument("command", choices=["fit", "assemble", "run"])
    ap.add_argument("project")
    ap.add_argument("--only", nargs="*", help="fit only these components")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    proj = load_project(a.project)
    models = None
    if a.command in ("fit", "run"):
        models = fit_components(proj, a.only, a.verbose)
    if a.command in ("assemble", "run"):
        assemble(proj, models, a.verbose)


if __name__ == "__main__":
    main()
