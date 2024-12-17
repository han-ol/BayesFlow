import asyncio
from asyncio.subprocess import PIPE
from logging import getLogger
from subprocess import CalledProcessError
import os
from pathlib import Path

from sphinx_polyversion.api import apply_overrides
from sphinx_polyversion.builder import BuildError
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.git import Git, file_predicate, refs_by_type
from sphinx_polyversion.pyvenv import VirtualPythonEnvironment
from sphinx_polyversion.sphinx import SphinxBuilder, Placeholder
from sphinx_polyversion.utils import to_thread

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    cast,
)

logger = getLogger(__name__)

#: Determine repository root directory
root = Git.root(Path(__file__).parent)

#: CodeRegex matching the branches to build docs for
BRANCH_REGEX = r"^(master)$"

#: Regex matching the tags to build docs for
TAG_REGEX = r"^v[\.0-9]*$"
TAG_REGEX = r"^v1.1.6$"

#: Output dir relative to project root
OUTPUT_DIR = "_build_polyversion"

#: Source directory
SOURCE_DIR = "docsrc"

#: Arguments to pass to `sphinx-build`
SPHINX_ARGS = "-a -v"

#: Extra packages for building docs
SPHINX_DEPS = [
    "sphinx",
    "numpydoc",
    "myst-nb",
    "sphinx_design",
    "sphinx-book-theme",
    "sphinxcontrib-bibtex",
    "sphinx-polyversion==1.0.0",
]

#: Extra dependencies to iinstall for version 1
V1_BACKEND_DEPS = [
    "tf-keras",
]

#: Extra dependencies to install for version 2
V2_BACKEND_DEPS = [
    "jax",
    "torch",
    "tensorflow",
]


#: Data passed to templates
def data(driver, rev, env):
    revisions = driver.targets
    branches, tags = refs_by_type(revisions)
    latest = max(tags or branches)
    # sort tags and branches by date, newest first
    return {
        "current": rev,
        "tags": sorted(tags, reverse=True),
        "branches": sorted(branches, reverse=True),
        "revisions": revisions,
        "latest": latest,
    }


def root_data(driver):
    revisions = driver.builds
    branches, tags = refs_by_type(revisions)
    latest = max(tags or branches)
    return {"revisions": revisions, "latest": latest}


# Load overrides read from commandline to global scope
apply_overrides(globals())


# adapted from Pip
class DynamicPip(VirtualPythonEnvironment):
    """
    Build Environment for using a venv and installing deps with pip.
    The name is added to the path to allow distinct virtual environments
    for each revision.

    Use this to run the build commands in a python virtual environment
    and install dependencies with pip into the venv before the build.


    Parameters
    ----------
    path : Path
        The path of the current revision.
    name : str
        The name of the environment (usually the name of the revision).
    venv : Path
        The path of the python venv.
    args : Iterable[str]
        The cmd arguments to pass to `pip install`.
    creator : Callable[[Path], Any] | None, optional
        A callable for creating the venv, by default None
    env_vars: Dict[str, str], optional, default []
        A dictionary of environment variables passed to `pip install`
    """

    def __init__(
        self,
        path: Path,
        name: str,
        venv: str | Path,
        *,
        args: Iterable[str],
        creator: Callable[[Path], Any] | None = None,
        env_vars: Dict[str, str] = {},
    ):
        """
        Build Environment for using a venv and pip.

        Parameters
        ----------
        path : Path
            The path of the current revision.
        name : str
            The name of the environment (usually the name of the revision).
        venv : Path
            The path of the python venv.
        args : Iterable[str]
            The cmd arguments to pass to `pip install`.
        creator : Callable[[Path], Any], optional
            A callable for creating the venv, by default None

        """
        logger.info("Setting dynamic venv name: " + str(Path(venv) / name))
        self.env_vars = env_vars
        self.args = args.copy()
        if name.startswith("v1."):
            # required, as setup-scm cannot determine the version without
            # the .git directory, which is not copied for the build.
            logger.info("Setting setuptools version 'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BAYESFLOW=" + name[1:] + "'")
            self.env_vars["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BAYESFLOW"] = name[1:]

        super().__init__(path, name, Path(venv) / name, creator=creator)

    async def __aenter__(self):
        """
        Set the venv up.

        Raises
        ------
        BuildError
            Running `pip install` failed.
        """
        await super().__aenter__()

        logger.info("Running `pip install`...")

        cmd: list[str] = ["pip", "install"]
        cmd += self.args

        env = self.activate(os.environ.copy())
        # add environment variables to environment
        for key, value in self.env_vars.items():
            env[key] = value

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.path,
            env=env,
            stdout=PIPE,
            stderr=PIPE,
        )
        out, err = await process.communicate()
        out = out.decode(errors="ignore")
        err = err.decode(errors="ignore")

        self.logger.debug("Installation output:\n %s", out)
        if process.returncode != 0:
            raise BuildError from CalledProcessError(cast(int, process.returncode), " ".join(cmd), out, err)
        return self


# for some reason, VenvWrapper did not work for me (probably something specific
# to my system, so we use subprocess.call directly to use the system utilities
# to create the environment.
class LocalVenvCreator:
    def _create(self, venv_path):
        if not os.path.exists(venv_path):
            import subprocess

            print(f"Creating venv '{venv_path}'...")
            subprocess.call(f"python -m venv {venv_path}", shell=True)

    async def __call__(self, path: Path) -> None:
        await to_thread(self._create, path)


# Setup environments for the different versions
src = Path(SOURCE_DIR)
vcs = Git(
    branch_regex=BRANCH_REGEX,
    tag_regex=TAG_REGEX,
    buffer_size=1 * 10**9,  # 1 GB
    predicate=file_predicate([src]),  # exclude refs without source dir
)


creator = LocalVenvCreator()


async def selector(rev, keys):
    """Select configuration based on revision"""
    # map all v1 revisions to one configuration
    if rev.name.startswith("v1."):
        return "v1"
    elif rev.name in ["master"]:
        # special configuration for v1 master branch
        return rev.name
    # common config for everything else
    return None


ENVIRONMENT = {
    # configuration for v2 and dev
    None: DynamicPip.factory(venv=Path(".venv"), args=["-e", "."] + SPHINX_DEPS + V2_BACKEND_DEPS, creator=creator),
    # configuration for v1 and master (remove master here and in selector when it moves to v2)
    "v1": DynamicPip.factory(venv=Path(".venv"), args=["-e", "."] + SPHINX_DEPS + V1_BACKEND_DEPS, creator=creator),
    "master": DynamicPip.factory(
        venv=Path(".venv"),
        args=["-vv", "-e", "."] + V1_BACKEND_DEPS + SPHINX_DEPS,
        creator=creator,
        env_vars={"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BAYESFLOW": "1.1.6dev"},
    ),
}

# Setup driver and run it
DefaultDriver(
    root,
    OUTPUT_DIR,
    vcs=vcs,
    builder=SphinxBuilder(
        src / "source",
        args=SPHINX_ARGS.split(),
        pre_cmd=["python", root / src / "pre-build.py", Placeholder.SOURCE_DIR],
    ),
    env=ENVIRONMENT,
    selector=selector,
    template_dir=root / src / "polyversion/templates",
    static_dir=root / src / "polyversion/static",
    data_factory=data,
    root_data_factory=root_data,
).run(False)