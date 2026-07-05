"""Platform-wheel builder for the bundled ThinTensor archive core."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parent


class PlatformDistribution(Distribution):
    """Mark wheels as platform-specific because they contain a Rust binary."""

    def has_ext_modules(self) -> bool:
        return True


class BuildWithCore(build_py):
    def run(self) -> None:
        # setuptools reuses build/lib by default and otherwise preserves
        # modules deleted from the source tree. Rebuild this package directory
        # from source so removed commands cannot reappear in a later wheel.
        package_build = Path(self.build_lib) / "thinruntime"
        if package_build.exists():
            shutil.rmtree(package_build)
        super().run()
        if os.environ.get("THINTENSOR_SKIP_CORE_BUILD") == "1":
            return
        subprocess.run(
            [
                "cargo",
                "build",
                "--locked",
                "--release",
                "--bin",
                "thintensor-core",
            ],
            cwd=ROOT,
            check=True,
        )
        suffix = ".exe" if os.name == "nt" else ""
        source = ROOT / "target" / "release" / f"thintensor-core{suffix}"
        destination = (
            Path(self.build_lib)
            / "thinruntime"
            / "bin"
            / f"thintensor-core{suffix}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        destination.chmod(0o755)


setup(
    cmdclass={"build_py": BuildWithCore},
    distclass=PlatformDistribution,
)
