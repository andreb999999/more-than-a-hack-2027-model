"""Process set-up shared by the command-line entry points: logging and a fixed thread count.

The gradient-boosting fits sum gradients in parallel, so the thread count is part of the recipe: it is
fixed by the configuration (runtime.threads) for bit-exact reproduction.
"""

import argparse
import contextlib
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from threadpoolctl import threadpool_limits

from .config import DEFAULT_CONFIG, Config, ConfigError, load_config


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging with timestamps.

    Parameters
    ----------
    verbose : bool, optional
        DEBUG level if True, else INFO.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")


def parse_args(description: str) -> argparse.Namespace:
    """Parse the common command line: ``--config`` and ``-v/--verbose``.

    Parameters
    ----------
    description : str
        Help text of the entry point.

    Returns
    -------
    argparse.Namespace
        ``config`` (Path, default ``configs/default.toml``) and ``verbose`` (bool).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML configuration (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args()


@contextlib.contextmanager
def session(description: str) -> Iterator[Config]:
    """Parse the command line, set up logging and the thread limit, and yield the configuration.
    A missing input file or a bad configuration ends the process with a one-line message (exit code 2).

    Parameters
    ----------
    description : str
        Help text of the entry point.

    Yields
    ------
    Config
        The loaded configuration; the body runs with BLAS / OpenMP limited to ``runtime.threads``.
    """
    args = parse_args(description)
    setup_logging(args.verbose)
    log = logging.getLogger("possession")
    try:
        cfg: Config = load_config(args.config)
        log.info("configuration %s, %d threads", args.config, cfg.runtime.threads)
        with threadpool_limits(limits=cfg.runtime.threads):
            yield cfg
    except (FileNotFoundError, ConfigError) as error:
        log.error("%s", error)
        sys.exit(2)
