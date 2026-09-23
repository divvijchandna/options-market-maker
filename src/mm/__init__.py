"""options-market-maker: quoting, hedging, fills and PnL attribution."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("options-market-maker")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
