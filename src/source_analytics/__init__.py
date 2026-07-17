"""Source Analytics: Statistical analysis toolkit for source-localized EEG data."""

from source_analytics._version import get_version, git_describe

__version__ = get_version()

__all__ = ["__version__", "get_version", "git_describe"]
