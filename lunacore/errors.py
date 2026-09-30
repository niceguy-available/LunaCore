"""Exception hierarchy for the LunaCore registration pipeline.

Every phase raises a subclass of :class:`LunaCoreError` so that callers can
distinguish pipeline failures (bad label, empty WMS response, too few matches)
from programming errors.
"""


class LunaCoreError(Exception):
    """Base class for all pipeline errors."""


class LabelParseError(LunaCoreError):
    """The PDS4 label is missing, malformed, or lacks a required field."""


class WMSError(LunaCoreError):
    """The WMS server returned an error, an exception report, or bad data."""


class HarmonizationError(LunaCoreError):
    """Preprocessing could not produce a usable image pair."""


class MatchingError(LunaCoreError):
    """The feature matcher failed or produced too few correspondences."""


class GeometryError(LunaCoreError):
    """Robust estimation failed or produced a degenerate transform."""
