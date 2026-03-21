class ModelNotFoundError(Exception):
    """Raised when a model or its associated files cannot be located."""

class InvalidContextError(Exception):
    """Raised when the input context does not contain a lacuna marker."""
