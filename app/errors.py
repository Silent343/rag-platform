"""Errors used to translate upstream AI failures into useful API responses."""


class ProviderError(RuntimeError):
    """The configured LLM or embedding provider could not complete a request."""

    def __init__(self, message: str = "The AI provider is temporarily unavailable.") -> None:
        super().__init__(message)
