class GreekSchoolsError(Exception):
    """Eccezione base del progetto."""

class ModelNotFoundError(GreekSchoolsError):
    """Modello o file associato non trovato."""

class InvalidContextError(GreekSchoolsError):
    """Il contesto fornito non contiene un marcatore di lacuna."""

class ModelAlreadyExistsError(GreekSchoolsError):
    """Sollevata quando si tenta di creare un modello già presente in MongoDB."""
