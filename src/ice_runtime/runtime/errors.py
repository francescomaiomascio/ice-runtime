"""
ICE Runtime — Error Definitions
===============================

Questo modulo definisce gli errori FONDATIVI del Runtime ICE.

Regole:
- rappresentano violazioni strutturali
- NON sono recoverable
- NON sono errori di dominio
- NON sono errori di IO
- NON sono errori applicativi

Se uno di questi errori emerge:
→ il Run DEVE abortire
→ il Runtime rimane sovrano
"""


class RuntimeError(Exception):
    """
    Errore base del Runtime ICE.

    Usato esclusivamente per:
    - uso illegittimo dell'API Runtime
    - violazioni di contratto interno
    - incoerenze strutturali

    NON deve essere intercettato da agenti
    o layer superiori.
    """
    pass


class RunNotFoundError(RuntimeError):
    """
    Sollevato quando un RunID non esiste
    nel contesto del Runtime.
    """
    pass


class RunAlreadyExecutedError(RuntimeError):
    """
    Sollevato quando si tenta di rieseguire
    un Run già terminato o abortito.
    """
    pass


class InvalidRunStateError(RuntimeError):
    """
    Sollevato quando un'operazione viene richiesta
    in uno stato del Run non valido.
    """
    pass


class RuntimeInvariantViolation(RuntimeError):
    """
    Sollevato quando un invariante fondativo
    del Runtime viene violato.

    Questo è SEMPRE un bug.
    """
    pass
