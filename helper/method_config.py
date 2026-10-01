"""Single source of truth for the fair FastGRPO/SpecNaacl comparison."""


METHOD_TO_REFLEX_MODE = {
    "fastgrpo": "off",
    "specnaacl": "active",
}


def resolve_method(method):
    normalized = str(method).strip().lower()
    if normalized not in METHOD_TO_REFLEX_MODE:
        choices = ", ".join(sorted(METHOD_TO_REFLEX_MODE))
        raise ValueError(f"METHOD must be one of: {choices}")
    return normalized, METHOD_TO_REFLEX_MODE[normalized]
