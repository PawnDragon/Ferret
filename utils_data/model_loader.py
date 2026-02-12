import os


def resolve_model_source(model_name_or_path):
    """
    Prefer local filesystem path when it exists; otherwise fallback to
    HuggingFace model id/path string.
    """
    local_path = os.path.expanduser(model_name_or_path)
    if os.path.exists(local_path):
        return local_path
    return model_name_or_path
