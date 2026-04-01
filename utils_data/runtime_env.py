import os


def _is_valid_positive_int(raw_value):
    if raw_value is None:
        return True
    text = str(raw_value).strip()
    if text == "":
        return False
    if not text.isdigit():
        return False
    return int(text) > 0


def sanitize_openmp_env(default_threads=1):
    env_name = "OMP_NUM_THREADS"
    raw_value = os.getenv(env_name, None)
    if _is_valid_positive_int(raw_value):
        return
    fallback = str(max(int(default_threads), 1))
    os.environ[env_name] = fallback
    print(f"[warn] invalid {env_name}={raw_value!r}; fallback to {fallback}")
