ENVIORNMENT_REGISTRY = {}

def register_environment(name):
    def decorator(env):
        ENVIORNMENT_REGISTRY[name] = env
        return env
    return decorator
