def kget(obj, *keys, default=None):
    for key in keys:
        try:
            obj = obj[key]
        except (KeyError, IndexError):
            return default
    return obj
