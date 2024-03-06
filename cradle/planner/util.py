def get_attr(attr, key, default=None):

    if isinstance(attr, dict):
        return attr.get(key, default)
    else:
        return getattr(attr, key, default)
