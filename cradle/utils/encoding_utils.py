import base64


def encode_base64(payload):

    if payload is None:
        raise ValueError("Payload cannot be None.")

    return base64.b64encode(payload).decode('utf-8')


def decode_base64(payload):

    if payload is None:
        raise ValueError("Payload cannot be None.")

    return base64.b64decode(payload)
