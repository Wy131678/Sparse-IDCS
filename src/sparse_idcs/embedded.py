"""Decode losslessly embedded binary-channel inputs."""

import base64
import hashlib
import zlib

import numpy as np


def decode_binary(payload, shape, packed_sha256):
    raw = zlib.decompress(base64.b85decode("".join(payload.split()).encode("ascii")))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != packed_sha256:
        raise ValueError("Embedded input checksum mismatch")
    n = int(np.prod(shape))
    if len(raw) != (n + 7) // 8:
        raise ValueError("Embedded input size does not match its declared shape")
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), count=n).reshape(shape).copy()
