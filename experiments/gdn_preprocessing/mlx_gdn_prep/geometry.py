"""Pure shape and addressing contract for the GDN preparation experiment."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    batch: int
    tokens: int
    key_heads: int
    value_heads: int
    taps: int = 4
    head_dim: int = 128

    def __post_init__(self):
        for name in ("batch", "tokens", "key_heads", "value_heads", "taps", "head_dim"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 2**31:
                raise ValueError(f"{name} must be a positive int below 2**31")
        if self.head_dim != 128:
            raise ValueError("This experiment requires key/value head dimensions of 128")
        if not 2 <= self.taps <= 8:
            raise ValueError("Convolution tap count must be between 2 and 8")
        if self.value_heads % self.key_heads:
            raise ValueError("value_heads must be a multiple of key_heads")
        if self.channels >= 2**31 or self.tokens + self.taps >= 2**31:
            raise ValueError("Channel/sequence metadata exceeds signed 32-bit limits")
        if self.batch * self.tokens * self.heads >= 2**32:
            raise ValueError("Fused grid exceeds the 32-bit Metal grid coordinate range")
        if max(self.batch * self.tokens * self.channels,
               self.batch * (self.taps - 1) * self.channels) >= 2**63 // 4:
            raise ValueError("Byte offsets exceed signed 64-bit address range")

    @property
    def heads(self):
        return 2 * self.key_heads + self.value_heads

    @property
    def channels(self):
        return self.heads * self.head_dim

    @property
    def state_shape(self):
        return (self.batch, self.taps - 1, self.channels)


def window_location(batch, position, channel, tokens, channels, history):
    """Return (old_state, element_offset) for a conceptual concatenation."""
    if not (batch >= 0 and 0 <= position < tokens + history and 0 <= channel < channels):
        raise ValueError("Window coordinate is out of range")
    old = position < history
    extent = history if old else tokens
    time = position if old else position - history
    return old, (batch * extent + time) * channels + channel


def clamped_length(value, tokens):
    return min(tokens, max(0, int(value)))
