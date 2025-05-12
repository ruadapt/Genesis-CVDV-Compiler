import math
from typing_extensions import Self, Any, Literal #For annotations only

wire_t = tuple[Literal['q']|Literal['qm'],int]

class PiAngle():
    def __init__(this, piMult:float):
        this.mult = piMult
    
    def get_normalized_mult(this):
        out = this.mult
        while out > 1:
            out -= 2
        while out <= -1:
            out += 2
        return out

    def __add__(this, other):
        if other == 0:
            return this
        if other.__class__ == PiAngle:
            other:PiAngle
            out = PiAngle(this.mult + other.mult)
            out.mult = out.get_normalized_mult()
            return out
        return (math.pi * this.mult) + other
    
    def __radd__(this, other):
        return this.__add__(other)

    def __neg__(this):
        return PiAngle(-this.mult)

    def __sub__(this, other):
        return this + (-other)

    def __rsub__(this, other):
        return other + (-this)

    def __mul__(this, other):
        return PiAngle(this.mult * other)

    def __rmul__(this, other):
        return PiAngle(other * this.mult)

    def __truediv__(this, other):
        return PiAngle(this.mult / other)

    def __rtruediv__(this, other):
        return other / (this.mult * math.pi)

    def __eq__(this, other):
        if other.__class__ == PiAngle:
            other:PiAngle = other
            return this.get_normalized_mult() == other.get_normalized_mult()
        return this.get_normalized_mult() * math.pi == other

    def __neq__(this, other):
        return not (this == other)
    
    def __repr__(this) -> str:
        return f"{this.mult}*pi"
    
    def __float__(this):
        return this.mult * math.pi
    
    def __str__(this):
        return this.__repr__()

if __name__ == '__main__':
    #testing, not important
    p = PiAngle(0.5)
    q = PiAngle(-0.2)
    r = PiAngle(1)
    print(p+q)
    print(p*0.8+5-q*0.56)
    print(p*2, p*3, p*4, p*5)
    print(p/6)
    print(-q + p*4 + q == 0)
    print(-r / 4, (-r+PiAngle(1)-PiAngle(1)) / 4)
    print((3*r) / 4, (3*r+PiAngle(1)-PiAngle(1)) / 4)
