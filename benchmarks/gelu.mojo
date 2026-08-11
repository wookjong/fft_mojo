"""GELU — port of M2NDP-public examples/benchmarks/gelu.

Reference kernel, tanh approximation, constants arriving through the
scratchpad (f1 = sqrt(2/pi), f2 = 0.044715, f3 = 0.5):

    vle32.v v1, (ADDR)
    vfmul.vv v2, v1, v1     ; x^2
    vfmul.vf v2, v2, f2
    vfadd.vi v2, v2, 1
    vfmul.vv v2, v2, v1     ; x + 0.044715 x^3
    vfmul.vf v2, v2, f1
    ...                     ; tanh, built out of vfexp
    vfadd.vi v6, v6, 1
    vfmul.vv v6, v6, v1
    vfmul.vf v6, v6, f3     ; 0.5 x (1 + tanh(...))
    vse32.v v6, (x1)

The reference spells tanh out of `vfexp` because that is the instruction it
has, so this does too: `tanh(y) + 1 == 2 / (1 + e**(-2y))`, one `m2ndp.exp`.
The host reference uses the stdlib's tanh, so the two agree within a tolerance.
The constants are ordinary parameters.
"""

from std.sys import argv, size_of
from std.math import tanh
from std.random import random_float64, seed

from m2ndp import (
    VECTOR_WIDTH,
    NDPTask,
    PooledRange,
    exp,
    global_uthread_id,
    launch_parallel,
)
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Float32]()   # lanes in one vector


@fieldwise_init
struct GeluParams(Movable):
    var input: UnsafePointer[Float32, MutAnyOrigin]
    var output: UnsafePointer[Float32, MutAnyOrigin]
    var scale: Float32      # sqrt(2/pi)
    var cubic: Float32      # 0.044715
    var half: Float32


struct Gelu(NDPTask):
    comptime Params = GeluParams

    @staticmethod
    def body():
        ref p = Gelu.params[]
        var i = global_uthread_id() * W
        var x = p.input.load[width=W](i)
        var inner = (x * x * p.cubic + 1) * x * p.scale
        # tanh(inner) + 1 == 2 / (1 + e**(-2 inner)), one `vfexp` for the tanh.
        var gate = 2.0 / (1.0 + exp(inner * -2.0))
        p.output.store(i, gate * x * p.half)

    @staticmethod
    def device_main():
        launch_parallel[Gelu.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh gelu
#
# Compared against the same formula on the host, within a tolerance: the two
# tanh implementations are not bit-identical. See `vector_exp`.


def main() raises:
    if Gelu.emit_ir_if_asked():
        return

    var n = W * 64 * 8
    var scale = Float32(0.7978845608028654)
    var cubic = Float32(0.044715)
    var half = Float32(0.5)

    var input = cxl_alloc[Float32](n)
    var output = cxl_alloc[Float32](n)

    seed(0)
    for i in range(n):
        input[i] = Float32(random_float64(-3.0, 3.0))

    var rc = Gelu.launch(
        PooledRange.over(input, n),
        GeluParams(input, output, scale, cubic, half)
    )
    if rc != 0:
        print("[host] gelu failed, exit", rc)
        return

    for i in range(n):
        var x = input[i]
        var want = (tanh((x * x * cubic + 1) * x * scale) + 1) * x * half
        var err = abs(output[i] - want)
        if err > 1e-5 * abs(want) + 1e-6:
            print("[host] wrong at", i, ":", output[i], "expected", want)
            return
    print("[host] gelu ok")
