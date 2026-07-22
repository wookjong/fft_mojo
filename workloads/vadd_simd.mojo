"""SIMD vector add — RVV 벡터화 확인용.

Mojo의 SIMD 연산이 RVV 명령(vsetivli/vle32.v/vfadd.vv/vse32.v)으로
코드젠되는지 보여준다.
"""

from m2ndp import global_uthread_id, m2ndp_target


def vadd_simd(a: UnsafePointer[Float32, MutAnyOrigin],
              b: UnsafePointer[Float32, MutAnyOrigin],
              c: UnsafePointer[Float32, MutAnyOrigin]):
    var i = global_uthread_id()
    var va = a.load[width=4](i * 4)
    var vb = b.load[width=4](i * 4)
    c.store(i * 4, va + vb)


def main():
    from std.gpu.host.compile import _compile_code
    comptime t = m2ndp_target()
    print(_compile_code[vadd_simd, emission_kind="llvm", target=t]().asm)
