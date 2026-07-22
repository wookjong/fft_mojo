"""Vector add — µthread 인덱싱만 쓰는 최소 워크로드.

global_uthread_id()가 __m2ndp_group_id/group_size/uthread_id 세 심볼
호출로 전개되는 것을 확인할 수 있다.
"""

from m2ndp import global_uthread_id, m2ndp_target
from std.gpu.host.compile import _compile_code


def vadd(a: UnsafePointer[Float32, MutAnyOrigin],
         b: UnsafePointer[Float32, MutAnyOrigin],
         c: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    var i = global_uthread_id()
    if i < n:
        c[i] = a[i] + b[i]


def main():
    comptime t = m2ndp_target()
    print(_compile_code[vadd, emission_kind="llvm", target=t]().asm)
