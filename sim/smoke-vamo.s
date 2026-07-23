# Smoke test for the M²NDP extension: one indexed vector atomic.
#
# Four lanes hit bins 1, 1, 3 and 0, adding one each, so the array must come
# out {1, 2, 0, 1}. Two lanes land on the same bin deliberately -- that is the
# case a per-lane scalar loop would also get right and a broken indexed atomic
# would not.
#
# Exits 0 when all four match, which is Spike's convention for a passing
# bare-metal test.

        .text
        .globl _start
_start:
        li      t0, (1 << 13) | (1 << 9)
        csrs    mstatus, t0
        vsetivli zero, 4, e32, m1, ta, ma

        la      a0, bins                  # base
        # offsets: bins[1], bins[1], bins[3], bins[0]  (byte offsets)
        la      t1, offs
        vle32.v v12, (t1)
        vmv.v.i v8, 1                     # add 1 per lane

        m2ndp.vamoaddei32.v v8, (a0), v12, v8

        # expect bins = {1, 2, 0, 1}
        lw      t2, 0(a0)
        addi    t2, t2, -1
        lw      t3, 4(a0)
        addi    t3, t3, -2
        or      t2, t2, t3
        lw      t3, 8(a0)
        or      t2, t2, t3
        lw      t3, 12(a0)
        addi    t3, t3, -1
        or      a0, t2, t3                # 0 when all four match

        slli    a0, a0, 1
        ori     a0, a0, 1
        la      t0, tohost
        sd      a0, 0(t0)
1:      j       1b

        .data
        .align 6
offs:   .word 4, 4, 12, 0
bins:   .word 0, 0, 0, 0

        .section .tohost,"aw",@progbits
        .align 6
        .globl tohost
tohost:   .dword 0
        .align 6
        .globl fromhost
fromhost: .dword 0
