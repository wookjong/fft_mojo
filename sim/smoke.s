# Smoke test: prove the path from our assembler through the linker into Spike.
#
# It computes something with RVV and checks the answer in target code, so the
# exit code is the verdict: 0 for pass, which is also Spike's own convention
# for a bare-metal test.
#
# Deliberately uses no M²NDP instruction. What is under test here is the
# pipeline -- assemble, link, load, execute, report -- so that when the
# extension is added, a failure means the extension.

        .text
        .globl _start
_start:
        # Bare metal starts with the FP and vector units off; touching either
        # traps as an illegal instruction until mstatus says otherwise. The
        # M²NDP runtime will have to do this too, whatever form it takes.
        li      t0, (1 << 13) | (1 << 9)   # FS=Initial, VS=Initial
        csrs    mstatus, t0

        vsetivli zero, 4, e32, m1, ta, ma
        vmv.v.i  v8, 3
        vmv.v.i  v9, 4
        vadd.vv  v10, v8, v9
        vmv.x.s  a0, v10                   # lane 0 of 3 + 4

        li      a1, 7
        sub     a0, a0, a1                 # 0 when correct

        # HTIF: bit 0 marks the word as a command, the rest is the exit code.
        slli    a0, a0, 1
        ori     a0, a0, 1
        la      t0, tohost
        sd      a0, 0(t0)
1:      j       1b

        .section .tohost,"aw",@progbits
        .align 6
        .globl tohost
tohost:   .dword 0
        .align 6
        .globl fromhost
fromhost: .dword 0
