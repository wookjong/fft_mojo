/* What to do when the target faults.
 *
 * Without a handler a trap goes to mtvec, which is zero, so the fetch at zero
 * faults too and the machine spins on that forever. What that looks like from
 * outside is a run that never ends: no message, no exit code, just a timeout.
 * Finding out why then means an instruction trace -- three million lines to
 * recover two numbers the hardware already had.
 *
 * So mtvec points here instead, and a fault says what it was and stops.
 */

#include "htif.h"

/* Saved before anything else runs, so the dump is the state at the fault
 * rather than the state after the handler got going.
 *
 * Not static: the entry sequence names it from inline assembly, which the
 * compiler cannot see, so a local symbol is not guaranteed to survive. */
u64 panic_regs[32];

static const char *const reg_name[32] = {
    "zero", "ra", "sp",  "gp",  "tp", "t0", "t1", "t2", "s0", "s1", "a0",
    "a1",   "a2", "a3",  "a4",  "a5", "a6", "a7", "s2", "s3", "s4", "s5",
    "s6",   "s7", "s8",  "s9",  "s10", "s11", "t3", "t4", "t5", "t6",
};

static const char *cause_name(u64 cause)
{
    if (cause >> 63)
        return "interrupt";
    switch (cause) {
    case 0: return "instruction address misaligned";
    case 1: return "instruction access fault";
    case 2: return "illegal instruction";
    case 3: return "breakpoint";
    case 4: return "load address misaligned";
    case 5: return "load access fault";
    case 6: return "store address misaligned";
    case 7: return "store access fault";
    case 8: return "ecall from U";
    case 9: return "ecall from S";
    case 11: return "ecall from M";
    case 12: return "instruction page fault";
    case 13: return "load page fault";
    case 15: return "store page fault";
    default: return "unknown";
    }
}

static void print_hex(u64 v)
{
    char buf[19];
    buf[0] = '0';
    buf[1] = 'x';
    for (int i = 0; i < 16; i++) {
        unsigned nib = (unsigned)(v >> (60 - 4 * i)) & 0xf;
        buf[2 + i] = (char)(nib < 10 ? '0' + nib : 'a' + nib - 10);
    }
    buf[18] = 0;
    htif_print(buf);
}

static void print_pad(const char *s, int width)
{
    htif_print(s);
    int n = 0;
    while (s[n])
        n++;
    while (n++ < width)
        htif_print(" ");
}

/* Called from the handler once the registers are safe. */
void m2ndp_panic_report(u64 cause, u64 epc, u64 tval, u64 status)
{
    htif_print("\n*** M2NDP panic: ");
    htif_print(cause_name(cause));
    htif_print("\n");

    htif_print("  mcause  ");
    print_hex(cause);
    htif_print("\n  mepc    ");
    print_hex(epc);
    htif_print("   <- the faulting instruction\n  mtval   ");
    print_hex(tval);
    htif_print("   <- the address or the instruction bits\n  mstatus ");
    print_hex(status);
    htif_print("\n\n  registers at the fault:\n");

    for (int i = 1; i < 32; i++) {
        htif_print("    ");
        print_pad(reg_name[i], 5);
        print_hex(panic_regs[i]);
        htif_print((i % 2) ? "  " : "\n");
    }
    /* No disassembly here on purpose: the host has llvm-objdump and knows
     * the vendor extension, so it can show real instructions around mepc
     * rather than the raw words a target-side dump could manage.
     * scripts/simulate.sh does that with the address printed above. */

    /* Distinct from the launcher's own failures, which use 2. */
    htif_exit(3);
}

/* Every register is saved before a single C instruction runs, because C would
 * start using them. t0 goes to mscratch first, since the base pointer for the
 * rest has to live somewhere. */
__attribute__((naked, aligned(4))) void m2ndp_trap_entry(void)
{
    __asm__ volatile(
        "csrw mscratch, t0\n"
        "la   t0, panic_regs\n"
        "sd   x1,   8(t0)\n"
        "sd   x2,  16(t0)\n"
        "sd   x3,  24(t0)\n"
        "sd   x4,  32(t0)\n"
        "sd   x6,  48(t0)\n"
        "sd   x7,  56(t0)\n"
        "sd   x8,  64(t0)\n"
        "sd   x9,  72(t0)\n"
        "sd   x10, 80(t0)\n"
        "sd   x11, 88(t0)\n"
        "sd   x12, 96(t0)\n"
        "sd   x13,104(t0)\n"
        "sd   x14,112(t0)\n"
        "sd   x15,120(t0)\n"
        "sd   x16,128(t0)\n"
        "sd   x17,136(t0)\n"
        "sd   x18,144(t0)\n"
        "sd   x19,152(t0)\n"
        "sd   x20,160(t0)\n"
        "sd   x21,168(t0)\n"
        "sd   x22,176(t0)\n"
        "sd   x23,184(t0)\n"
        "sd   x24,192(t0)\n"
        "sd   x25,200(t0)\n"
        "sd   x26,208(t0)\n"
        "sd   x27,216(t0)\n"
        "sd   x28,224(t0)\n"
        "sd   x29,232(t0)\n"
        "sd   x30,240(t0)\n"
        "sd   x31,248(t0)\n"
        /* and t0 itself, from where it was stashed */
        "csrr t1, mscratch\n"
        "sd   t1, 40(t0)\n"
        /* A fault in the launcher leaves sp usable; one deep in a kernel may
         * not. Either way the report needs a stack of its own. */
        "la   sp, __stack_top\n"
        "csrr a0, mcause\n"
        "csrr a1, mepc\n"
        "csrr a2, mtval\n"
        "csrr a3, mstatus\n"
        "tail m2ndp_panic_report\n");
}
