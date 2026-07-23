/* Linker script for an M2NDP task.
 *
 * The scratchpad layout is decided by the compiler, not here -- see
 * RISCVM2ndpLowerScratchpad.cpp. Every scratchpad variable in a task becomes a
 * constant offset from the base pointer the hardware supplies, and the whole
 * set collapses into one block in .spad. So this script has two jobs and
 * neither of them is placement:
 *
 *   1. fail the build if a task asks for more scratchpad than a core has
 *   2. tell the launcher how much of it the task uses
 *
 * The layout the two halves have agreed on is
 *
 *     [ globals, SIZEOF(.spad) bytes ][ arguments ]
 *                                     ^ base register
 *
 * so a launcher that has a region at R passes base = R + __m2ndp_spad_size and
 * writes the task's arguments there. The globals end up below the base at the
 * negative offsets the compiler already emitted.
 */

PHDRS {
  /* One loadable segment, and .spad is explicitly in none of them.
   *
   * Left alone, .spad ends up in a segment because the section is
   * allocatable and has to go somewhere -- GNU ld gives it one of its own at
   * address 0, lld folds it into the following one and drags that down to 0.
   * Either way a loader honours the address and fails: there is no memory at
   * 0 and there was never meant to be. The region is the launcher's to
   * provide and the section reserves a size, nothing more.
   *
   * ":NONE" says so outright. Leaving the assignment off is not the same
   * thing -- lld then inherits the neighbouring segment, which is how the
   * second half of that failure happens.
   */
  dram PT_LOAD;
}

MEMORY {
  /* Per-core scratchpad. Overflowing it is a link error rather than a silent
   * overlap at run time. */
  spad (rw)  : ORIGIN = 0,          LENGTH = 128K
  /* Placeholder: the real code/data address is an M2NDP memory-map fact.
   * It has to stay inside the default code model's +-2GB of zero, or every
   * global reference overflows its R_RISCV_HI20. */
  dram (rwx) : ORIGIN = 0x00010000, LENGTH = 1024M
}

SECTIONS {
  /* NOLOAD: on-chip memory, uninitialized at load, occupies nothing in the
   * file. Nothing addresses these symbols -- the offsets are already in the
   * instructions -- so this only reserves space. */
  .spad (NOLOAD) : {
    *(.spad)
  } > spad :NONE

  /* What the launcher reads to place the base pointer. */
  __m2ndp_spad_size = SIZEOF(.spad);

  .text   : { *(.text.init) *(.text .text.*) } > dram :dram
  .rodata : { *(.rodata .rodata.*) *(.srodata .srodata.*) } > dram :dram
  .data   : { *(.data .data.*) *(.sdata .sdata.*) } > dram :dram
  /* The small-data sections belong here too. Left uncollected they become
   * orphans and the linker puts them where it likes -- which was directly
   * after .spad, at an address nothing loads, and only for the tasks whose
   * .spad is non-empty. The other tasks worked by luck. */
  .bss (NOLOAD) : { *(.bss .bss.* .sbss .sbss.* COMMON) } > dram :dram

  /* HTIF's mailbox. The host finds these by symbol name, so the section only
   * has to exist and be loaded; where it sits does not matter. */
  .tohost : { *(.tohost) } > dram :dram

  /* A launcher runs on a stack; a kernel does not have one to speak of. Kept
   * here rather than in the launcher so that the size is one thing to change
   * when a task needs more. */
  .stack (NOLOAD) : {
    . = ALIGN(16);
    . += 64K;
    __stack_top = .;
  } > dram :dram
}
