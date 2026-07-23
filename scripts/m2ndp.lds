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
  /* One loadable segment, and .spad is deliberately not in it.
   *
   * Without this the linker wraps .spad in a segment of its own at address 0,
   * because the section is allocatable and has to go somewhere. A loader then
   * tries to honour that address and fails -- there is no memory at 0, and
   * there was never meant to be: the region is the launcher's to provide. The
   * section reserves a size and nothing else.
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
  } > spad

  /* What the launcher reads to place the base pointer. */
  __m2ndp_spad_size = SIZEOF(.spad);

  .text   : { *(.text .text.*) }     > dram :dram
  .rodata : { *(.rodata .rodata.*) } > dram :dram
  .data   : { *(.data .data.*) }     > dram :dram
  .bss (NOLOAD) : { *(.bss .bss.* COMMON) } > dram :dram
}
