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
 * so a launcher passes the region's own address as the base: the compiler laid
 * the globals out from there. The size is what a launcher checks its region
 * against.
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
   * global reference overflows its R_RISCV_HI20.
   *
   * .spad's own addresses (above) are never real memory -- see this file's
   * own docstring: they're offsets the launcher reinterprets against a
   * completely different physical base at run time. But lld doesn't know
   * that; it still checks every section's *numeric* VMA for uniqueness
   * across the whole link, .spad included, even though .spad's segment is
   * :NONE (not loaded) rather than :dram. ORIGIN must therefore clear
   * spad's full LENGTH (128K = 0x20000), or a kernel using more than 64K of
   * scratchpad makes .spad's own top-end VMA collide with dram's -- the
   * failure only a large enough kernel's scratchpad footprint reaches, so a
   * small kernel's clean link doesn't mean this gap is safe. */
  dram (rwx) : ORIGIN = 0x00020000, LENGTH = 1024M
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

  /* -pie (scripts/link-m2ndp.sh) always synthesizes a .dynamic (plus
   * .dynsym/.dynstr/.hash/.rela.dyn, ...) even though nothing here ever
   * loads a real dynamic linker. None of those are named above, so lld
   * treats them as orphans and inserts them whichever place its default
   * heuristic picks -- observed landing right at ORIGIN(dram), the same
   * address .text is explicitly told to start at below, which is only a
   * silent double-booking for a small kernel (nothing yet overlaps .text's
   * own small footprint) and an outright link error once a big enough
   * kernel's .text actually reaches that address. Naming them here first
   * gives them the real few hundred bytes they need before the location
   * counter reaches .text, so .text always starts after them instead of on
   * top of them. */
  .dynamic  : { *(.dynamic) }                     > dram :dram
  .dynsym   : { *(.dynsym) }                       > dram :dram
  .dynstr   : { *(.dynstr) }                       > dram :dram
  .hash     : { *(.hash) }                         > dram :dram
  .gnu.hash : { *(.gnu.hash) }                     > dram :dram
  .rela.dyn : { *(.rela.dyn) }                     > dram :dram

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
