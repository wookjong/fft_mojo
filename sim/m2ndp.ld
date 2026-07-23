/* Link script for running M²NDP kernels under Spike.
 *
 * Two things it has to get right that a default link does not:
 *
 *   - The ELF headers must not be pulled in front of .text. Left alone, the
 *     linker starts the first LOAD segment a page below the text address to
 *     make room for them, and Spike refuses to load below its memory base.
 *
 *   - .spad needs an address. The compiler emits scratchpad globals into that
 *     section but says nothing about where it lives; here is where the
 *     simulator's answer goes. See docs/INTERFACE.md.
 */

OUTPUT_ARCH("riscv")
ENTRY(_start)

/* Spike's default memory starts at 0x80000000. */
MEMORY_BASE = 0x80000000;

/* Scratchpad. On hardware this is on-chip memory private to an NDP core; here
 * it is ordinary memory at a fixed address, which is all a functional model
 * needs. One instance, since only one core is modelled. */
SPAD_BASE = 0x88000000;
SPAD_SIZE = 0x00010000;

SECTIONS
{
  /* Leave the headers room ahead of the code rather than below it. */
  . = MEMORY_BASE + SIZEOF_HEADERS;
  . = ALIGN(0x1000);

  .text : { *(.text.init) *(.text) *(.text.*) }
  .rodata : { *(.rodata) *(.rodata.*) *(.srodata) *(.srodata.*) }
  .data : { *(.data) *(.data.*) *(.sdata) *(.sdata.*) }
  .bss (NOLOAD) : { *(.bss) *(.bss.*) *(.sbss) *(.sbss.*) *(COMMON) }

  /* HTIF: how a bare-metal program tells Spike it is done. The symbols have
   * to be called tohost and fromhost, and Spike looks them up by name. */
  . = ALIGN(64);
  .tohost : { *(.tohost) }

  . = SPAD_BASE;
  .spad (NOLOAD) : { *(.spad) . = SPAD_SIZE; }

  /DISCARD/ : { *(.comment) *(.note*) *(.riscv.attributes) }
}
