/* The Mojo runtime symbols a device task links against.
 *
 * Mojo's standard formatting (Int/Float to text, used by a DeviceConsole's
 * write) leaves a reference to the allocator's free even when it never runs --
 * the formatting is done in a stack buffer, so nothing is ever allocated to
 * free. The device is freestanding and has no such runtime, so the linker still
 * wants the name. Defining it here as a no-op is what lets a task that prints
 * link. It is the device counterpart of sim/host_stubs.c.
 */

void KGEN_CompilerRT_AlignedFree(void *p) { (void)p; }
