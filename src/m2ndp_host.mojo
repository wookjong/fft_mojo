"""The host-side machinery a task launch runs on.

Files, processes and the toolchain -- the parts of running a task that are not
about tasks. `NDPTask.launch` in m2ndp.mojo is what puts them in order; nothing
here knows what a task is, which is what keeps the dependency one-way.

A host program that names a task has to be linked against sim/host_stubs.c.
See there for why.
"""

from std.ffi import external_call
from std.memory import Span, UnsafePointer
from std.sys import size_of


def _add_export_alias(ir: String, path: String) raises:
    """Give the entry point the name the launcher calls it by.

    `compile_info` emits one function under the mangled name it has inside the
    module. Whether the unmangled `@export` alias comes with it depends on what
    else the module contains -- histogram's does, vector_add's does not -- so
    this adds it when it is missing and leaves it alone when it is not.

    The mangled name is pulled out with one grep pipeline, because the frontend
    spells it with quotes, brackets and commas that are easier matched than
    picked apart by hand. The decision and the append are done here, though: a
    compound shell statement to do all three was one `/bin/sh` parsed less
    reliably than this does.
    """
    # Already unmangled -- a whole-module build produced the plain symbol, and
    # adding an alias would collide with it.
    if ir.find("\ndefine dso_local void @__m2ndp_rt_launch_task(") >= 0:
        return

    # The one line that declares the entry point, mangled, reduced to just its
    # @"..." name.
    var namefile = path + ".name"
    _ = _run(
        "grep -o 'define dso_local void @\"[^\"]*__m2ndp_rt_launch_task"
        + "[^\"]*\"' '" + path + "' | head -1"
        + " | sed 's/^define dso_local void //' > '" + namefile + "'"
    )
    var name: String
    with open(namefile, "r") as f:
        name = String(f.read())
    _ = _run(String("rm -f '") + namefile + "'")
    name = String(name.strip())
    if not name:
        return

    with open(path, "a") as f:
        f.write(
            String("\n@__m2ndp_rt_launch_task = alias void (i64, i64, ptr), ptr ")
            + name + "\n"
        )


# ---------------------------------------------------------------- host memory


struct Arg[T: Copyable & Movable, writes: Bool](Movable):
    """One field of a task's parameter block: a buffer, and which way it goes.

    A task declares its parameters once and both sides read that declaration:

        @fieldwise_init
        struct HistogramParams(Movable):
            var samples: In[Int32]
            var out_hist: Out[Int32]

    On the host a field is built from the caller's list, which is why `launch`
    takes lists and nothing else -- direction and element type are properties
    of the parameter, so they belong in the declaration rather than at every
    call site. On the device the same field is where the launcher put the
    buffer, read back out with `ptr()`.

    An input carries a *copy* of the caller's bytes, taken when the block is
    built. An output carries the caller's *address*, written back into after
    the run. The asymmetry is about lifetimes: a field that only remembered an
    address would not keep the list it points at alive, and the list's last
    mention is often the very expression that built the block, so nothing would
    stop it being freed before the upload reads it. An output escapes that the
    other way, since the caller reads its list after the launch and it is alive
    across the call by construction.

    **One pointer, and nothing else.** A parameter block is as big as the
    buffers it names -- eight bytes a field -- so what the device carries is
    the task's, not the runtime's. Everything a launch needs to know about a
    buffer besides its address lives in a descriptor beside the bytes, off to
    one side of the block entirely.

    That the block is one uniform word per field is also what lets `launch`
    walk it: Mojo has no field reflection, so the number of fields comes from
    dividing the block's size by one field's, and each is read as an `_ArgRec`.
    """

    var ptr: UnsafePointer[Self.T, MutAnyOrigin]
    """Where the buffer is -- and the only thing a kernel wants.

    Two values live here in turn. On the host it addresses this field's
    descriptor, which is where `launch` reads the length and direction from.
    The launcher then overwrites it with the address the buffer landed at on
    the device, and that is what a kernel loads. Neither side sees the other's,
    since the host block stays on the host and the copy in the scratchpad is
    written by the launcher.

    A pointer rather than an address: reading it is then a field load, where
    converting an integer to a pointer costs a stack slot that survives into
    the kernel's frame, our llc not being asked to run the passes that would
    remove it."""

    @implicit
    def __init__(out self, ref data: List[Self.T]):
        """Build from the caller's list. Implicit, so a workload names the list
        and nothing else."""
        var n = len(data) * size_of[Self.T]()
        # The descriptor, and for an input the copy immediately after it. One
        # allocation, freed below. Straight from libc: this is host-only, and
        # a `List` would carry its own three words for nothing.
        var d = external_call["malloc", UnsafePointer[Int, MutAnyOrigin]](
            _DESC_BYTES + (0 if Self.writes else n)
        )
        d[_D_NBYTES] = n
        d[_D_IS_OUT] = 1 if Self.writes else 0
        comptime if Self.writes:
            # An output is downloaded straight back into the caller's list,
            # which is alive across the launch by construction -- the caller
            # reads the result afterwards.
            d[_D_DATA] = Int(data.unsafe_ptr())
        else:
            # An input is copied now, while the list is still the expression
            # being evaluated. A field that only remembered the address would
            # not keep the list alive, and its last mention is often the very
            # expression that built the block.
            var dst = UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(d) + _DESC_BYTES
            )
            var src = UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(data.unsafe_ptr())
            )
            for i in range(n):
                dst[i] = src[i]
            d[_D_DATA] = Int(d) + _DESC_BYTES
        self.ptr = UnsafePointer[Self.T, MutAnyOrigin](
            unsafe_from_address=Int(d)
        )

    def __del__(deinit self):
        """Frees the descriptor, and an input's copy along with it.

        Not `Copyable`: duplicating one would have to duplicate what it owns,
        and nothing needs to -- `launch` takes the block by reference. Leaving
        it out makes an accidental copy a compile error instead."""
        _ = external_call["free", NoneType](
            UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(self.ptr)
            )
        )

    # What `launch` reads, through a block whose element types it has
    # forgotten. Read from the descriptor rather than from `writes`, which an
    # erased field no longer carries.

    @always_inline
    def _desc(self) -> UnsafePointer[Int, MutAnyOrigin]:
        return UnsafePointer[Int, MutAnyOrigin](
            unsafe_from_address=Int(self.ptr)
        )

    @always_inline
    def nbytes(self) -> Int:
        return self._desc()[_D_NBYTES]

    @always_inline
    def is_out(self) -> Bool:
        return self._desc()[_D_IS_OUT] != 0

    @always_inline
    def data(self) -> Int:
        """The bytes themselves: an input's copy, or an output's destination."""
        return self._desc()[_D_DATA]


# A field's descriptor, in words. Host-side only; it never reaches the device.
comptime _D_NBYTES = 0
comptime _D_DATA = 1
comptime _D_IS_OUT = 2
comptime _DESC_BYTES = 3 * size_of[Int]()


comptime In[T: Copyable & Movable] = Arg[T, False]
"""A buffer the task reads."""

comptime Out[T: Copyable & Movable] = Arg[T, True]
"""A buffer the task writes. Pass a list already sized to hold the result and
keep it in scope until the launch returns; reading the result does that."""

comptime _ArgRec = Arg[UInt8, False]
"""One field of a parameter block with its element type forgotten.

`Arg`'s layout does not depend on its parameters, so a block of any task's
fields can be walked as these. That is how `launch` finds what to upload
without field reflection, which Mojo does not have."""


def _write_bytes(path: String, addr: Int, nbytes: Int) raises:
    """Upload: a host buffer's raw bytes to a file the device reads."""
    var ptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)
    with open(path, "w") as f:
        f.write_bytes(Span(ptr=ptr, length=nbytes))


def _read_bytes(path: String, ptr: UnsafePointer[UInt8, MutAnyOrigin],
                cap: Int) raises:
    """Download: a file the device wrote back into a host buffer."""
    with open(path, "r") as f:
        var data = f.read_bytes()
        var n = min(Int(len(data)), cap)
        for i in range(n):
            ptr[i] = data[i]


# ---------------------------------------------------------------- running


def _run(cmd: String) raises -> Int:
    """A shell command, for the steps that are other programs.

    llc, the linker and the simulator are processes; there is no in-process
    form of any of them to call instead.

    The command is copied into a buffer with an explicit trailing NUL first.
    `String.unsafe_ptr()` does not promise one, and `system()` reads to a NUL --
    without it the shell is handed the command plus whatever followed it in
    memory, which surfaces as a spurious "syntax error" on a command that is
    in fact well formed.
    """
    var n = cmd.byte_length()
    var src = cmd.unsafe_ptr()
    var buf = List[UInt8](capacity=n + 1)
    for i in range(n):
        buf.append(src[i])
    buf.append(0)
    var rc = Int(external_call["system", Int32](buf.unsafe_ptr()))
    # system() returns the wait status, not the exit code.
    return (rc >> 8) & 0xFF if rc >= 0 else rc


def _getenv(name: String) -> String:
    var p = external_call[
        "getenv", UnsafePointer[UInt8, MutAnyOrigin]
    ](name.unsafe_ptr())
    if Int(p) == 0:
        return String("")
    var n = Int(external_call["strlen", Int64](p))
    return String(StringSlice(unsafe_from_utf8=Span(ptr=p, length=n)))


def _mktemp() raises -> String:
    """A private workdir the runtime makes for itself. No argument, no
    leftovers -- it is removed when the launch returns.

    `mkdtemp` directly rather than through the shell: it fills the template in
    place with a name no other run will get, which a shell `mktemp` would too
    but only after a round trip through a file to read the name back. The
    buffer is mutable and NUL-terminated because that is what mkdtemp writes
    into and reads."""
    var base = _getenv("TMPDIR")
    var tmpl = (base if base else String("/tmp")) + "/m2ndp-launch-XXXXXX"
    var n = tmpl.byte_length()
    var buf = List[UInt8](capacity=n + 1)
    var src = tmpl.unsafe_ptr()
    for i in range(n):
        buf.append(src[i])
    buf.append(0)
    var made = external_call[
        "mkdtemp", UnsafePointer[UInt8, MutAnyOrigin]
    ](buf.unsafe_ptr())
    if Int(made) == 0:
        raise Error("could not make a working directory")
    return String(StringSlice(unsafe_from_utf8=Span(ptr=made, length=n)))


struct Config(Copyable, Movable):
    """The machine description a run is configured from.

    Which file is an environment setting, not a program's decision:
    `M2NDP_MACHINE_CONFIG` names one, and failing that it is
    `config/machine.conf` under `M2NDP_ROOT`. So the same binary models a
    different machine by being pointed at a different description.

    Host code can read it too. Most workloads have no reason to -- how many
    cores exist is the runtime's business -- but one that genuinely depends on
    the machine can ask:

        var cores = Config.load().get("cores")

    Key = value, one per line, `#` starts a comment. Values are integers,
    because everything the model needs so far is a count.
    """

    var keys: List[String]
    var values: List[Int]
    var path: String

    def __init__(out self, var keys: List[String], var values: List[Int],
                 var path: String):
        self.keys = keys^
        self.values = values^
        self.path = path^

    @staticmethod
    def load() raises -> Config:
        """Read the description the environment points at."""
        var path = _getenv("M2NDP_MACHINE_CONFIG")
        if not path:
            var root = _getenv("M2NDP_ROOT")
            if not root:
                raise Error(
                    "M2NDP_ROOT is not set; point it at the repo root, or set"
                    " M2NDP_MACHINE_CONFIG at a machine description"
                )
            path = root + "/config/machine.conf"

        var text: String
        with open(path, "r") as f:
            text = String(f.read())

        var keys = List[String]()
        var values = List[Int]()
        for line in text.split("\n"):
            var entry = String(String(line).strip())
            if not entry or entry.startswith("#"):
                continue
            var parts = entry.split("=")
            if len(parts) != 2:
                continue
            keys.append(String(String(parts[0]).strip()))
            values.append(Int(String(String(parts[1]).strip())))
        return Config(keys^, values^, path^)

    def get(self, key: String) raises -> Int:
        """The value for `key`, or an error naming the file that lacks it."""
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return self.values[i]
        raise Error(String("no '") + key + "' in " + self.path)


struct Toolchain(Copyable, Movable):
    """Where the tools live and what they are told about the target.

    Discovered from M2NDP_ROOT rather than passed in: which directory the
    toolchain was built in is deployment, not something a task or a launch
    should have to state. The ISA and feature strings match the M²NDP target;
    a task compiled for a different one would need a different toolchain here,
    which is the seam a second target would extend.
    """

    var llc: String
    var lld: String
    var spike: String
    var extlib: String
    var isa: String
    var features: String
    var memory: String
    var link_script: String

    def __init__(out self) raises:
        var root = _getenv("M2NDP_ROOT")
        if not root:
            raise Error("M2NDP_ROOT is not set; point it at the repo root")
        self.llc = root + "/build/llvm/bin/llc"
        self.lld = root + "/build/llvm/bin/ld.lld"
        self.spike = root + "/build/spike/install/bin/spike"
        self.extlib = root + "/build/spike/libm2ndp_ext.so"
        self.isa = String("rv64gcv_zvl128b")
        self.features = String("+m,+a,+f,+d,+v,+zvl128b,+xm2ndp")
        # scripts/m2ndp.lds puts code at 0x10000, below where Spike puts
        # memory by default, and short of the CLINT at 0x2000000.
        self.memory = String("-m0x10000:0x1ff0000")
        self.link_script = root + "/scripts/m2ndp.lds"
