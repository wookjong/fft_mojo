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


struct Buffer(Copyable, Movable):
    """A host buffer handed to a task, and which way it goes.

    An input carries a *copy* of the caller's bytes, taken when the buffer is
    made. An output carries the caller's *address*, written back into after the
    run. The asymmetry is deliberate, and it is about lifetimes.

    A buffer that only remembered an address would not keep the list it points
    at alive: the list's last mention is the `Buffer.input(xs)` that made the
    buffer, so nothing stops it being freed before the launch reads it -- and
    the upload would dump whatever reused that memory. Copying the bytes up
    front, while the list is still the argument being evaluated, sidesteps that
    entirely. An output escapes it the other way: the caller reads its list
    after the launch, so it is alive across the call by construction.

    Bytes rather than elements, because at this level a buffer is just memory
    to move to and from a file. The task's own params struct is what gives the
    bytes a type again on the device.
    """

    var data: List[UInt8]  # an input's copied bytes; empty for an output
    var out_addr: Int      # an output's destination; 0 for an input
    var nbytes: Int
    var is_out: Bool

    def __init__(out self, var data: List[UInt8], out_addr: Int, nbytes: Int,
                 is_out: Bool):
        self.data = data^
        self.out_addr = out_addr
        self.nbytes = nbytes
        self.is_out = is_out

    @staticmethod
    def input[T: Copyable & Movable](ref data: List[T]) -> Buffer:
        """A buffer the task reads. Its bytes are copied now, then uploaded."""
        var n = len(data) * size_of[T]()
        var src = data.unsafe_ptr().bitcast[UInt8]()
        var copy = List[UInt8](capacity=n)
        for i in range(n):
            copy.append(src[i])
        return Buffer(copy^, 0, n, False)

    @staticmethod
    def output[T: Copyable & Movable](ref data: List[T]) -> Buffer:
        """A buffer the task writes. Downloaded after the run, into `data`.

        Pass a list already sized to hold the result; the run fills it. Keep it
        in scope until the launch returns -- reading the result does that."""
        return Buffer(
            List[UInt8](), Int(data.unsafe_ptr().bitcast[UInt8]()),
            len(data) * size_of[T](), True,
        )

    def out_ptr(self) -> UnsafePointer[UInt8, MutAnyOrigin]:
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.out_addr
        )


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
