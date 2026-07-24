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


struct Pool(Movable):
    """The memory a task's data lives in, shared with the device.

    M2NDP is near-data processing: the data is already in the CXL pool and the
    cores are attached to it, so there is nothing to upload or download and a
    parameter is a plain pointer. A simulator in its own process does not get
    that for free, so the pool is a file both sides map -- spike through the
    `m2ndp_pool` device, the host here at the same address.

        var pool = Pool()
        var a = pool.alloc[Int32](n)
        a[0] = ...                      # the device reads exactly this

    Allocation bumps a pointer and there is no free: a run is short and the
    pool goes with the process.
    """

    var _path: String
    var _base: Int
    var _bytes: Int
    var _next: Int

    def __init__(out self) raises:
        var config = Config.load()
        self._path = _mktemp() + "/pool.bin"
        self._base = config.get("pool_base")
        self._bytes = config.get("pool_bytes")
        if self._bytes <= 0:
            raise Error("pool_bytes must be positive")

        # `open64`, not `open`: the plain name resolves to Mojo's builtin.
        var fd = Int(external_call["open64", Int32](
            self._path.unsafe_ptr(), Int32(_O_RDWR | _O_CREAT), Int32(0o600)))
        if fd < 0:
            raise Error(String("could not create ") + self._path)
        if Int(external_call["ftruncate", Int32](Int32(fd), self._bytes)) != 0:
            raise Error("could not size the pool file")

        var got = external_call["mmap", UnsafePointer[UInt8, MutAnyOrigin]](
            UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=self._base),
            self._bytes, _PROT_READ | _PROT_WRITE,
            _MAP_SHARED | _MAP_FIXED_NOREPLACE, Int32(fd), 0)
        _ = external_call["close", Int32](Int32(fd))
        if Int(got) != self._base:
            # Landing elsewhere would leave the two sides disagreeing about
            # what a pointer means, so fail rather than continue.
            raise Error(String("pool_base ") + String(self._base)
                        + " is already taken in this process")
        self._next = 0

    def path(self) -> String:
        return self._path

    def base(self) -> Int:
        return self._base

    def bytes(self) -> Int:
        return self._bytes

    def alloc[T: Movable](mut self, count: Int) raises -> UnsafePointer[
        T, MutAnyOrigin
    ]:
        """`count` elements, zeroed. The pointer is a device address too."""
        var bytes = count * size_of[T]()
        var off = (self._next + _POOL_ALIGN - 1) // _POOL_ALIGN * _POOL_ALIGN
        if off + bytes > self._bytes:
            raise Error("the pool is full; raise pool_bytes")
        self._next = off + bytes
        var p = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self._base + off)
        for i in range(bytes):
            p[i] = 0
        return p.bitcast[T]()


comptime _POOL_ALIGN = 64
comptime _O_RDWR = 2
comptime _O_CREAT = 0o100
comptime _PROT_READ = 1
comptime _PROT_WRITE = 2
comptime _MAP_SHARED = 1
comptime _MAP_FIXED_NOREPLACE = 0x100000


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
