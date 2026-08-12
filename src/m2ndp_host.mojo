"""The host-side machinery a task launch runs on.

Files, processes and the toolchain -- the parts of running a task that are not
about tasks. `NDPTask.launch` in m2ndp.mojo is what puts them in order; nothing
here knows what a task is, which is what keeps the dependency one-way.

A host program that names a task has to be linked against sim/host_stubs.c.
See there for why.
"""

from std.ffi import _Global, external_call
from std.memory import Span, UnsafePointer
from std.os import abort
from std.sys import size_of


def _align_masked_gathers(ir: String) -> String:
    """Give each `llvm.masked.gather` the alignment its element type has.

    The frontend takes an alignment and emits the intrinsic without one.
    ScalarizeMaskedMemIntrin reads a missing attribute as align 1, RISC-V calls
    a gather narrower than its element illegal, and the pass scalarizes it to
    one `lbu` per byte. With `<8 x ptr> align 4` the same IR selects
    `vluxei64.v v8, (a0), v12, v0.t`.

    Comes out when the frontend emits the attribute itself.
    """
    comptime MARK = "@llvm.masked.gather."
    var out = String()
    var at = 0
    while True:
        var start = ir.find(MARK, at)
        if start < 0:
            return out + ir[byte=at:]
        var open = ir.find("(", start)
        if open < 0:
            return out + ir[byte=at:]
        var close = ir.find(">", open)   # end of the `<N x ptr>` operand
        if close < 0:
            return out + ir[byte=at:]

        # `v8i32.v8p0` -- the element width, past the lane count.
        var overload = String(ir[byte = start + len(MARK) : open])
        var dot = overload.find(".")
        var bits = 0
        for i in range(dot if dot > 0 else len(overload)):
            var c = overload[byte = i : i + 1]
            if c == "i" or c == "f":
                try:
                    bits = Int(String(overload[byte = i + 1 : dot]))
                except:
                    bits = 0
                break

        out += ir[byte = at : close + 1]
        var tail = String(ir[byte = close + 1 : close + 8])
        if bits >= 8 and not tail.startswith(" align"):
            out += " align " + String(bits // 8)
        at = close + 1


def _add_export_alias(path: String) raises:
    """Give the entry point the name the launcher calls it by.

    `compile_info` emits the entry point under the mangled name it has inside
    the module, and this is where it gets the fixed one. The task does not
    export that name itself: it is one per binary, and a host program may hold
    several tasks.

    The mangled name is pulled out with one grep pipeline, because the frontend
    spells it with quotes, brackets and commas that are easier matched than
    picked apart by hand. The decision and the append are done here, though: a
    compound shell statement to do all three was one `/bin/sh` parsed less
    reliably than this does.
    """
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
            String("\n@__m2ndp_rt_launch_task = alias void (i64, i64), ptr ")
            + name + "\n"
        )


# ---------------------------------------------------------------- host memory


struct Pool(Movable):
    """The memory a task's data lives in, shared with the device.

    M2NDP is near-data processing: the data is already in the CXL pool and the
    cores are attached to it, so there is nothing to upload or download and a
    parameter is a plain pointer. A simulator in its own process does not get
    that for free, so the pool is a file both sides map at the same address --
    the host here, and the simulator when it attaches it.

        var a = cxl_alloc[Int32](n)     # the process's pool
        a[0] = ...                      # the device reads exactly this

    Allocation bumps a pointer and there is no free: a run is short and the
    pool goes with the process.
    """

    var _path: String
    var _base: Int
    var _bytes: Int
    var _data_limit: Int
    var _next: Int

    def __init__(out self) raises:
        var config = Config.load()
        self._path = _mktemp() + "/pool.bin"
        self._base = _POOL_BASE + _addr_offset()
        self._bytes = config.get("memory_expander_size")
        if self._bytes <= 0:
            raise Error("memory_expander_size must be positive")

        # The stacks sit at the top of the pool (see address_map.h); the device
        # sizes them from these same counts, so data must stop below them.
        var slots = config.get("num_ndp_units") * config.get("num_sub_core") * config.get("uthread_slots")
        var stack_region = (config.get("controller_stack_size") + _GUARD_SIZE) \
                         + slots * (config.get("uthread_stack_size") + _GUARD_SIZE)
        self._data_limit = self._bytes - stack_region
        if self._data_limit <= 0:
            raise Error("memory_expander_size is too small for the machine's stacks")

        # `open64`, not `open`: the plain name resolves to Mojo's builtin.
        var cpath = List[UInt8](capacity=self._path.byte_length() + 1)
        var pp = self._path.unsafe_ptr()
        for i in range(self._path.byte_length()):
            cpath.append(pp[i])
        cpath.append(0)
        var fd = Int(external_call["open64", Int32](
            cpath.unsafe_ptr(), Int32(_O_RDWR | _O_CREAT), Int32(0o600)))
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
            raise Error(String("pool base ") + String(self._base)
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
        if off + bytes > self._data_limit:
            raise Error("the pool is full; raise memory_expander_size")
        self._next = off + bytes
        var p = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self._base + off)
        for i in range(bytes):
            p[i] = 0
        return p.bitcast[T]()


comptime _POOL_ALIGN = 64
# Mirror of address_map.h: the pool base and guard-page size are fixed there.
comptime _POOL_BASE = 0x40000000
comptime _GUARD_SIZE = 0x1000
comptime _O_RDWR = 2
comptime _O_CREAT = 0o100
comptime _PROT_READ = 1
comptime _PROT_WRITE = 2
comptime _MAP_SHARED = 1
comptime _MAP_FIXED_NOREPLACE = 0x100000


# ---------------------------------------------------------------- the pool
#
# M2NDP has one CXL pool, so the runtime holds one and hands it out. A workload
# calls `cxl_alloc` and never names it; `launch` reaches the same one.

def _init_pool() -> Pool:
    try:
        return Pool()
    except e:
        abort(String("could not open the CXL pool: ") + String(e))


comptime _pool = _Global["m2ndp_pool", _init_pool]


def cxl_pool(out result: UnsafePointer[Pool, MutUntrackedOrigin]):
    """The process's CXL pool, created on first use."""
    try:
        result = _pool.get_or_create_ptr()
    except e:
        abort(String("could not open the CXL pool: ") + String(e))


def cxl_alloc[T: Movable](count: Int) raises -> UnsafePointer[T, MutAnyOrigin]:
    """`count` elements of `T` in the CXL pool, zeroed. The pointer is a device
    address too, the pool being memory both sides map."""
    return cxl_pool()[].alloc[T](count)


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


def _addr_offset() raises -> Int:
    """The device address map's build-time shift (see address_map.h); 0 unless
    `M2NDP_ADDR_OFFSET` is set, which an ASAN build does. Sim/launcher/host agree."""
    var s = _getenv("M2NDP_ADDR_OFFSET")
    return _parse_uint(s) if s else 0


def _parse_uint(s: String) raises -> Int:
    """A decimal or `0x`-prefixed integer; raises on anything else."""
    var t = String(s.strip())
    if t.startswith("0x") or t.startswith("0X"):
        var data = t.as_bytes()
        var v = 0
        for i in range(2, len(data)):
            var c = Int(data[i])
            var d: Int
            if c >= ord("0") and c <= ord("9"):
                d = c - ord("0")
            elif c >= ord("a") and c <= ord("f"):
                d = c - ord("a") + 10
            elif c >= ord("A") and c <= ord("F"):
                d = c - ord("A") + 10
            else:
                raise Error("not a hex digit")
            v = v * 16 + d
        return v
    return Int(t)


struct Config(Copyable, Movable):
    """The machine description a run is configured from -- the same simulator
    config the device runs on, so the build and the run see one machine.

    `M2NDP_CONFIG` names the file; failing that it is the M2NDP performance
    config under the detour submodule (`M2NDP_DET`, else `M2NDP_ROOT`). Host code
    can read a count from it:

        var units = Config.load().get("num_ndp_units")

    Key = value, one per line, `#` starts a comment. Only integer-valued keys are
    kept (decimal or `0x` hex); the config's non-integer entries are ignored.
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
        var path = _getenv("M2NDP_CONFIG")
        if not path:
            var det = _getenv("M2NDP_DET")
            if not det:
                var root = _getenv("M2NDP_ROOT")
                if not root:
                    raise Error(
                        "set M2NDP_CONFIG at a simulator config, or M2NDP_ROOT at"
                        " the repo root"
                    )
                det = root + "/third_party/m2ndp-detour"
            path = det + "/config/performance/M2NDP/m2ndp.config"

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
            try:
                var v = _parse_uint(String(parts[1]))
                keys.append(String(String(parts[0]).strip()))
                values.append(v)
            except:
                continue  # a non-integer entry (a path, a mapping string): skip it
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
    var features: String
    # M²NDP-Detour, the timing simulator a launch runs on.
    var det: String
    var runner: String
    var det_config: String
    var link_m2ndp: String

    def __init__(out self) raises:
        var root = _getenv("M2NDP_ROOT")
        if not root:
            raise Error("M2NDP_ROOT is not set; point it at the repo root")
        self.llc = root + "/build/llvm/bin/llc"
        self.features = String("+m,+a,+f,+d,+v,+zvl128b,+zfh,+zvfh,+xm2ndp")
        self.det = _getenv("M2NDP_DET")
        if not self.det:
            self.det = root + "/third_party/m2ndp-detour"
        self.runner = self.det + "/build/bin/m2ndp_run"
        self.det_config = _getenv("M2NDP_CONFIG")
        if not self.det_config:
            self.det_config = self.det + "/config/performance/M2NDP/m2ndp.config"
        self.link_m2ndp = root + "/scripts/link-m2ndp.sh"
