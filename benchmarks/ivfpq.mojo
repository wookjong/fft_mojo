"""IVF-PQ search — port of M2NDP-Detour benchmarks/ivf_pq_whole.

Reference kernel (Detour `.ndp.cc`, one launch = one query):

    kernel_body void cdist   (...) { do_coarse_dist(f); }
    kernel_body void csel    (...) { do_coarse_sel();   }
    kernel_body void precomp0(...) { do_precomp(0,f,u); }   #if C_CLUSTERS > 0
    kernel_body void lut0    (...) { do_lut    (0,f,u); }
    ...                                                     #if C_CLUSTERS > 7
    kernel_body void publish (...) { do_publish(o);     }

One query, searched the way IVF-PQ searches it: rank the coarse centroids, take
the nearest NPROBE, and for each of those clusters build a distance table over
the codebook and score the cluster's PQ codes through it. A core keeps its
running top-K in scratchpad and folds each cluster into it, so a cluster's
scores never reach DRAM.

The reference spells its cluster stages out as eight `#if`-guarded copies
because its toolchain has no block type that can branch, so a loop over
clusters cannot cross a launch boundary. Here `device_main` is ordinary
controller code and the loop is a loop -- which is also what lifts the
reference's ceiling of eight clusters per core.

Work splits the same way it does there. Core `g` owns probe ranks
`g, g+cores, g+2*cores, ...`, and every core runs the whole coarse search
rather than sharing it: scratchpad is per core, so splitting the centroids
would cost a round trip through DRAM, while recomputing costs a loop the cores
run in lockstep. Every core therefore reaches the same probe list and needs no
communication to agree on who takes which cluster.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import (
    PACKET,
    Machine,
    NDPTask,
    PooledRange,
    group_id,
    launch_parallel,
    launch_serial,
    local_uthread_id,
    scratchpad,
)
from m2ndp_host import cxl_alloc, Config

# ------------------------------------------------------------ the index
#
# Geometry is compile-time here as it is in the reference, because the
# scratchpad arrays are sized from it. `CLUSTERS_PER_CORE` is the one value
# that has to agree with the machine: the reference's run.sh derived it from
# `nprobe` and a core count, and got it wrong once by deriving it from a stride
# the simulator was not using. `main` reads the same config the runtime does
# and checks it instead of deriving it twice.

comptime VECTOR_DIM = 64        # dimensions per vector
comptime NLIST = 16             # coarse centroids
comptime LIST_SIZE = 128        # vectors per cluster
comptime NPROBE = 4             # clusters searched per query
comptime TOPK = 8               # results returned
comptime PQ_M = 32              # subquantizers
comptime CLUSTERS_PER_CORE = 4  # >= ceil(NPROBE / cores); `main` checks it
comptime UTHREADS_PER_CORE = 64 # microthreads a core runs; stride = this * packet (ref default)

comptime PQ_SHIFTS = 256                     # codebook entries per subquantizer
comptime SUBVECTOR_DIM = VECTOR_DIM // PQ_M  # dsub
comptime LUT_SIZE = PQ_M * PQ_SHIFTS
comptime W = PACKET // size_of[Float32]()    # lanes in one packet
comptime INF = Float32(3.0e38)


@fieldwise_init
struct IvfPqParams(Movable):
    """The task's parameters, declared once for both sides.

    `uthreads` is how many microthreads share one core. The reference took that
    as another `-D` which had to match the config's `ndp_stride`, and the
    mismatch silently collapsed its work split onto a single core. It is a
    machine fact, so here it travels with the launch.
    """

    var query: UnsafePointer[Float32, MutAnyOrigin]
    var centroids: UnsafePointer[Float32, MutAnyOrigin]
    var pq_center: UnsafePointer[Float32, MutAnyOrigin]
    var codes: UnsafePointer[UInt8, MutAnyOrigin]
    var part_score: UnsafePointer[Float32, MutAnyOrigin]
    var part_id: UnsafePointer[Int32, MutAnyOrigin]
    var out_score: UnsafePointer[Float32, MutAnyOrigin]
    var out_id: UnsafePointer[Int32, MutAnyOrigin]
    var cores: Int32
    var uthreads: Int32


struct IvfPq(NDPTask):
    comptime Params = IvfPqParams

    # Scratchpad, declared once and shared by every kernel below. Contents
    # survive a launch, which is what lets the running top-K stay on chip
    # across the cluster loop. Each region's name is its identity, so the three
    # 64-byte regions (`cdist`, `top`, `topi`) stay distinct without ceremony.
    comptime cdist = scratchpad[NLIST, Float32, name="ivfpq_cdist"]()
    comptime probe = scratchpad[NPROBE, Int32, name="ivfpq_probe"]()
    comptime diff = scratchpad[VECTOR_DIM, Float32, name="ivfpq_diff"]()
    comptime lut = scratchpad[LUT_SIZE, Float32, name="ivfpq_lut"]()
    comptime score = scratchpad[LIST_SIZE, Float32, name="ivfpq_score"]()
    # Running top-K, double buffered. A stage reads one parity and writes the
    # other: with a single buffer a microthread could overwrite an entry
    # another is still folding, and a launch is the only ordering there is.
    comptime top = scratchpad[2 * TOPK, Float32, name="ivfpq_top"]()
    comptime topi = scratchpad[2 * TOPK, Int32, name="ivfpq_topi"]()
    # Which cluster step this core is on. `device_main` runs the loop, but a
    # kernel takes no arguments, so the step lives here and `advance` moves it.
    comptime step = scratchpad[1, Int32, name="ivfpq_step"]()

    # ---------------------------------------------------------------- coarse

    @staticmethod
    def coarse_dist():
        """Squared distance from the query to every coarse centroid."""
        ref p = IvfPq.params[]
        var stride = Int(p.uthreads)
        for c in range(local_uthread_id(), NLIST, stride):
            var cen = p.centroids + c * VECTOR_DIM
            var acc = SIMD[DType.float32, W](0)
            for d in range(0, VECTOR_DIM, W):
                var e = cen.load[width=W](d) - p.query.load[width=W](d)
                acc += e * e
            IvfPq.cdist[c] = acc.reduce_add()

    @staticmethod
    def coarse_sel():
        """The NPROBE nearest centroids, in rank order.

        Rank-select: count how many centroids are closer, and that count is the
        rank. Distances are distinct, so the ranks are a permutation and each
        probe slot is claimed exactly once -- no coordination between
        microthreads, and no pass per output.
        """
        ref p = IvfPq.params[]
        for c in range(local_uthread_id(), NLIST, Int(p.uthreads)):
            var x = IvfPq.cdist[c]
            var rank = 0
            for j in range(NLIST):
                if IvfPq.cdist[j] < x:
                    rank += 1
            if rank < NPROBE:
                IvfPq.probe[rank] = Int32(c)

    @staticmethod
    def start():
        """Begin the cluster loop, and seed the top-K this core folds into.

        A core with no cluster at all still publishes, and INF is what makes
        those entries lose the final merge rather than win it with a zero.
        """
        IvfPq.step[0] = 0
        for r in range(2 * TOPK):
            IvfPq.top[r] = INF
            IvfPq.topi[r] = 0

    # ----------------------------------------------------------- per cluster

    @staticmethod
    def precomp():
        """`q - c` for this core's cluster at this step.

        The codes encode the residual from the centroid, so the table below
        depends on the cluster as well as on the query, and is rebuilt for each
        cluster.
        """
        ref p = IvfPq.params[]
        var probe = group_id() + Int(IvfPq.step[0]) * Int(p.cores)
        if probe >= NPROBE:
            return
        var cen = p.centroids + Int(IvfPq.probe[probe]) * VECTOR_DIM
        for d in range(local_uthread_id(), VECTOR_DIM, Int(p.uthreads)):
            IvfPq.diff[d] = p.query[d] - cen[d]

    @staticmethod
    def build_lut():
        """`lut[m][k] = ||(q-c)_m - pq_center[m][k]||^2`.

        The dominant loop -- four fifths of the kernel at the operating points
        the reference measured -- and the one worth widening: `k` is contiguous
        in the codebook, so both the load and the store are unit stride, and
        the sum over a subvector runs inside a lane rather than across them.

        Work splits by chunk rather than by `k`. Handing microthread `u` the
        k's `{u, u+uthreads, ...}` strides every access it makes; flattening
        `(m, chunk)` into one index gives each a contiguous run instead. A
        chunk never straddles an `m`, which matters because the residual term
        is per-`m`.
        """
        ref p = IvfPq.params[]
        if group_id() + Int(IvfPq.step[0]) * Int(p.cores) >= NPROBE:
            return
        var chunks = PQ_SHIFTS // W
        for g in range(local_uthread_id(), PQ_M * chunks, Int(p.uthreads)):
            var m = g // chunks
            var k0 = (g % chunks) * W
            var pc = p.pq_center + m * SUBVECTOR_DIM * PQ_SHIFTS
            var acc = SIMD[DType.float32, W](0)
            for t in range(SUBVECTOR_DIM):
                var e = pc.load[width=W](t * PQ_SHIFTS + k0) - IvfPq.diff[
                    m * SUBVECTOR_DIM + t
                ]
                acc += e * e
            IvfPq.lut.store(m * PQ_SHIFTS + k0, acc)

    @staticmethod
    def scan():
        """Score this cluster's vectors through the table.

        Asymmetric distance: a vector's score is the sum over subquantizers of
        the table entry its code selects. The lookup is a gather however it is
        written -- each lane wants a different row -- and the simulator charges
        an issue per distinct 32-byte address, so a wide gather costs what the
        scalar loop costs. It stays scalar.

        Clusters after the first prune against the running top-K's worst entry.
        That is exact rather than approximate: TOPK vectors already score at or
        below the threshold, table entries are squared distances and so never
        negative, and a partial sum therefore only grows -- once it passes the
        threshold the vector cannot place, and the rest of its subquantizers
        are dead work. Pruned vectors score INF so the fold below never reads a
        slot from an earlier cluster.
        """
        ref p = IvfPq.params[]
        var probe = group_id() + Int(IvfPq.step[0]) * Int(p.cores)
        if probe >= NPROBE:
            return

        var first = IvfPq.step[0] == 0
        var thr = IvfPq.top[(Int(IvfPq.step[0]) & 1) * TOPK + TOPK - 1]
        var base = p.codes + Int(IvfPq.probe[probe]) * LIST_SIZE * PQ_M

        for v in range(local_uthread_id(), LIST_SIZE, Int(p.uthreads)):
            var code = base + v * PQ_M
            var acc = Float32(0)
            var pruned = False
            for m in range(PQ_M):
                acc += IvfPq.lut[m * PQ_SHIFTS + Int(code[m])]
                if not first and thr < acc:
                    pruned = True
                    break
            IvfPq.score[v] = INF if pruned else acc

    @staticmethod
    def merge():
        """Fold this cluster into the running top-K.

        Rank-select again, over this cluster's scores together with the
        incoming top-K. Every microthread takes candidates strided by the
        core's microthread count and counts how many are smaller; that count is
        where its candidate lands. Nothing coordinates, and nothing iterates
        once per output, so the cost does not grow with TOPK.

        Pruned candidates sit at exactly INF, so their rank is the number of
        finite candidates, which is always at least TOPK -- several can share
        that rank harmlessly because none of them is ever written out.

        Ids are global (`label * LIST_SIZE + i`) because the fold mixes vectors
        from different clusters.
        """
        ref p = IvfPq.params[]
        var probe = group_id() + Int(IvfPq.step[0]) * Int(p.cores)
        var pin = Int(IvfPq.step[0]) & 1
        var ti = IvfPq.top + pin * TOPK
        var tii = IvfPq.topi + pin * TOPK
        var to = IvfPq.top + (1 - pin) * TOPK
        var toi = IvfPq.topi + (1 - pin) * TOPK

        # Out of clusters: carry the chain forward, so `publish` can always
        # read the parity the last step wrote.
        if probe >= NPROBE:
            if local_uthread_id() == 0:
                for r in range(TOPK):
                    to[r] = ti[r]
                    toi[r] = tii[r]
            return

        var first = IvfPq.step[0] == 0
        var gbase = Int32(IvfPq.probe[probe]) * Int32(LIST_SIZE)
        var n = LIST_SIZE if first else LIST_SIZE + TOPK

        for a in range(local_uthread_id(), n, Int(p.uthreads)):
            var x = IvfPq.score[a] if a < LIST_SIZE else ti[a - LIST_SIZE]
            var xi = (gbase + Int32(a)) if a < LIST_SIZE else tii[a - LIST_SIZE]

            var rank = 0
            for j in range(LIST_SIZE):
                if IvfPq.score[j] < x:
                    rank += 1
            if not first:
                for j in range(TOPK):
                    if ti[j] < x:
                        rank += 1
            if rank < TOPK:
                to[rank] = x
                toi[rank] = xi

    @staticmethod
    def advance():
        """Next cluster. Its own launch, so no microthread reads a step that
        another has already moved."""
        IvfPq.step[0] += 1

    # --------------------------------------------------------------- publish

    @staticmethod
    def publish():
        """This core's top-K, into its slice of the partial results."""
        ref p = IvfPq.params[]
        if local_uthread_id() != 0:
            return
        var pin = Int(IvfPq.step[0]) & 1
        var ti = IvfPq.top + pin * TOPK
        var tii = IvfPq.topi + pin * TOPK
        var slot = group_id() * TOPK
        for r in range(TOPK):
            p.part_score[slot + r] = ti[r]
            p.part_id[slot + r] = tii[r]

    @staticmethod
    def reduce():
        """The cores' partial lists into the query's answer.

        A launch boundary orders every core against every other, so this is one
        more kernel rather than a second task. The reference needed a whole
        separate launch for it, its barrier being per core.
        """
        ref p = IvfPq.params[]
        if group_id() != 0 or local_uthread_id() != 0:
            return
        var n = Int(p.cores) * TOPK
        for a in range(n):
            var x = p.part_score[a]
            var rank = 0
            for b in range(n):
                if p.part_score[b] < x:
                    rank += 1
            if rank < TOPK:
                p.out_score[rank] = x
                p.out_id[rank] = p.part_id[a]

    @staticmethod
    def device_main():
        """The task, as the device runs it.

        Correctness rests on the launches being synchronous: a stage reads what
        the stage before it wrote, and a launch boundary is the only ordering
        there is. The cluster loop is a loop, so how many clusters a core walks
        is a constant of the workload rather than a count of hand-written
        stages.
        """
        launch_parallel[IvfPq.coarse_dist]()
        launch_parallel[IvfPq.coarse_sel]()
        launch_serial[IvfPq.start]()
        for _ in range(CLUSTERS_PER_CORE):
            launch_parallel[IvfPq.precomp]()
            launch_parallel[IvfPq.build_lut]()
            launch_parallel[IvfPq.scan]()
            launch_parallel[IvfPq.merge]()
            launch_serial[IvfPq.advance]()
        launch_serial[IvfPq.publish]()
        launch_serial[IvfPq.reduce]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh ivfpq
#
# The host builds the index, runs the search, and checks the answer against an
# exhaustive one it computes itself -- so the two do not come from the same
# place.


def main() raises:
    if IvfPq.emit_ir_if_asked():
        return

    var machine = Machine.from_config()
    var cores = Config.load().get("num_ndp_units")
    var uthreads = UTHREADS_PER_CORE
    var stride = uthreads * machine.packet

    # The one thing the workload cannot derive: how many clusters a core walks
    # depends on how many cores there are. Checked rather than assumed, because
    # too few is a cluster that is silently never searched.
    if CLUSTERS_PER_CORE * cores < NPROBE:
        raise Error(
            String("CLUSTERS_PER_CORE=")
            + String(CLUSTERS_PER_CORE)
            + " x cores="
            + String(cores)
            + " cannot cover NPROBE="
            + String(NPROBE)
        )

    var query = cxl_alloc[Float32](VECTOR_DIM)
    var centroids = cxl_alloc[Float32](NLIST * VECTOR_DIM)
    var pq_center = cxl_alloc[Float32](LUT_SIZE * SUBVECTOR_DIM)
    var codes = cxl_alloc[UInt8](NLIST * LIST_SIZE * PQ_M)
    var part_score = cxl_alloc[Float32](cores * TOPK)
    var part_id = cxl_alloc[Int32](cores * TOPK)
    var out_score = cxl_alloc[Float32](TOPK)
    var out_id = cxl_alloc[Int32](TOPK)
    # The range is a microthread count, not the data: every kernel indexes by
    # its own id rather than by the address it was mapped to. One stride per
    # core is what puts microthreads on all of them.
    var spawn = cxl_alloc[UInt8](cores * stride)

    seed(0)
    for d in range(VECTOR_DIM):
        query[d] = Float32(random_si64(-1000, 1000)) / 1000.0
    for i in range(NLIST * VECTOR_DIM):
        centroids[i] = Float32(random_si64(-1000, 1000)) / 1000.0
    for i in range(LUT_SIZE * SUBVECTOR_DIM):
        pq_center[i] = Float32(random_si64(-1000, 1000)) / 1000.0
    for i in range(NLIST * LIST_SIZE * PQ_M):
        codes[i] = UInt8(Int(random_si64(0, PQ_SHIFTS - 1)))

    # ------------------------------------------------- the answer, the long way

    var cd = List[Float32](length=NLIST, fill=0)
    for c in range(NLIST):
        var s = Float32(0)
        for d in range(VECTOR_DIM):
            var e = centroids[c * VECTOR_DIM + d] - query[d]
            s += e * e
        cd[c] = s

    var labels = List[Int]()
    for _ in range(NPROBE):
        var best = -1
        for c in range(NLIST):
            var taken = False
            for k in range(len(labels)):
                if labels[k] == c:
                    taken = True
            if not taken and (best < 0 or cd[c] < cd[best]):
                best = c
        labels.append(best)

    var want_score = List[Float32]()
    var want_id = List[Int32]()
    for li in range(len(labels)):
        var label = labels[li]
        var lut = List[Float32](length=LUT_SIZE, fill=0)
        for m in range(PQ_M):
            for k in range(PQ_SHIFTS):
                var s = Float32(0)
                for t in range(SUBVECTOR_DIM):
                    var d = m * SUBVECTOR_DIM + t
                    var e = (
                        pq_center[m * SUBVECTOR_DIM * PQ_SHIFTS + t * PQ_SHIFTS + k]
                        - (query[d] - centroids[label * VECTOR_DIM + d])
                    )
                    s += e * e
                lut[m * PQ_SHIFTS + k] = s
        for v in range(LIST_SIZE):
            var s = Float32(0)
            for m in range(PQ_M):
                var c8 = codes[(label * LIST_SIZE + v) * PQ_M + m]
                s += lut[m * PQ_SHIFTS + Int(c8)]
            want_score.append(s)
            want_id.append(Int32(label * LIST_SIZE + v))

    # The fold reads "already taken" off the ranks below a candidate, which
    # needs the scores to be distinct: a tie sends two candidates to one slot
    # and leaves the next one stale. That costs recall rather than raising an
    # error, so the index is rejected here instead of scored later.
    for a in range(len(want_score)):
        for b in range(a + 1, len(want_score)):
            if want_score[a] == want_score[b]:
                print("[host] index has tied scores; reseed")
                return

    for a in range(len(want_score)):
        for b in range(a + 1, len(want_score)):
            if want_score[b] < want_score[a]:
                var s = want_score[a]
                want_score[a] = want_score[b]
                want_score[b] = s
                var i = want_id[a]
                want_id[a] = want_id[b]
                want_id[b] = i

    var rc = IvfPq.launch(
        PooledRange.of_bytes(spawn, cores * stride),
        IvfPqParams(
            query,
            centroids,
            pq_center,
            codes,
            part_score,
            part_id,
            out_score,
            out_id,
            Int32(cores),
            Int32(uthreads),
        ),
    )
    if rc != 0:
        print("[host] ivfpq failed, exit", rc)
        return

    for r in range(TOPK):
        if out_id[r] != want_id[r]:
            print("[host] wrong at", r, ":", out_id[r], "expected", want_id[r])
            return
    print("[host] ivfpq ok")
