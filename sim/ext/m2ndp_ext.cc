// The M²NDP instructions, as a Spike extension.
//
// Spike's extension interface looked scalar-oriented -- the bundled examples
// are a RoCC accelerator and a cache flush -- but processor_t::VU is public
// and vectorUnit_t::elt<T>() gives element access, so the whole set fits here
// without patching the simulator. The submodule stays pristine.
//
// Two families, matching RISCVInstrInfoXM2ndp.td in the LLVM fork.
//
// Indexed vector atomics, 13 operations x 4 index widths = 52:
//
//     for each active lane i < vl:
//         addr  = rs1 + vs2[i]        (vs2 holds byte offsets)
//         old   = *addr
//         *addr = old OP vd[i]
//         vd[i] = old
//
// The index width is in the encoding; the data width comes from vtype, the
// same split the indexed loads and stores use. So an instruction is picked by
// (operation, index width) and the element type is decided at run time from
// vsew.
//
// Scalar floating-point atomics, 4 operations x 3 widths = 12. RISC-V has
// none of these -- the A extension is integer-only -- which is why an
// atomicrmw fadd is otherwise a cmpxchg loop.
//
// Floating-point arithmetic goes through softfloat rather than the host's
// float, so NaN propagation, signed zero and the exception flags match what
// the rest of Spike does.

// libstdc++ 11's <atomic> reaches for SYS_futex under C++20 without pulling
// in the header that declares it.
#include <sys/syscall.h>

#include "extension.h"
#include "decode_macros.h"   // f16/f32/f64, freg, NaN boxing
#include "mmu.h"
#include <type_traits>
#include <vector>

namespace {

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//
//
//   31-27 funct5 | 26 wd | 25 vm | 24-20 vs2 | 19-15 rs1 | 14-12 funct3 |
//   11-7 vd | 6-0 opcode
//
// The scalar forms reuse bits 26 and 25 for aq and rl instead of wd and vm.

constexpr insn_bits_t OPC_AMO = 0x2f;

// The integer funct5 values are the scalar AMO encodings of the same
// operation, which is what the RVV 0.10 draft did. The floating-point ones
// are ours, chosen to be free in the scalar space as well -- 0b00010 and
// 0b00011 would have been the obvious neighbours of add and swap, but there
// they are lr and sc.
enum Funct5 : unsigned {
  F5_ADD   = 0b00000,
  F5_SWAP  = 0b00001,
  F5_XOR   = 0b00100,
  F5_OR    = 0b01000,
  F5_AND   = 0b01100,
  F5_MIN   = 0b10000,
  F5_MAX   = 0b10100,
  F5_MINU  = 0b11000,
  F5_MAXU  = 0b11100,
  F5_FMIN  = 0b10001,
  F5_FADD  = 0b10010,
  F5_FSWAP = 0b10011,
  F5_FMAX  = 0b10101,
};

// The vector forms put the index element width in funct3, following the
// indexed loads and stores. The scalar forms put the access width there.
constexpr unsigned F3_EI8 = 0b000, F3_EI16 = 0b101, F3_EI32 = 0b110,
                   F3_EI64 = 0b111;
constexpr unsigned F3_H = 0b001, F3_W = 0b010, F3_D = 0b011;

// Spike tests (insn & mask) == match, so every bit set in match has to be set
// in mask too: a match bit outside the mask can never be satisfied, and the
// instruction traps as illegal with nothing pointing at why. Both are built
// from the same pieces here so they cannot drift apart.
constexpr insn_bits_t vamo_mask()
{
  return (0x1fULL << 27) | (1ULL << 26) | (0x7ULL << 12) | 0x7fULL;
}
constexpr insn_bits_t vamo_match(unsigned f5, unsigned f3)
{
  // wd = 1: these always return the previous value.
  return ((insn_bits_t)f5 << 27) | (1ULL << 26) | ((insn_bits_t)f3 << 12) |
         OPC_AMO;
}

// aq and rl stay out of the scalar mask, so one entry covers all four
// ordering variants. Nothing here reorders anything, so they need no effect.
constexpr insn_bits_t famo_mask()
{
  return (0x1fULL << 27) | (0x7ULL << 12) | 0x7fULL;
}
constexpr insn_bits_t famo_match(unsigned f5, unsigned f3)
{
  return ((insn_bits_t)f5 << 27) | ((insn_bits_t)f3 << 12) | OPC_AMO;
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

enum Op {
  OP_ADD, OP_SWAP, OP_XOR, OP_AND, OP_OR,
  OP_MIN, OP_MAX, OP_MINU, OP_MAXU,
  OP_FADD, OP_FSWAP, OP_FMIN, OP_FMAX,
};

constexpr bool is_fp_op(Op op)
{
  return op == OP_FADD || op == OP_FSWAP || op == OP_FMIN || op == OP_FMAX;
}

// Integer. The signed comparisons cast to the signed counterpart of T, which
// is the only difference between min/max and minu/maxu.
template <typename T, Op op> T apply_int(T old, T operand)
{
  using S = typename std::make_signed<T>::type;
  switch (op) {
  case OP_ADD:  return (T)(old + operand);
  case OP_SWAP: return operand;
  case OP_XOR:  return (T)(old ^ operand);
  case OP_AND:  return (T)(old & operand);
  case OP_OR:   return (T)(old | operand);
  case OP_MIN:  return (S)old < (S)operand ? old : operand;
  case OP_MAX:  return (S)old > (S)operand ? old : operand;
  case OP_MINU: return old < operand ? old : operand;
  case OP_MAXU: return old > operand ? old : operand;
  default:      return old;
  }
}

// Floating point, on raw bits. Softfloat's f*_min and f*_max are the forms
// Spike uses everywhere else, so NaN handling matches the rest of the
// simulator rather than the host's.
template <Op op> uint16_t apply_fp(uint16_t old, uint16_t operand)
{
  switch (op) {
  case OP_FADD:  return f16_add(f16(old), f16(operand)).v;
  case OP_FSWAP: return operand;
  case OP_FMIN:  return f16_min(f16(old), f16(operand)).v;
  case OP_FMAX:  return f16_max(f16(old), f16(operand)).v;
  default:       return old;
  }
}
template <Op op> uint32_t apply_fp(uint32_t old, uint32_t operand)
{
  switch (op) {
  case OP_FADD:  return f32_add(f32(old), f32(operand)).v;
  case OP_FSWAP: return operand;
  case OP_FMIN:  return f32_min(f32(old), f32(operand)).v;
  case OP_FMAX:  return f32_max(f32(old), f32(operand)).v;
  default:       return old;
  }
}
template <Op op> uint64_t apply_fp(uint64_t old, uint64_t operand)
{
  switch (op) {
  case OP_FADD:  return f64_add(f64(old), f64(operand)).v;
  case OP_FSWAP: return operand;
  case OP_FMIN:  return f64_min(f64(old), f64(operand)).v;
  case OP_FMAX:  return f64_max(f64(old), f64(operand)).v;
  default:       return old;
  }
}

// 8-bit floating point does not exist here, so the templates below can be
// instantiated for it but must never run.
template <Op op> uint8_t apply_fp(uint8_t old, uint8_t) { return old; }

template <typename T, Op op> T apply(T old, T operand)
{
  if (is_fp_op(op))
    return apply_fp<op>(old, operand);
  return apply_int<T, op>(old, operand);
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

/// Fold softfloat's accumulated flags into fflags, as every other
/// floating-point instruction in Spike does on the way out.
void raise_exceptions(processor_t *p)
{
  auto fflags = p->get_state()->fflags;
  if (softfloat_exceptionFlags)
    fflags->write(fflags->read() | softfloat_exceptionFlags);
  softfloat_exceptionFlags = 0;
}

/// One element width of an indexed vector atomic.
template <typename DataT, typename IdxT, Op op>
void vamo_lanes(processor_t *p, insn_t insn)
{
  auto &VU = p->VU;
  auto *mmu = p->get_mmu();

  const reg_t vd = insn.rd();
  const reg_t vs2 = insn.rs2();
  const reg_t base = p->get_state()->XPR[insn.rs1()];
  const bool masked = ((insn.bits() >> 25) & 1) == 0;

  const reg_t vl = VU.vl->read();
  for (reg_t i = VU.vstart->read(); i < vl; i++) {
    if (masked && !VU.mask_elt(0, i))
      continue;

    const reg_t addr = base + (reg_t)VU.elt<IdxT>(vs2, i);
    const DataT old = mmu->load<DataT>(addr);
    const DataT operand = VU.elt<DataT>(vd, i);

    mmu->store<DataT>(addr, apply<DataT, op>(old, operand));
    VU.elt<DataT>(vd, i, true) = old;
  }
  VU.vstart->write(0);
}

/// An indexed vector atomic. One per (operation, index width); the data width
/// is picked from vtype rather than from the encoding.
template <typename IdxT, Op op>
reg_t exec_vamo(processor_t *p, insn_t insn, reg_t pc)
{
  switch (p->VU.vsew) {
  case 8:
    // There is no 8-bit float, so a floating-point form at SEW=8 has no
    // meaning rather than a wrong one.
    if (is_fp_op(op))
      throw trap_illegal_instruction(insn.bits());
    vamo_lanes<uint8_t, IdxT, op>(p, insn);
    break;
  case 16: vamo_lanes<uint16_t, IdxT, op>(p, insn); break;
  case 32: vamo_lanes<uint32_t, IdxT, op>(p, insn); break;
  case 64: vamo_lanes<uint64_t, IdxT, op>(p, insn); break;
  default: throw trap_illegal_instruction(insn.bits());
  }

  if (is_fp_op(op))
    raise_exceptions(p);
  return pc + 4;
}

/// A scalar floating-point atomic. rs1 is a GPR holding the address; rs2 and
/// rd are floating-point registers, so the operand is unboxed on the way in
/// and the previous value is boxed on the way out.
template <typename UIntT, Op op>
reg_t exec_famo(processor_t *p, insn_t insn, reg_t pc)
{
  auto *state = p->get_state();
  auto *mmu = p->get_mmu();

  const reg_t addr = state->XPR[insn.rs1()];
  const freg_t rs2 = state->FPR[insn.rs2()];
  const UIntT old = mmu->load<UIntT>(addr);

  UIntT operand;
  freg_t previous;
  if (sizeof(UIntT) == 2) {
    operand = (UIntT)f16(rs2).v;
    previous = freg(f16((uint16_t)old));
  } else if (sizeof(UIntT) == 4) {
    operand = (UIntT)f32(rs2).v;
    previous = freg(f32((uint32_t)old));
  } else {
    operand = (UIntT)f64(rs2).v;
    previous = freg(f64((uint64_t)old));
  }

  mmu->store<UIntT>(addr, apply_fp<op>(old, operand));
  state->FPR.write(insn.rd(), previous);
  state->sstatus->dirty(SSTATUS_FS);

  raise_exceptions(p);
  return pc + 4;
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

insn_desc_t desc(insn_bits_t match, insn_bits_t mask, insn_func_t f)
{
  return insn_desc_t{match, mask, f, f, f, f, f, f, f, f};
}

/// The four index widths of one vector operation.
template <Op op> void add_vamo(std::vector<insn_desc_t> &v, unsigned f5)
{
  v.push_back(desc(vamo_match(f5, F3_EI8), vamo_mask(), exec_vamo<uint8_t, op>));
  v.push_back(desc(vamo_match(f5, F3_EI16), vamo_mask(), exec_vamo<uint16_t, op>));
  v.push_back(desc(vamo_match(f5, F3_EI32), vamo_mask(), exec_vamo<uint32_t, op>));
  v.push_back(desc(vamo_match(f5, F3_EI64), vamo_mask(), exec_vamo<uint64_t, op>));
}

/// The three access widths of one scalar operation.
template <Op op> void add_famo(std::vector<insn_desc_t> &v, unsigned f5)
{
  v.push_back(desc(famo_match(f5, F3_H), famo_mask(), exec_famo<uint16_t, op>));
  v.push_back(desc(famo_match(f5, F3_W), famo_mask(), exec_famo<uint32_t, op>));
  v.push_back(desc(famo_match(f5, F3_D), famo_mask(), exec_famo<uint64_t, op>));
}

class m2ndp_t : public extension_t
{
public:
  const char *name() const override { return "m2ndp"; }

  std::vector<insn_desc_t> get_instructions(const processor_t &) override
  {
    std::vector<insn_desc_t> v;

    add_vamo<OP_ADD>(v, F5_ADD);
    add_vamo<OP_SWAP>(v, F5_SWAP);
    add_vamo<OP_XOR>(v, F5_XOR);
    add_vamo<OP_AND>(v, F5_AND);
    add_vamo<OP_OR>(v, F5_OR);
    add_vamo<OP_MIN>(v, F5_MIN);
    add_vamo<OP_MAX>(v, F5_MAX);
    add_vamo<OP_MINU>(v, F5_MINU);
    add_vamo<OP_MAXU>(v, F5_MAXU);
    add_vamo<OP_FADD>(v, F5_FADD);
    add_vamo<OP_FSWAP>(v, F5_FSWAP);
    add_vamo<OP_FMIN>(v, F5_FMIN);
    add_vamo<OP_FMAX>(v, F5_FMAX);

    add_famo<OP_FADD>(v, F5_FADD);
    add_famo<OP_FSWAP>(v, F5_FSWAP);
    add_famo<OP_FMIN>(v, F5_FMIN);
    add_famo<OP_FMAX>(v, F5_FMAX);

    return v;
  }

  std::vector<disasm_insn_t *> get_disasms(const processor_t * = nullptr) override
  {
    // Spike falls back to printing the raw bits, which is enough to follow a
    // trace, and llvm-objdump is what these get read with anyway.
    return {};
  }
};

} // namespace

REGISTER_EXTENSION(m2ndp, []() { return new m2ndp_t; })
