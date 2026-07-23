// Phase 0 proof: one M²NDP vector instruction, implemented as a Spike
// extension rather than as a fork.
//
// The open question was whether Spike's extension interface can reach the
// vector unit. It can: processor_t::VU is public, and vectorUnit_t::elt<T>()
// gives element access, so an extension can do everything an indexed vector
// atomic needs without touching the simulator's own source.
//
// This implements m2ndp.vamoaddei32.v only -- enough to answer the question.
// The other 65 instructions follow the same shape and belong to Phase 2.
//
//   for each lane i < vl:
//       addr    = rs1 + vs2[i]        (vs2 holds byte offsets)
//       old     = *addr
//       *addr   = old + vd[i]
//       vd[i]   = old

// libstdc++ 11's <atomic> reaches for SYS_futex under C++20 without pulling
// in the header that declares it.
#include <sys/syscall.h>

#include "extension.h"
#include "mmu.h"
#include <vector>

namespace {

// 31-27 funct5 | 26 wd | 25 vm | 24-20 vs2 | 19-15 rs1 | 14-12 width |
// 11-7 vd | 6-0 opcode. See RISCVInstrInfoXM2ndp.td.
// match must be a subset of mask: Spike tests (insn & mask) == match, so a
// match bit outside the mask can never be satisfied.
constexpr insn_bits_t VAMOADDEI32_MASK  = 0xfc00707fULL; // funct5|wd|funct3|opcode
constexpr insn_bits_t VAMOADDEI32_MATCH = 0x0400602fULL; // vamoadd, wd=1, ei32

reg_t exec_vamoaddei32(processor_t *p, insn_t insn, reg_t pc)
{
  auto &VU = p->VU;
  auto *mmu = p->get_mmu();

  const reg_t vd  = insn.rd();
  const reg_t vs2 = insn.rs2();
  const reg_t base = p->get_state()->XPR[insn.rs1()];
  const bool masked = (insn.bits() >> 25 & 1) == 0;

  const reg_t vl = VU.vl->read();
  for (reg_t i = VU.vstart->read(); i < vl; i++) {
    if (masked && !VU.mask_elt(0, i))
      continue;

    const reg_t off = VU.elt<uint32_t>(vs2, i);
    const reg_t addr = base + off;

    const uint32_t old = mmu->load<uint32_t>(addr);
    const uint32_t add = VU.elt<uint32_t>(vd, i);
    mmu->store<uint32_t>(addr, old + add);
    VU.elt<uint32_t>(vd, i, true) = old;
  }
  VU.vstart->write(0);

  return pc + 4;
}

class m2ndp_t : public extension_t
{
public:
  const char *name() const override { return "m2ndp"; }

  std::vector<insn_desc_t> get_instructions(const processor_t &) override
  {
    return {insn_desc_t{VAMOADDEI32_MATCH, VAMOADDEI32_MASK,
                        exec_vamoaddei32, exec_vamoaddei32,
                        exec_vamoaddei32, exec_vamoaddei32,
                        exec_vamoaddei32, exec_vamoaddei32,
                        exec_vamoaddei32, exec_vamoaddei32}};
  }

  std::vector<disasm_insn_t *> get_disasms(const processor_t * = nullptr) override
  {
    return {};
  }
};

} // namespace

REGISTER_EXTENSION(m2ndp, []() { return new m2ndp_t; })
