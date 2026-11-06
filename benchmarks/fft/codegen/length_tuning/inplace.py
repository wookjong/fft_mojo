"""IP: in-place transform, the result is written over the input buffer (round 25, 2026-11-02).

The host binds `output_real/imag` to the input buffers instead of allocating them and drops the zero
fill of the output (it would clear the input). Kernels are unchanged: the pass that wrote the output
now writes the input buffer. Valid only when no pass reads the input after (or while) the output is
written; the study's Python execution screen checked this for all 18 measured configurations
(implementations/all_lengths_20261031, report_r25_ko.md).
"""

ALLOC = '    var output_{c} = cxl_alloc[Float32](total_elems)\n'
ZERO = '        output_{c}[i] = Float32(0)\n'


def apply(src):
    for c in ('real', 'imag'):
        assert src.count(ALLOC.format(c=c)) == 1 and src.count(ZERO.format(c=c)) == 1, 'unexpected host buffers'
        src = src.replace(ALLOC.format(c=c), f'    var output_{c} = input_{c}\n')
        src = src.replace(ZERO.format(c=c), '')
    return src
