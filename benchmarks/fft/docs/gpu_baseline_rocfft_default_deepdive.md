# rocFFT Default 1D C2C Planner Deep-Dive (single precision, lengths 2..~20K)

Repo: `github.com/ROCm/rocm-libraries`, path `projects/rocfft/`, branch `develop`.
All line numbers refer to the files as fetched on 2026-09-08 via
`raw.githubusercontent.com/ROCm/rocm-libraries/develop/projects/rocfft/...`.

Files fetched in full for this pass:
- `library/src/device/kernels/configs/config_sbrr.py` (509 lines, single-kernel `CS_KERNEL_STOCKHAM` configs)
- `library/src/device/kernels/configs/config_sbcc.py` (95 lines, block-column `CS_KERNEL_STOCKHAM_BLOCK_CC` configs)
- `library/src/device/kernels/configs/config_sbrc.py` (53 lines, block-row `CS_KERNEL_STOCKHAM_BLOCK_RC` configs)
- `library/src/device/kernels/configs/config_arch.py` (47 lines, LDS-size / arch enums)
- `library/src/node_factory.cpp` (1131 lines, full)
- `library/src/include/node_factory.h` (93 lines, full)
- `library/src/tree_node_1D.cpp` (1419 lines, full — `TRTRT1DNode`, `CC1DNode`, `CRT1DNode`, `Stockham1DNode`, `SBCCNode`, `SBRCNode`)
- `library/src/include/function_pool.h` (relevant sections, ~lines 355-475)
- `library/src/include/tree_node.h` (relevant sections, ~lines 860-935, default `GetKernelKey()`/`GetKernel()`)

---

## 1. Full `config_sbrr.py` table

`config_sbrr.py` defines the `sbrr_kernels` Python list consumed by the kernel generator to build the AOT-compiled `CS_KERNEL_STOCKHAM` (single-kernel, register-only Stockham) entries in `function_pool`. Each row is a `NS(length=..., workgroup_size=..., threads_per_transform=..., factors=..., [half_lds=..., direct_to_from_reg=..., runtime_compile=..., precision=[...], lds_size_bytes=...])`.

**Every length is present, contiguous from 1 through 30, then increasingly sparse** (only lengths whose radix factorization is efficiently supported get an entry — this is the definitive, non-elided answer to what the previous pass could not confirm). Full row list (length: workgroup_size / threads_per_transform / factors / notable flags):

```
1: wgs=64  tpt=1   factors=(1,)                              runtime_compile
2: wgs=64  tpt=1   factors=(2,)                              runtime_compile
3: wgs=64  tpt=1   factors=(3,)                              runtime_compile
4: wgs=128 tpt=1   factors=(4,)                              runtime_compile
5: wgs=128 tpt=1   factors=(5,)                              runtime_compile
6: wgs=128 tpt=1   factors=(6,)                              runtime_compile
7: wgs=64  tpt=1   factors=(7,)                              runtime_compile
8: wgs=64  tpt=4   factors=(4,2)                             runtime_compile
9: wgs=64  tpt=3   factors=(3,3)                             runtime_compile
10: wgs=64  tpt=1   factors=(10,)                            runtime_compile
11: wgs=128 tpt=1   factors=(11,)                            runtime_compile
12: wgs=128 tpt=6   factors=(6,2)                            runtime_compile
13: wgs=64  tpt=1   factors=(13,)                            runtime_compile
14: wgs=128 tpt=7   factors=(7,2)                            runtime_compile
15: wgs=128 tpt=5   factors=(3,5)                            runtime_compile
16: wgs=64  tpt=4   factors=(4,4)                            runtime_compile
17: wgs=256 tpt=1   factors=(17,)                            runtime_compile
18: wgs=64  tpt=6   factors=(3,6)                            runtime_compile
20: wgs=256 tpt=10  factors=(5,4)                            runtime_compile
21: wgs=128 tpt=7   factors=(3,7)                            runtime_compile
22: wgs=64  tpt=2   factors=(11,2)                           runtime_compile
24: wgs=256 tpt=8   factors=(8,3)                            runtime_compile
25: wgs=256 tpt=5   factors=(5,5)                            runtime_compile
26: wgs=64  tpt=2   factors=(13,2)                           runtime_compile
27: wgs=256 tpt=9   factors=(3,3,3)                          runtime_compile
28: wgs=64  tpt=4   factors=(7,4)                            runtime_compile
30: wgs=128 tpt=10  factors=(10,3)                           runtime_compile
32: wgs=128 tpt=16  factors=(8,4)                            (AOT, no runtime_compile)
33: wgs=256 tpt=11  factors=(11,3)                            runtime_compile
34: wgs=256 tpt=17  factors=(17,2)                            runtime_compile
35: wgs=256 tpt=7   factors=(5,7)          half_lds=False     runtime_compile
36: wgs=64  tpt=6   factors=(6,6)                            (AOT)
39: wgs=256 tpt=13  factors=(13,3)                            runtime_compile
40: wgs=128 tpt=10  factors=(10,4)                           (AOT)
42: wgs=256 tpt=7   factors=(7,6)                             (AOT)
44: wgs=64  tpt=4   factors=(11,4)                            (AOT)
45: wgs=128 tpt=15  factors=(5,3,3)                           (AOT)
48: wgs=64  tpt=16  factors=(4,3,4)                           (AOT)
49: wgs=64  tpt=7   factors=(7,7)                             (AOT)
50: wgs=256 tpt=10  factors=(10,5)                            (AOT)
51: wgs=256 tpt=17  factors=(17,3)                            runtime_compile
52: wgs=64  tpt=4   factors=(13,4)                            (AOT)
54: wgs=256 tpt=18  factors=(6,3,3)                           (AOT)
55: wgs=256 tpt=11  factors=(5,11)         half_lds=False     runtime_compile
56: wgs=128 tpt=8   factors=(7,8)                             (AOT)
60: wgs=64  tpt=10  factors=(6,10)                            (AOT)
63: wgs=256 tpt=21  factors=(3,3,7)        half_lds=False     runtime_compile
64: wgs=64  tpt=16  factors=(4,4,4)        half_lds=False, direct_to_from_reg=True
65: wgs=256 tpt=13  factors=(13,5)                            runtime_compile
66: wgs=256 tpt=11  factors=(6,11)         half_lds=False     runtime_compile
68: wgs=256 tpt=17  factors=(17,4)                            runtime_compile
70: wgs=256 tpt=14  factors=(2,5,7)                           runtime_compile
72: wgs=64  tpt=9   factors=(8,3,3)                           (AOT)
75: wgs=256 tpt=25  factors=(5,5,3)                           (AOT)
77: wgs=256 tpt=11  factors=(7,11)                            runtime_compile
78: wgs=256 tpt=13  factors=(6,13)         half_lds=False     runtime_compile
80: wgs=64  tpt=10  factors=(5,2,8)                           (AOT)
81: wgs=128 tpt=27  factors=(3,3,3,3)                          (AOT)
84: wgs=128 tpt=12  factors=(7,2,6)                            (AOT)
85: wgs=256 tpt=17  factors=(17,5)                            runtime_compile
88: wgs=128 tpt=11  factors=(11,8)                             (AOT)
90: wgs=64  tpt=9   factors=(3,3,10)                           (AOT)
91: wgs=256 tpt=13  factors=(7,13)         half_lds=False     runtime_compile
96: wgs=128 tpt=16  factors=(6,16)         half_lds=False, direct_to_from_reg=False
98: wgs=256 tpt=14  factors=(2,7,7)        half_lds=False     runtime_compile
99: wgs=256 tpt=11  factors=(3,3,11)       half_lds=False     runtime_compile
100: wgs=64  tpt=10  factors=(10,10)                          (AOT)
102: wgs=128 tpt=17  factors=(17,6)                            runtime_compile
104: wgs=64  tpt=8   factors=(13,8)                            (AOT)
105: wgs=256 tpt=21  factors=(7,3,5)        half_lds=False     runtime_compile
108: wgs=256 tpt=36  factors=(6,6,3)                           (AOT)
110: wgs=256 tpt=11  factors=(2,5,11)       half_lds=False     runtime_compile
112: wgs=256 tpt=16  factors=(16,7)         half_lds=False, direct_to_from_reg=False
117: wgs=64  tpt=13  factors=(13,9)                            runtime_compile
119: wgs=256 tpt=17  factors=(17,7)                            runtime_compile
120: wgs=64  tpt=12  factors=(6,10,2)                           runtime_compile
121: wgs=128 tpt=11  factors=(11,11)                            runtime_compile
125: wgs=256 tpt=25  factors=(5,5,5)        half_lds=False, direct_to_from_reg=False
126: wgs=256 tpt=42  factors=(6,7,3)        half_lds=False     runtime_compile
128: wgs=256 tpt=16  factors=(16,8)                            (AOT)
130: wgs=64  tpt=13  factors=(13,10)        half_lds=False, direct_to_from_reg=False, runtime_compile
132: wgs=128 tpt=22  factors=(11,6,2)       half_lds=False, direct_to_from_reg=False, runtime_compile
135: wgs=128 tpt=9   factors=(5,3,3,3)                          runtime_compile
136: wgs=128 tpt=17  factors=(17,8)                             runtime_compile
140: wgs=64  tpt=28  factors=(7,5,4)        half_lds=False, direct_to_from_reg=False, runtime_compile
143: wgs=256 tpt=13  factors=(13,11)        half_lds=False     runtime_compile
144: wgs=128 tpt=12  factors=(6,6,4)                            (AOT)
147: wgs=64  tpt=21  factors=(7,7,3)        half_lds=False, direct_to_from_reg=False, runtime_compile
150: wgs=64  tpt=5   factors=(10,5,3)                            runtime_compile
153: wgs=128 tpt=17  factors=(17,9)                             runtime_compile
154: wgs=128 tpt=22  factors=(11,7,2)       half_lds=False, direct_to_from_reg=False, runtime_compile
156: wgs=128 tpt=13  factors=(3,4,13)       half_lds=False     runtime_compile
160: wgs=256 tpt=16  factors=(16,10)                            (AOT)
162: wgs=256 tpt=27  factors=(6,3,3,3)                            runtime_compile
165: wgs=64  tpt=11  factors=(11,5,3)       half_lds=False, direct_to_from_reg=False, runtime_compile
168: wgs=256 tpt=56  factors=(8,7,3)        half_lds=False, direct_to_from_reg=False
169: wgs=256 tpt=13  factors=(13,13)                            runtime_compile
170: wgs=128 tpt=17  factors=(17,10)                             runtime_compile
175: wgs=256 tpt=35  factors=(5,5,7)        half_lds=False      runtime_compile
176: wgs=64  tpt=16  factors=(11,16)                             runtime_compile
180: wgs=256 tpt=60  factors=(10,6,3)       half_lds=False, direct_to_from_reg=False
182: wgs=64  tpt=13  factors=(13,2,7)       half_lds=False     runtime_compile
187: wgs=128 tpt=17  factors=(17,11)                             runtime_compile
189: wgs=64  tpt=21  factors=(7,3,3,3)      half_lds=False, direct_to_from_reg=False, runtime_compile
192: wgs=128 tpt=16  factors=(6,4,4,2)                            (AOT)
195: wgs=64  tpt=13  factors=(13,5,3)       half_lds=False, direct_to_from_reg=False, runtime_compile
196: wgs=64  tpt=28  factors=(4,7,7)        half_lds=False, direct_to_from_reg=False, runtime_compile
198: wgs=128 tpt=22  factors=(11,2,9)       half_lds=False     runtime_compile
200: wgs=64  tpt=20  factors=(10,10,2)                            (AOT)
204: wgs=128 tpt=17  factors=(17,4,3)                             runtime_compile
208: wgs=64  tpt=16  factors=(13,16)                              (AOT)
210: wgs=64  tpt=30  factors=(10,7,3)       half_lds=False, direct_to_from_reg=False, runtime_compile
216: wgs=256 tpt=36  factors=(6,6,6)                              (AOT)
220: wgs=128 tpt=22  factors=(10,2,11)      half_lds=False     runtime_compile
221: wgs=128 tpt=17  factors=(17,13)                              runtime_compile
224: wgs=64  tpt=16  factors=(7,2,2,2,2,2)                        (AOT)
225: wgs=256 tpt=75  factors=(5,5,3,3)                             runtime_compile
231: wgs=256 tpt=33  factors=(11,7,3)       half_lds=False, direct_to_from_reg=False, runtime_compile
234: wgs=64  tpt=26  factors=(13,9,2)       half_lds=False, direct_to_from_reg=False, runtime_compile
238: wgs=64  tpt=17  factors=(17,7,2)                             runtime_compile
240: wgs=128 tpt=48  factors=(8,5,6)                              (AOT)
242: wgs=128 tpt=22  factors=(11,2,11)      half_lds=False     runtime_compile
243: wgs=256 tpt=81  factors=(3,3,3,3,3)                          (AOT)
245: wgs=256 tpt=35  factors=(7,5,7)        half_lds=False      runtime_compile
250: wgs=128 tpt=25  factors=(10,5,5)                             runtime_compile
252: wgs=64  tpt=63  factors=(7,3,3,4)      half_lds=False, direct_to_from_reg=False, runtime_compile
255: wgs=64  tpt=17  factors=(17,5,3)                             runtime_compile
256: wgs=64  tpt=64  factors=(4,4,4,4)                            (AOT)
260: wgs=64  tpt=26  factors=(13,10,2)      half_lds=False      runtime_compile
264: wgs=256 tpt=33  factors=(8,3,11)       half_lds=False      runtime_compile
270: wgs=128 tpt=27  factors=(10,3,3,3)                           (AOT)
272: wgs=128 tpt=17  factors=(16,17)                              runtime_compile
273: wgs=64  tpt=13  factors=(13,3,7)       half_lds=False, direct_to_from_reg=False, runtime_compile
275: wgs=64  tpt=55  factors=(11,5,5)       half_lds=False      runtime_compile
280: wgs=64  tpt=56  factors=(8,7,5)        half_lds=False, direct_to_from_reg=False, runtime_compile
286: wgs=64  tpt=26  factors=(13,11,2)      half_lds=False, direct_to_from_reg=False, runtime_compile
288: wgs=128 tpt=24  factors=(6,6,4,2)                            runtime_compile
289: wgs=128 tpt=17  factors=(17,17)                              runtime_compile
294: wgs=128 tpt=42  factors=(6,7,7)        half_lds=False, direct_to_from_reg=False, runtime_compile
297: wgs=256 tpt=33  factors=(9,3,11)                             runtime_compile
300: wgs=64  tpt=30  factors=(10,10,3)                            runtime_compile
306: wgs=256 tpt=34  factors=(17,2,9)                             runtime_compile
308: wgs=64  tpt=44  factors=(11,7,4)       half_lds=False, direct_to_from_reg=False, runtime_compile
312: wgs=64  tpt=26  factors=(13,4,3,2)      half_lds=False     runtime_compile
315: wgs=64  tpt=63  factors=(7,3,3,5)      half_lds=False, direct_to_from_reg=False, runtime_compile
320: wgs=64  tpt=16  factors=(10,4,4,2)                           runtime_compile
324: wgs=64  tpt=54  factors=(3,6,6,3)                            runtime_compile
325: wgs=64  tpt=13  factors=(13,5,5)       half_lds=False, direct_to_from_reg=False, runtime_compile
330: wgs=128 tpt=33  factors=(11,10,3)      half_lds=False, direct_to_from_reg=False, runtime_compile
336: wgs=128 tpt=56  factors=(8,7,6)                              (AOT)
338: wgs=64  tpt=26  factors=(13,2,13)                            runtime_compile
340: wgs=128 tpt=34  factors=(17,2,10)                            runtime_compile
343: wgs=256 tpt=49  factors=(7,7,7)                              runtime_compile
350: wgs=64  tpt=50  factors=(5,7,10)       half_lds=False, direct_to_from_reg=False, runtime_compile
351: wgs=128 tpt=39  factors=(13,3,9)       half_lds=False      runtime_compile
352: wgs=64  tpt=32  factors=(11,2,16)      half_lds=False, direct_to_from_reg=False, runtime_compile
357: wgs=256 tpt=17  factors=(17,3,7)                             runtime_compile
360: wgs=256 tpt=60  factors=(10,6,6)                             runtime_compile
363: wgs=128 tpt=33  factors=(11,3,11)                            runtime_compile
364: wgs=64  tpt=52  factors=(13,7,4)       half_lds=False, direct_to_from_reg=False, runtime_compile
374: wgs=256 tpt=34  factors=(17,2,11)                            runtime_compile
375: wgs=128 tpt=25  factors=(5,5,5,3)                            runtime_compile
378: wgs=128 tpt=126 factors=(6,3,3,7)      half_lds=False      runtime_compile
384: wgs=128 tpt=32  factors=(6,4,4,4)                            runtime_compile
385: wgs=64  tpt=55  factors=(11,7,5)       half_lds=False, direct_to_from_reg=False, runtime_compile
390: wgs=128 tpt=39  factors=(13,3,10)      half_lds=False      runtime_compile
392: wgs=64  tpt=56  factors=(8,7,7)        half_lds=False, direct_to_from_reg=False, runtime_compile
396: wgs=64  tpt=44  factors=(11,9,4)       half_lds=False, direct_to_from_reg=False, runtime_compile
400: wgs=128 tpt=40  factors=(4,10,10)                            runtime_compile
405: wgs=128 tpt=27  factors=(5,3,3,3,3)                          runtime_compile
408: wgs=64  tpt=17  factors=(17,3,8)                             runtime_compile
416: wgs=64  tpt=32  factors=(13,2,16)      half_lds=False      runtime_compile
420: wgs=64  tpt=60  factors=(10,7,6)       half_lds=False, direct_to_from_reg=False, runtime_compile
425: wgs=64  tpt=17  factors=(17,5,5)                             runtime_compile
429: wgs=128 tpt=39  factors=(13,3,11)      half_lds=False      runtime_compile
432: wgs=64  tpt=27  factors=(3,16,3,3)                           runtime_compile
440: wgs=64  tpt=55  factors=(11,8,5)       half_lds=False, direct_to_from_reg=False, runtime_compile
441: wgs=64  tpt=63  factors=(9,7,7)        half_lds=False, direct_to_from_reg=False, runtime_compile
442: wgs=256 tpt=34  factors=(17,2,13)                            runtime_compile
448: wgs=128 tpt=64  factors=(8,7,8)        half_lds=False, direct_to_from_reg=False, runtime_compile
450: wgs=128 tpt=30  factors=(10,5,3,3)                           runtime_compile
455: wgs=256 tpt=65  factors=(13,5,7)       half_lds=False      runtime_compile
459: wgs=256 tpt=51  factors=(17,3,9)                             runtime_compile
462: wgs=256 tpt=77  factors=(11,6,7)       half_lds=False, direct_to_from_reg=False, runtime_compile
468: wgs=64  tpt=52  factors=(13,9,4)       half_lds=False, direct_to_from_reg=False, runtime_compile
476: wgs=128 tpt=34  factors=(17,2,7,2)                           runtime_compile
480: wgs=64  tpt=16  factors=(10,8,6)                             runtime_compile
484: wgs=64  tpt=44  factors=(4,11,11)      half_lds=False, direct_to_from_reg=False, runtime_compile
486: wgs=256 tpt=162 factors=(6,3,3,3,3)                          runtime_compile
490: wgs=256 tpt=70  factors=(10,7,7)       half_lds=False      runtime_compile
495: wgs=64  tpt=55  factors=(11,9,5)       half_lds=False, direct_to_from_reg=False, runtime_compile
500: wgs=128 tpt=100 factors=(10,5,10)                            runtime_compile
504: wgs=64  tpt=63  factors=(7,9,4,2)      half_lds=False, direct_to_from_reg=False, runtime_compile
507: wgs=128 tpt=39  factors=(13,3,13)                            runtime_compile
510: wgs=256 tpt=34  factors=(17,2,3,5)                           runtime_compile
512: wgs=64  tpt=64  factors=(8,8,8)                              (AOT)
520: wgs=64  tpt=52  factors=(13,10,4)      half_lds=False, direct_to_from_reg=False, runtime_compile
525: wgs=128 tpt=105 factors=(7,3,5,5)      half_lds=False, direct_to_from_reg=False, runtime_compile
528: wgs=64  tpt=48  factors=(4,4,3,11)                           runtime_compile
539: wgs=256 tpt=77  factors=(11,7,7)                             runtime_compile
540: wgs=256 tpt=54  factors=(3,10,6,3)                           runtime_compile
544: wgs=128 tpt=34  factors=(17,2,16)                            runtime_compile
546: wgs=128 tpt=39  factors=(13,3,7,2)                           runtime_compile
550: wgs=64  tpt=55  factors=(11,10,5)      half_lds=False, direct_to_from_reg=False, runtime_compile
560: wgs=64  tpt=56  factors=(8,7,5,2)      half_lds=False, direct_to_from_reg=False, runtime_compile
561: wgs=256 tpt=51  factors=(17,3,11)                            runtime_compile
567: wgs=64  tpt=63  factors=(7,9,3,3)      half_lds=False, direct_to_from_reg=False, runtime_compile
572: wgs=64  tpt=52  factors=(13,11,4)      half_lds=False, direct_to_from_reg=False, runtime_compile
576: wgs=128 tpt=96  factors=(16,6,6)                             runtime_compile
578: wgs=256 tpt=34  factors=(17,17,2)                             runtime_compile
585: wgs=256 tpt=65  factors=(13,5,9)       half_lds=False      runtime_compile
588: wgs=256 tpt=84  factors=(7,3,4,7)      half_lds=False, direct_to_from_reg=False, runtime_compile
594: wgs=128 tpt=99  factors=(11,3,6,3)     half_lds=False      runtime_compile
595: wgs=64  tpt=17  factors=(7,17,5)                             runtime_compile
600: wgs=64  tpt=60  factors=(10,6,10)                            runtime_compile
605: wgs=64  tpt=55  factors=(11,5,11)      half_lds=False      runtime_compile
612: wgs=64  tpt=51  factors=(17,3,6,2)                           runtime_compile
616: wgs=128 tpt=88  factors=(11,7,8)       half_lds=False      runtime_compile
624: wgs=64  tpt=52  factors=(13,4,6,2)     half_lds=False      runtime_compile
625: wgs=128 tpt=125 factors=(5,5,5,5)                            runtime_compile
630: wgs=64  tpt=63  factors=(3,3,5,7,2)                          runtime_compile
637: wgs=128 tpt=91  factors=(13,7,7)                             runtime_compile
640: wgs=128 tpt=64  factors=(8,10,8)                             runtime_compile
648: wgs=256 tpt=216 factors=(8,3,3,3,3)                          runtime_compile
650: wgs=256 tpt=65  factors=(10,5,13)      half_lds=False      runtime_compile
660: wgs=128 tpt=110 factors=(11,6,10)                            runtime_compile
663: wgs=64  tpt=51  factors=(17,13,3)      half_lds=False      runtime_compile
672: wgs=64  tpt=56  factors=(2,2,2,2,2,3,7)                      runtime_compile
675: wgs=256 tpt=225 factors=(5,5,3,3,3)                          runtime_compile
676: wgs=64  tpt=52  factors=(13,13,4)      half_lds=False      runtime_compile
680: wgs=256 tpt=68  factors=(17,4,10)                            runtime_compile
686: wgs=64  tpt=49  factors=(7,7,7,2)      half_lds=False, direct_to_from_reg=False, runtime_compile
693: wgs=128 tpt=99  factors=(11,7,9)                             runtime_compile
700: wgs=128 tpt=100 factors=(10,7,10)      half_lds=False, direct_to_from_reg=False, runtime_compile
702: wgs=128 tpt=117 factors=(13,3,6,3)                           runtime_compile
704: wgs=256 tpt=88  factors=(2,2,2,2,11,2,2)                     runtime_compile
714: wgs=64  tpt=51  factors=(3,17,7,2)                           runtime_compile
715: wgs=256 tpt=65  factors=(13,5,11)                            runtime_compile
720: wgs=256 tpt=120 factors=(10,3,8,3)                           runtime_compile
726: wgs=256 tpt=66  factors=(11,6,11)      half_lds=False      runtime_compile
728: wgs=128 tpt=104 factors=(13,7,8)                             runtime_compile
729: wgs=256 tpt=243 factors=(3,3,3,3,3,3)                        runtime_compile
735: wgs=256 tpt=147 factors=(7,3,5,7)      half_lds=False      runtime_compile
748: wgs=256 tpt=68  factors=(17,4,11)                            runtime_compile
750: wgs=256 tpt=250 factors=(10,5,3,5)                           runtime_compile
756: wgs=64  tpt=63  factors=(2,2,3,3,3,7) half_lds=False, direct_to_from_reg=False, runtime_compile
765: wgs=256 tpt=51  factors=(17,3,5,3)                           runtime_compile
768: wgs=64  tpt=48  factors=(16,3,16)                            runtime_compile
770: wgs=256 tpt=110 factors=(11,10,7)      half_lds=False      runtime_compile
780: wgs=256 tpt=78  factors=(2,3,13,5,2)                         runtime_compile
784: wgs=64  tpt=56  factors=(2,2,2,2,7,7)                        runtime_compile
792: wgs=256 tpt=88  factors=(2,2,2,3,3,11) half_lds=False      runtime_compile
800: wgs=256 tpt=160 factors=(16,5,10)                            runtime_compile
810: wgs=128 tpt=81  factors=(3,10,3,3,3)                         runtime_compile
816: wgs=64  tpt=51  factors=(17,2,3,2,2,2)                       runtime_compile
819: wgs=128 tpt=117 factors=(9,7,13)       half_lds=False      runtime_compile
825: wgs=64  tpt=55  factors=(11,5,5,3)     half_lds=False, direct_to_from_reg=False, runtime_compile
832: wgs=128 tpt=104 factors=(13,2,2,2,2,2,2)                     runtime_compile
833: wgs=128 tpt=119 factors=(17,7,7)                             runtime_compile
840: wgs=64  tpt=56  factors=(2,2,2,3,5,7)                        runtime_compile
845: wgs=256 tpt=65  factors=(13,5,13)                            runtime_compile
847: wgs=256 tpt=77  factors=(11,7,11)                            runtime_compile
850: wgs=128 tpt=85  factors=(10,5,17)      half_lds=False      runtime_compile
858: wgs=256 tpt=78  factors=(13,11,6)                            runtime_compile
864: wgs=64  tpt=54  factors=(3,6,16,3)                           runtime_compile
867: wgs=64  tpt=51  factors=(17,17,3)                            runtime_compile
875: wgs=256 tpt=175 factors=(7,5,5,5)      half_lds=False      runtime_compile
880: wgs=256 tpt=88  factors=(2,2,2,2,11,5)                       runtime_compile
882: wgs=64  tpt=63  factors=(9,7,7,2)      half_lds=False, direct_to_from_reg=False, runtime_compile
884: wgs=256 tpt=68  factors=(13,4,17)                            runtime_compile
891: wgs=256 tpt=99  factors=(9,11,3,3)                            runtime_compile
896: wgs=128 tpt=112 factors=(2,2,2,2,2,2,2,7) half_lds=False, direct_to_from_reg=False, runtime_compile
900: wgs=256 tpt=90  factors=(10,10,3,3)                          runtime_compile
910: wgs=256 tpt=91  factors=(13,2,7,5)     half_lds=False      runtime_compile
918: wgs=128 tpt=102 factors=(17,9,2,3)                            runtime_compile
924: wgs=64  tpt=44  factors=(2,2,3,7,11)                          runtime_compile
935: wgs=256 tpt=85  factors=(17,11,5)                             runtime_compile
936: wgs=256 tpt=78  factors=(2,2,13,2,3,3)                        runtime_compile
945: wgs=64  tpt=63  factors=(3,3,3,5,7)                          runtime_compile
952: wgs=256 tpt=68  factors=(17,4,2,7)                            runtime_compile
960: wgs=256 tpt=160 factors=(16,10,6)      half_lds=False, direct_to_from_reg=False, runtime_compile
968: wgs=256 tpt=88  factors=(2,2,2,11,11)  half_lds=False      runtime_compile
972: wgs=256 tpt=162 factors=(3,6,3,6,3)                           runtime_compile
975: wgs=128 tpt=39  factors=(13,5,3,5)                            runtime_compile
980: wgs=256 tpt=196 factors=(7,5,7,4)      half_lds=False, direct_to_from_reg=False, runtime_compile
990: wgs=128 tpt=110 factors=(2,3,3,5,11)   half_lds=False      runtime_compile
1000: wgs=128 tpt=100 factors=(10,10,10)                           runtime_compile
1001: wgs=256 tpt=91  factors=(13,7,11)                            runtime_compile
1008: wgs=64  tpt=56  factors=(2,2,2,2,3,3,7)                      runtime_compile
1014: wgs=256 tpt=78  factors=(13,6,13)     half_lds=False      runtime_compile
1020: wgs=256 tpt=68  factors=(2,17,2,3,5)                         runtime_compile
1024: wgs=128 tpt=128 factors=(8,8,4,4)                            (AOT)
1040: wgs=256 tpt=208 factors=(13,16,5)                            runtime_compile
1050: wgs=256 tpt=210 factors=(2,3,5,5,7)   half_lds=False      runtime_compile
1053: wgs=128 tpt=117 factors=(3,3,13,3,3)                        runtime_compile
1056: wgs=256 tpt=176 factors=(2,2,2,2,11,6)                       runtime_compile
1071: wgs=128 tpt=119 factors=(17,7,9)                             runtime_compile
1078: wgs=256 tpt=77  factors=(2,11,7,7)                           runtime_compile
1080: wgs=256 tpt=108 factors=(6,10,6,3)                          runtime_compile
1088: wgs=256 tpt=68  factors=(17,4,4,2,2)                         runtime_compile
1089: wgs=128 tpt=121 factors=(3,11,3,11)   half_lds=False      runtime_compile
1092: wgs=64  tpt=52  factors=(2,2,13,7,3)                         runtime_compile
1100: wgs=128 tpt=110 factors=(2,2,11,5,5) half_lds=False      runtime_compile
1105: wgs=256 tpt=85  factors=(17,13,5)                            runtime_compile
1120: wgs=256 tpt=224 factors=(2,2,2,2,2,5,7)                      runtime_compile
1122: wgs=256 tpt=102 factors=(17,11,6)                            runtime_compile
1125: wgs=256 tpt=225 factors=(5,5,3,3,5)                          runtime_compile
1134: wgs=128 tpt=126 factors=(2,3,3,3,3,7) half_lds=False, direct_to_from_reg=False, runtime_compile
1144: wgs=128 tpt=104 factors=(13,11,8)     half_lds=False, direct_to_from_reg=False, runtime_compile
1152: wgs=256 tpt=144 factors=(4,3,8,3,4)                          runtime_compile
1155: wgs=64  tpt=55  factors=(11,5,7,3)                           runtime_compile
1156: wgs=256 tpt=68  factors=(17,2,17,2)                          runtime_compile
1170: wgs=256 tpt=117 factors=(2,13,3,5,3) half_lds=False      runtime_compile
1176: wgs=64  tpt=56  factors=(2,2,2,3,7,7)                        runtime_compile
1183: wgs=256 tpt=91  factors=(7,13,13)                            runtime_compile
1188: wgs=256 tpt=66  factors=(6,11,2,3,3)                        runtime_compile
1190: wgs=256 tpt=85  factors=(17,2,5,7)                          runtime_compile
1200: wgs=256 tpt=75  factors=(5,5,16,3)                          runtime_compile
1210: wgs=128 tpt=110 factors=(2,5,11,11)                         runtime_compile
1215: wgs=256 tpt=243 factors=(5,3,3,3,3,3)                        runtime_compile
1224: wgs=256 tpt=102 factors=(17,3,4,6)                          runtime_compile
1225: wgs=256 tpt=175 factors=(5,5,7,7)                            runtime_compile
1232: wgs=256 tpt=176 factors=(2,2,2,2,11,7)                       runtime_compile
1248: wgs=64  tpt=52  factors=(2,2,13,2,3,2,2)                     runtime_compile
1250: wgs=256 tpt=250 factors=(5,10,5,5)                           runtime_compile
1260: wgs=64  tpt=63  factors=(2,2,3,3,5,7)                        runtime_compile
1274: wgs=256 tpt=182 factors=(2,13,7,7)                          runtime_compile
1275: wgs=256 tpt=85  factors=(17,3,5,5)                          runtime_compile
1280: wgs=128 tpt=80  factors=(16,5,16)                           runtime_compile
1287: wgs=128 tpt=117 factors=(3,13,3,11)   half_lds=False      runtime_compile
1296: wgs=128 tpt=108 factors=(6,6,6,6)                            runtime_compile
1300: wgs=256 tpt=130 factors=(10,10,13)    half_lds=False      runtime_compile
1309: wgs=128 tpt=119 factors=(17,7,11)                           runtime_compile
1320: wgs=256 tpt=165 factors=(11,2,3,5,4)  half_lds=False      runtime_compile
1323: wgs=256 tpt=189 factors=(3,3,3,7,7)   half_lds=False      runtime_compile
1326: wgs=256 tpt=102 factors=(17,6,13)                           runtime_compile
1331: wgs=256 tpt=121 factors=(11,11,11)                           runtime_compile
1344: wgs=256 tpt=224 factors=(2,2,2,2,2,2,3,7)                    runtime_compile
1350: wgs=256 tpt=135 factors=(5,10,3,3,3)                         runtime_compile
1352: wgs=64  tpt=52  factors=(2,13,13,4)                          runtime_compile
1360: wgs=256 tpt=85  factors=(17,5,16)                            runtime_compile
1365: wgs=256 tpt=91  factors=(13,7,5,3)                           runtime_compile
1372: wgs=256 tpt=98  factors=(2,2,7,7,7)                          runtime_compile
1375: wgs=64  tpt=55  factors=(11,5,5,5)                           runtime_compile
1377: wgs=64  tpt=51  factors=(17,3,9,3)                           runtime_compile
1386: wgs=256 tpt=231 factors=(2,7,3,11,3)                         runtime_compile
1400: wgs=64  tpt=56  factors=(2,2,2,5,7,5)                        runtime_compile
1404: wgs=128 tpt=117 factors=(2,2,3,13,3,3)                       runtime_compile
1408: wgs=256 tpt=176 factors=(2,2,2,2,2,2,11,2)                   runtime_compile
1428: wgs=128 tpt=119 factors=(17,2,7,6)                           runtime_compile
1430: wgs=256 tpt=143 factors=(13,11,10)    half_lds=False      runtime_compile
1440: wgs=128 tpt=90  factors=(10,16,3,3)                          runtime_compile
1445: wgs=128 tpt=85  factors=(17,5,17)                            runtime_compile
1452: wgs=256 tpt=132 factors=(11,3,11,4)                          runtime_compile
1456: wgs=256 tpt=182 factors=(13,4,7,2,2)                         runtime_compile
1458: wgs=256 tpt=243 factors=(6,3,3,3,3,3)                        runtime_compile
1470: wgs=256 tpt=210 factors=(2,3,5,7,7)                          runtime_compile
1485: wgs=256 tpt=165 factors=(3,5,11,3,3)  half_lds=False      runtime_compile
1496: wgs=256 tpt=187 factors=(17,8,11)                           runtime_compile
1500: wgs=256 tpt=150 factors=(5,10,10,3)                         runtime_compile
1512: wgs=64  tpt=63  factors=(2,2,2,3,3,3,7)                      runtime_compile
1521: wgs=128 tpt=117 factors=(13,3,3,13)                          runtime_compile
1530: wgs=128 tpt=102 factors=(17,3,6,5)                          runtime_compile
1536: wgs=256 tpt=256 factors=(16,16,6)                           runtime_compile
1540: wgs=256 tpt=154 factors=(11,2,7,5,2)                         runtime_compile
1547: wgs=128 tpt=119 factors=(17,7,13)                            runtime_compile
1560: wgs=256 tpt=156 factors=(13,2,2,10,3) half_lds=False      runtime_compile
1568: wgs=256 tpt=224 factors=(2,2,2,2,2,7,7)                      runtime_compile
1573: wgs=256 tpt=143 factors=(13,11,11)    half_lds=False      runtime_compile
1575: wgs=64  tpt=63  factors=(3,3,5,7,5)                          runtime_compile
1584: wgs=256 tpt=176 factors=(4,2,2,11,3,3)                       runtime_compile
1600: wgs=256 tpt=100 factors=(10,16,10)                          runtime_compile
1617: wgs=256 tpt=231 factors=(3,7,7,11)    half_lds=False      runtime_compile
1620: wgs=256 tpt=162 factors=(10,3,3,6,3)                        runtime_compile
1625: wgs=256 tpt=65  factors=(13,5,5,5)                           runtime_compile
1632: wgs=128 tpt=102 factors=(17,2,2,3,8)                        runtime_compile
1638: wgs=256 tpt=182 factors=(13,2,3,7,3)                        runtime_compile
1650: wgs=128 tpt=110 factors=(11,2,3,5,5)                        runtime_compile
1664: wgs=256 tpt=208 factors=(13,2,2,4,2,2,2)                     runtime_compile
1666: wgs=128 tpt=119 factors=(17,2,7,7)                          runtime_compile
1680: wgs=128 tpt=112 factors=(2,2,2,2,3,7,5)                      runtime_compile
1683: wgs=64  tpt=51  factors=(17,3,11,3)                          runtime_compile
1690: wgs=256 tpt=169 factors=(13,10,13)    half_lds=False      runtime_compile
1694: wgs=256 tpt=154 factors=(11,2,11,7)                          runtime_compile
1700: wgs=256 tpt=170 factors=(17,10,10)                          runtime_compile
1701: wgs=64  tpt=63  factors=(3,3,3,3,3,7)                        runtime_compile
1715: wgs=256 tpt=245 factors=(5,7,7,7)                            runtime_compile
1716: wgs=256 tpt=156 factors=(13,2,6,11)   half_lds=False      runtime_compile
1728: wgs=128 tpt=108 factors=(3,6,6,16)                          runtime_compile
1734: wgs=128 tpt=102 factors=(17,17,6)                           runtime_compile
1750: wgs=256 tpt=175 factors=(2,5,5,7,5)                          runtime_compile
1755: wgs=128 tpt=117 factors=(13,3,3,3,5)                        runtime_compile
1760: wgs=256 tpt=176 factors=(2,2,2,2,2,11,5)                     runtime_compile
1764: wgs=128 tpt=126 factors=(2,2,3,3,7,7)                        runtime_compile
1768: wgs=256 tpt=136 factors=(17,13,8)                            runtime_compile
1782: wgs=128 tpt=99  factors=(11,3,3,3,3,2)                       runtime_compile
1785: wgs=128 tpt=119 factors=(17,3,5,7)                          runtime_compile
1792: wgs=256 tpt=224 factors=(4,4,4,4,7)                          runtime_compile
1800: wgs=256 tpt=180 factors=(10,6,10,3)                          runtime_compile
1815: wgs=256 tpt=165 factors=(11,3,5,11)   half_lds=False      runtime_compile
1820: wgs=256 tpt=182 factors=(10,13,7,2)                          runtime_compile
1836: wgs=256 tpt=153 factors=(17,3,3,2,6)                        runtime_compile
1848: wgs=256 tpt=231 factors=(3,11,7,4,2)                         runtime_compile
1859: wgs=256 tpt=169 factors=(13,11,13)                          runtime_compile
1870: wgs=256 tpt=187 factors=(17,10,11)                          runtime_compile
1872: wgs=256 tpt=156 factors=(13,3,4,6,2)                         runtime_compile
1875: wgs=256 tpt=125 factors=(5,5,5,5,3)                          runtime_compile
1890: wgs=128 tpt=126 factors=(2,3,3,3,7,5)                        runtime_compile
1904: wgs=128 tpt=119 factors=(17,2,2,7,4)                        runtime_compile
1911: wgs=128 tpt=91  factors=(13,7,7,3)                          runtime_compile
1920: wgs=256 tpt=120 factors=(10,6,16,2)                          runtime_compile
1925: wgs=64  tpt=55  factors=(7,11,5,5)                           runtime_compile
1936: wgs=256 tpt=176 factors=(2,2,4,11,11) half_lds=False      runtime_compile
1944: wgs=256 tpt=243 factors=(3,3,3,3,8,3)                        runtime_compile
1950: wgs=256 tpt=195 factors=(13,5,10,3)   half_lds=False      runtime_compile
1960: wgs=64  tpt=56  factors=(4,7,2,7,5)                          runtime_compile
1980: wgs=256 tpt=198 factors=(11,2,3,3,5,2)                       runtime_compile
1989: wgs=256 tpt=153 factors=(17,13,9)                            runtime_compile
2000: wgs=128 tpt=125 factors=(5,5,5,16)                           runtime_compile
2002: wgs=256 tpt=182 factors=(2,13,7,11)                          runtime_compile
2016: wgs=256 tpt=112 factors=(2,2,2,2,2,3,3,7)                    runtime_compile
2023: wgs=128 tpt=119 factors=(17,7,17)                            runtime_compile
2025: wgs=256 tpt=135 factors=(3,3,5,5,3,3)                        runtime_compile
2028: wgs=256 tpt=156 factors=(13,4,3,13)   half_lds=False      runtime_compile
2040: wgs=256 tpt=170 factors=(17,4,3,10)                          runtime_compile
2048: wgs=256 tpt=256 factors=(16,16,8)                            runtime_compile
2160: wgs=256 tpt=60  factors=(10,6,6,6)                           runtime_compile
2187: wgs=256 tpt=243 factors=(3,3,3,3,3,3,3)                      runtime_compile
2197: wgs=256 tpt=169 factors=(13,13,13)                           runtime_compile
2250: wgs=256 tpt=90  factors=(10,3,5,3,5)                        runtime_compile
2304: wgs=256 tpt=192 factors=(6,6,4,4,4)                          runtime_compile
2400: wgs=256 tpt=240 factors=(4,10,10,6)                          runtime_compile
2401: wgs=256 tpt=49  factors=(7,7,7,7)                            runtime_compile
2430: wgs=256 tpt=81  factors=(10,3,3,3,3,3)                       runtime_compile
2500: wgs=256 tpt=250 factors=(10,5,10,5)                          runtime_compile
2560: wgs=128 tpt=128 factors=(4,4,4,10,4)                        runtime_compile
2592: wgs=256 tpt=216 factors=(6,6,6,6,2)                          runtime_compile
2700: wgs=128 tpt=90  factors=(3,10,10,3,3)                        runtime_compile
2880: wgs=256 tpt=96  factors=(10,6,6,2,2,2)                       runtime_compile
2916: wgs=256 tpt=243 factors=(6,6,3,3,3,3)                        runtime_compile
3000: wgs=128 tpt=100 factors=(10,3,10,10)                        runtime_compile
3072: wgs=256 tpt=256 factors=(6,4,4,4,4,2)                        runtime_compile
3125: wgs=128 tpt=125 factors=(5,5,5,5,5)                          runtime_compile
3200: wgs=256 tpt=160 factors=(10,10,4,4,2)                        runtime_compile
3240: wgs=128 tpt=108 factors=(3,3,10,6,6)                        runtime_compile
3375: wgs=256 tpt=225 factors=(5,5,5,3,3,3)                        runtime_compile
3456: wgs=256 tpt=144 factors=(6,6,6,4,4)                          runtime_compile
3600: wgs=256 tpt=120 factors=(10,10,6,6)                          runtime_compile
3645: wgs=256 tpt=243 factors=(5,3,3,3,3,3,3)                      runtime_compile
3750: wgs=256 tpt=125 factors=(3,5,5,10,5)                        runtime_compile
3840: wgs=256 tpt=128 factors=(10,6,2,2,2,2,2,2)                   runtime_compile
3888: wgs=512 tpt=324 factors=(16,3,3,3,3,3)                       runtime_compile
4000: wgs=256 tpt=200 factors=(10,10,10,4)                        runtime_compile
4050: wgs=256 tpt=135 factors=(10,5,3,3,3,3)                       runtime_compile
4096: wgs=256 tpt=256 factors=(16,16,16)                          runtime_compile
```

**Lengths > 4096 (non-power-of-2, `precision=['sp','hp']` only — no double!):**
```
4704: wgs=256 tpt=224 factors=(8,4,7,7,3)   precision=[sp,hp]  runtime_compile
5488: wgs=256 tpt=196 factors=(7,4,7,4,7)   precision=[sp,hp]  runtime_compile
6144: wgs=512 tpt=512 factors=(16,4,8,3,4)  precision=[sp,hp]  runtime_compile
6561: wgs=256 tpt=243 factors=(3,3,3,3,3,3,3,3) precision=[sp,hp] runtime_compile
8192: wgs=512 tpt=512 factors=(16,4,4,4,8)  precision=[sp,hp]  runtime_compile
```

**Second block — "configs for 160KiB LDS"** (guarded by `lds_size_bytes=lds_config.SIZE_160KiB.value`; only usable on GPUs whose `max_lds_bytes` is set to 160KiB, i.e. CDNA3/CDNA4-class parts such as gfx942/gfx950 — see `config_arch.py`'s `supported_arch`/`lds_config` enums, confirmed to exist but the exact arch→LDS-size wiring was not traced further in this pass):
```
4704:  wgs=256 tpt=224 factors=(8,4,7,7,3)               [160KiB, all precisions]
5488:  wgs=256 tpt=196 factors=(7,4,7,4,7)               [160KiB, all precisions]
6144:  wgs=384 tpt=256 factors=(4,8,8,8,3)               [160KiB, all precisions]
6561:  wgs=256 tpt=243 factors=(3,3,3,3,3,3,3,3)         [160KiB, all precisions]
8192:  wgs=512 tpt=512 factors=(16,4,16,8)               [160KiB, all precisions]
9216:  wgs=512 tpt=512 factors=(4,8,4,4,3,6)             [160KiB, all precisions]
10000: wgs=512 tpt=500 factors=(4,5,5,10,10)             [160KiB, all precisions]
10240: wgs=512 tpt=512 factors=(8,4,4,4,5,4)             [160KiB, all precisions]
10752: wgs=512 tpt=512 factors=(4,16,8,7,3)  precision=[sp,hp] [160KiB]
11200: wgs=512 tpt=448 factors=(4,7,5,16,5)  precision=[sp,hp] [160KiB]
12288: wgs=512 tpt=512 factors=(8,8,4,6,8)   precision=[sp,hp] [160KiB]
16384: wgs=512 tpt=512 factors=(8,16,4,8,4)  precision=[sp,hp] [160KiB]
16807: wgs=384 tpt=343 factors=(7,7,7,7,7)   precision=[sp,hp] [160KiB]
18816: wgs=512 tpt=448 factors=(8,8,7,7,6)   precision=[sp,hp] [160KiB]
19200: wgs=512 tpt=480 factors=(8,10,8,5,6)  precision=[sp,hp] [160KiB]
20480: wgs=512 tpt=512 factors=(4,4,16,10,8) precision=[sp,hp] [160KiB]
```

**Highest length defined in `config_sbrr.py`: 20480** (only for 160-KiB-LDS archs and only single/half precision). For the **default 64-KiB-LDS path** (which covers the vast majority of production GPUs, e.g. gfx906/gfx908/gfx90a/RDNA), the highest single-kernel `CS_KERNEL_STOCKHAM` length is **8192** (single/half only) / **4096** for double precision (since the >4096 entries are `precision=['sp','hp']`-restricted).

Source: `library/src/device/kernels/configs/config_sbrr.py`, lines 30-509 (full file quoted above, reformatted for readability; exact source values verified line-by-line).

---

## 2. `map1DLengthSingle` / `map1DLengthDouble` — exact contents

These are **not** computed by a formula — they are literal hardcoded `std::map<size_t,size_t>` tables in `library/src/node_factory.cpp`, lines 42-207. Each entry maps `length[0]` (the full 1D length) → `divLength1` (the **SBCC "column" kernel length**; the companion **SBRC "row" kernel length is `length[0] / divLength1`**, and this exact relationship is enforced by `NodeFactory::Large1DLengthsValid`, quoted below).

```cpp
// node_factory.cpp:42-123
NodeFactory::Map1DLength const NodeFactory::map1DLengthSingle = {
    // pow2 lengths
    {8192, 64},      // CC (64cc + 128rc)
    {16384, 64},     // CC (64cc + 256rc)
    {32768, 128},    // CC (128cc + 256rc)
    {65536, 256},    // CC (256cc + 256rc)
    {131072, 256},   // CC (256cc + 512rc)
    {262144, 512},   // CC (512cc + 512rc)

    // non-pow2 lengths in (4096, 8192)
    {4704, 96}, {4913, 289}, {5488, 112}, {6144, 96}, {6561, 81},

    // non-pow2 lengths in (8192, 16384)
    {9216, 72}, {10000, 100}, {10240, 160}, {10752, 96}, {11200, 224},
    {12288, 192}, {15625, 125},

    // non-pow2 lengths in (16384, 32768)
    {16807, 343}, {17576, 104}, {18816, 168}, {19200, 192}, {19683, 243},
    {20480, 160}, {21504, 168}, {21952, 343}, {23232, 192}, {24576, 192},
    {26000, 208}, {28672, 256}, {32256, 168},

    // non-pow2 lengths in (32768, 65536)
    {34969, 289}, {36864, 192}, {38880, 160}, {40000, 200}, {40960, 160},
    {43008, 168}, {46080, 240}, {48000, 240}, {49152, 256}, {51200, 512},
    {53248, 208}, {57344, 512},

    // non-pow2 lengths in (65536, 131072)
    {68600, 343}, {71344, 208}, {73984, 289}, {76832, 224}, {79860, 60},
    {81920, 160}, {83521, 289}, {87808, 343}, {95832, 72}, {98304, 512},
    {102400, 512}, {106496, 208}, {110592, 216}, {114688, 224},
};
```

`map1DLengthDouble` (node_factory.cpp:125-207) is **almost identical** — same pow2 rows except it *also* includes `{4096, 64}` (single precision has no 4096 entry because the compiled `sbrr` single-kernel already covers 4096 directly), and its (65536,131072) non-pow2 section differs slightly: it has `{78125, 125}` instead of single's `{73984, 289}`, and both then continue with the same remaining entries. (See lines 129-206 for the exact double table if bit-exact double behavior is needed — not reproduced fully here since the task is FP32-focused.)

There is a **third**, much smaller table for the `CS_L1D_TRTRT` non-pow2 fallback:
```cpp
// node_factory.cpp:209-213
NodeFactory::Map1DLength const NodeFactory::map1DLengthTRTRT = {
    // 3^18 performs better when decomposing with 3^7 kernel even when
    // 3^8 is available
    {387420489, 177147},
};
```
This only has one exceptional entry (for `3^18`); it is irrelevant for lengths ≤ 4096.

**Validity invariant** (`node_factory.cpp:285-302`), enforced once per process via `CheckLarge1DMaps`:
```cpp
bool NodeFactory::Large1DLengthsValid(const function_pool&            pool,
                                      const NodeFactory::Map1DLength& map1DLength,
                                      rocfft_precision                precision)
{
    for(const auto& pair : map1DLength)
    {
        if(pair.first % pair.second != 0)
            return false;
        if(!pool.has_SBCC_kernel(pair.second, precision))
            return false;
        if(!pool.has_SBRC_kernel(pair.first / pair.second, precision))
            return false;
    }
    return true;
}
```
i.e. for every `{L, D}` entry: `D` must divide `L` exactly, `D` must have a compiled `CS_KERNEL_STOCKHAM_BLOCK_CC` (SBCC) kernel (see `config_sbcc.py`), and `L/D` must have a compiled `CS_KERNEL_STOCKHAM_BLOCK_RC` (SBRC) kernel (see `config_sbrc.py`).

**Important scope note**: since the smallest key in `map1DLengthSingle` is 4096 (double) / 8192 (single), **this table is never consulted for any pow2 length ≤ 4096 in single precision** — those are always served directly by the compiled `CS_KERNEL_STOCKHAM` single-kernel entries from `config_sbrr.py` (see §1 and §3). The table only becomes relevant starting at length 8192 in single precision.

For **non-power-of-2 lengths**, the same table is consulted at *any* size (not gated by the 4096/8192 pow2 boundary) — see §3's non-pow2 branch. In the ranges of interest here (up to a few thousand, mixed-radix), `map1DLengthSingle`/`Double` have **no entries below 4704**, so non-pow2 lengths under ~4700 that aren't directly in `config_sbrr.py` fall through to `CS_L1D_TRTRT`/`get_explicitly_supported_factor`.

---

## 3. `CS_L1D_CC` vs `CS_L1D_TRTRT` decision logic — `Decide1DScheme`

Full quote, `library/src/node_factory.cpp:628-819`:

```cpp
ComputeScheme
    NodeFactory::Decide1DScheme(const function_pool& pool, NodeMetaData& nodeData, TreeNode* parent)
{
    ComputeScheme scheme = CS_NONE;

    // Build a node for a 1D FFT
    if(!SupportedLength(pool, nodeData.precision, nodeData.length[0]))
        return CS_BLUESTEIN;

    if(pool.has_function(FMKey(nodeData.length[0], nodeData.precision)))
    {
        // 2-kernel plans for lengths > 4k can still do better at
        // smaller batch size due to using more CUs.  So prefer
        // single-kernel only if we launch enough workgroups for all
        // CUs
        if(nodeData.length[0] > 4096)
        {
            auto kernel = pool.get_kernel(FMKey(nodeData.length[0], nodeData.precision));

            const auto totalBatch
                = product(nodeData.length.begin() + 1, nodeData.length.end()) * nodeData.batch;

            if((parent && parent->scheme == CS_BLUESTEIN)
               || totalBatch / kernel.transforms_per_block
                      >= static_cast<size_t>(pool.deviceProp->multiProcessorCount))
                return CS_KERNEL_STOCKHAM;
            // otherwise, fall through to multi-kernel plan
        }
        else
        {
            return CS_KERNEL_STOCKHAM;
        }
    }

    size_t divLength1 = 1;
    bool   failed     = false;

    if(IsPo2(nodeData.length[0])) // multiple kernels involving transpose
    {
        size_t block_threshold = 262144;
        if(nodeData.length[0] <= block_threshold)
        {
            if(nodeData.precision == rocfft_precision_single || nodeData.precision == rocfft_precision_half)
            {
                if(map1DLengthSingle.find(nodeData.length[0]) != map1DLengthSingle.end())
                    divLength1 = map1DLengthSingle.at(nodeData.length[0]);
                else
                    failed = true;
            }
            else
            {
                if(map1DLengthDouble.find(nodeData.length[0]) != map1DLengthDouble.end())
                    divLength1 = map1DLengthDouble.at(nodeData.length[0]);
                else
                    failed = true;
            }
            // for gfx906, 512 CC/RC isn't as fast, so use CRT with a nicer length
            if(is_device_gcn_arch(nodeData.deviceProp, "gfx906") && nodeData.length[0] == 262144)
            {
                divLength1 = 64;
                scheme     = CS_L1D_CRT;
            }
            else
            {
                scheme = CS_L1D_CC;
            }
        }
        else
        {
            // get largest pow2 1D length
            auto largest = pool.get_largest_pow2_length(nodeData.precision);
            if(largest <= 1)
                failed = true;
            else if(nodeData.length[0] > largest * largest)
                divLength1 = nodeData.length[0] / largest;
            else
            {
                size_t in_x = 0;
                size_t len  = nodeData.length[0];
                while(len != 1) { len >>= 1; in_x++; }
                in_x /= 2;
                divLength1 = (size_t)1 << in_x;
            }
            scheme = CS_L1D_TRTRT;
        }
    }
    else // if not Pow2
    {
        if(nodeData.precision == rocfft_precision_single || nodeData.precision == rocfft_precision_half)
        {
            if(map1DLengthSingle.find(nodeData.length[0]) != map1DLengthSingle.end())
            {
                divLength1 = map1DLengthSingle.at(nodeData.length[0]);
                scheme     = CS_L1D_CC;
            }
            else
                failed = true;
        }
        else if(nodeData.precision == rocfft_precision_double)
        {
            if(map1DLengthDouble.find(nodeData.length[0]) != map1DLengthDouble.end())
            {
                divLength1 = map1DLengthDouble.at(nodeData.length[0]);
                scheme     = CS_L1D_CC;
                if(nodeData.length[0] == 43008 && is_device_gcn_arch(nodeData.deviceProp, "gfx90a"))
                    divLength1 = 224;
            }
            else
                failed = true;
        }

        if(failed)
        {
            scheme = CS_L1D_TRTRT;
            auto it = map1DLengthTRTRT.find(nodeData.length[0]);
            if(it != map1DLengthTRTRT.end())
            {
                divLength1 = it->second;
                failed     = false;
            }
            else
            {
                divLength1 = get_explicitly_supported_factor(pool, nodeData.precision, nodeData.length[0]);
                if(divLength1 == 0)
                {
                    auto divLength0 = get_largest_supported_factor(pool, nodeData.precision, nodeData.length[0]);
                    divLength1 = (divLength0 <= 1) ? 0 : nodeData.length[0] / divLength0;
                }
                failed = divLength1 == 0;
            }
        }
    }

    if(failed)
    {
        PrintFailInfo(nodeData.precision, nodeData.length[0], scheme);
        return CS_NONE;
    }

    nodeData.length.emplace_back(divLength1);
    return scheme;
}
```

**Threshold summary:**
- `pool.has_function(FMKey(length, precision))` = single AOT/RTC `CS_KERNEL_STOCKHAM` kernel exists (from `config_sbrr.py`). If `length ≤ 4096`: always used. If `length > 4096`: used only if occupancy heuristic passes (§7) or if the parent is `CS_BLUESTEIN`.
- No single kernel + **power-of-2** + `length ≤ 262144`: `CS_L1D_CC` via `map1DLengthSingle`/`Double` lookup (fails → `CS_NONE`/Bluestein path upstream, since `SupportedLength` would already have caught it — in practice this only "fails" for currently-uncovered exotic archs).
- No single kernel + power-of-2 + `length > 262144`: `CS_L1D_TRTRT` via `get_largest_pow2_length()`.
- No single kernel + **non-power-of-2**: `CS_L1D_CC` via `map1DLengthSingle`/`Double` lookup if present; else `CS_L1D_TRTRT` via `map1DLengthTRTRT` (only `3^18`), else `get_explicitly_supported_factor`, else `get_largest_supported_factor`-based fallback.
- There is **no intermediate fallback** between "in the map" and "TRTRT" for the non-pow2 branch — a miss goes straight into the `failed` block, which is entirely the `CS_L1D_TRTRT` machinery.

---

## 4. `get_largest_pow2_length`, `get_explicitly_supported_factor`, `get_largest_supported_factor`

All three are in `library/src/node_factory.cpp`. `get_largest_pow2_length` is a `function_pool` member (`library/src/include/function_pool.h:391-399`):

```cpp
size_t get_largest_pow2_length(rocfft_precision precision) const
{
    auto supported
        = get_lengths(precision, CS_KERNEL_STOCKHAM, [](size_t len) { return IsPo2(len); });
    auto itr = std::max_element(supported.cbegin(), supported.cend());
    if(itr != supported.cend())
        return *itr;
    return 0;
}
```
i.e. it scans every compiled `CS_KERNEL_STOCKHAM` entry in `function_pool` for the given precision, filters to powers of two, and returns the max. For single precision this will be **4096** on 64-KiB-LDS archs (since the single-kernel pow2 rows in `config_sbrr.py` stop at 4096; 8192 is pow2 but only listed as `precision=['sp','hp']`, so it **is** included for single — meaning on those archs `get_largest_pow2_length(single)` = 8192, not 4096). On 160-KiB-LDS archs it would be even larger via the extra `16384` entry in that config block.

Factor-search helpers, `node_factory.cpp:227-283`:

```cpp
// Search function pool for length where is_supported_factor(length) returns true.
inline size_t search_pool(const function_pool&               pool,
                          rocfft_precision                   precision,
                          size_t                             length,
                          const std::function<bool(size_t)>& is_supported_factor)
{
    auto supported  = pool.get_lengths(precision, CS_KERNEL_STOCKHAM);
    auto comparison = std::greater<size_t>();
    std::sort(supported.begin(), supported.end(), comparison);

    if(supported.empty())
        return 0;

    // start search slightly smaller than sqrt(length)
    auto v     = (size_t)sqrt(length);
    auto lower = std::lower_bound(supported.cbegin(), supported.cend(), v, comparison);
    if(*lower < sqrt(length) && lower != supported.cbegin())
        lower--;

    auto upper = supported.cend();
    auto itr = std::find_if(lower, upper, is_supported_factor);
    if(itr != supported.cend())
        return *itr;
    return 0;
}

// Return largest factor that has BOTH functions in the pool.
inline size_t get_explicitly_supported_factor(const function_pool& pool,
                                              rocfft_precision     precision,
                                              size_t               length)
{
    auto supported_factor = [length, precision, &pool](size_t factor) -> bool {
        bool is_factor        = length % factor == 0;
        bool has_other_kernel = pool.has_function(FMKey(length / factor, precision));
        return is_factor && has_other_kernel;
    };
    auto factor = search_pool(pool, precision, length, supported_factor);
    if(factor > 0 && reverse_factors(length))
        return length / factor;
    return factor;
}

// Return largest factor that has a function in the pool.
inline size_t get_largest_supported_factor(const function_pool& pool,
                                           rocfft_precision     precision,
                                           size_t               length)
{
    auto supported_factor = [length](size_t factor) -> bool {
        bool is_factor = length % factor == 0;
        return is_factor;
    };
    return search_pool(pool, precision, length, supported_factor);
}
```

Algorithm in words: take every compiled single-kernel `CS_KERNEL_STOCKHAM` length for this precision, sort descending, binary-search to a starting point near `sqrt(length)`, then scan **downward through decreasing candidate size** (since the list is sorted largest→smallest and the search starts at ~√length, walking `find_if` from `lower` to `upper=cend()` effectively walks from the sqrt-vicinity point down to the smallest kernel) for the first candidate `factor` that (a) evenly divides `length`, and for `get_explicitly_supported_factor` additionally (b) the complementary factor `length/factor` **also** has its own compiled single kernel. `get_explicitly_supported_factor` is tried first (both halves have kernels — enables `CS_L1D_CRT`/clean TRTRT with two "nice" row kernels); if that returns 0, `get_largest_supported_factor` (only requires the divisor itself to have a kernel, used as `divLength0`, with `divLength1 = length/divLength0` computed afterward) is used as a fallback, and `divLength0 <= 1` is guarded against to avoid infinite recursion.

`reverse_factors` (`node_factory.cpp:221-225`) is a tiny hardcoded exception set `{32256, 43008}` where the factor/complement roles are swapped for performance.

---

## 5. `CS_L1D_CC` — column/row kernel relationship & how each sub-kernel gets its config

From `Decide1DScheme`: for chosen length `L`, `divLength1` = the map's value. `CC1DNode::BuildTree_internal` (`library/src/tree_node_1D.cpp:320-392`) then does:

```cpp
size_t lenFactor1 = length.back();          // = divLength1 (popped off the temp slot)
size_t lenFactor0 = length[0] / lenFactor1; // = L / divLength1
...
// first plan, column-to-column (SBCC)
auto col2colPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CC, this);
col2colPlan->length.push_back(lenFactor1);   // length[0] of the SBCC node = lenFactor1 = divLength1
col2colPlan->length.push_back(lenFactor0);   // length[1] = number of columns = L/divLength1
...
// second plan, row-to-column (SBRC)
auto row2colPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_RC, this);
row2colPlan->length.push_back(lenFactor0);   // length[0] of the SBRC node = lenFactor0 = L/divLength1
row2colPlan->length.push_back(lenFactor1);   // length[1] = lenFactor1
```

So: **the table's `divLength1` value IS the SBCC ("cc") per-block FFT length**, and **`L/divLength1` IS the SBRC ("rc") per-block FFT length** — exactly matching the inline comments in the map (e.g. `{8192, 64} // CC (64cc + 128rc)`, confirming `8192/64=128`).

**Sub-kernel config lookup is a direct, non-recursive `function_pool` lookup, not a call back into `Decide1DScheme`/`has_function`-driven search.** Each leaf node's default `GetKernelKey()` (`library/src/include/tree_node.h:862-869`) is:
```cpp
virtual FMKey GetKernelKey() const
{
    if(specified_key)
        return *specified_key.get();
    return (dimension == 1) ? FMKey(length[0], precision, scheme)
                            : FMKey(length[0], length[1], precision, scheme);
}
```
and `GetKernel()` (`tree_node.h:894-906`) just calls `pool.get_kernel(GetKernelKey())`. For the SBCC node, `scheme = CS_KERNEL_STOCKHAM_BLOCK_CC` and `length[0] = divLength1`, so it looks up `FMKey(divLength1, precision, CS_KERNEL_STOCKHAM_BLOCK_CC)` — i.e. a **direct, exact-length index into the table compiled from `config_sbcc.py`**. Likewise the SBRC node looks up `FMKey(L/divLength1, precision, CS_KERNEL_STOCKHAM_BLOCK_RC, sbrcTranstype)` compiled from `config_sbrc.py` (with the tile-aligned/unaligned transpose-type distinguisher resolved in `SBRCNode::GetKernelKey()`/`sbrc_transpose_type()`, `tree_node_1D.cpp:1195-1216, 1293-1297`).

Both `config_sbcc.py` (full, 26 rows) and `config_sbrc.py` (full, 21 rows) were fetched and are far smaller/simpler than `config_sbrr.py` — they only need to cover the finite set of `divLength1` / `L/divLength1` values that actually appear in `map1DLengthSingle`/`Double`.

`config_sbcc.py` full table (`length: factors, workgroup_size[default 128 if unspecified... actually generator-default applies], notes`):
```
50:  factors=[10,5]                                wgs=256
52:  factors=[13,4]
60:  factors=[6,10]
64:  factors=[8,8]                                 wgs=256
72:  factors=[8,3,3]
80:  factors=[10,8]
81:  factors=[3,3,3,3]
84:  factors=[7,2,6]                                tpt=14
96:  factors=[8,3,4]                                wgs=256
100: factors=[5,5,4]                                wgs=100, half_lds=True
104: factors=[13,8]
108: factors=[6,6,3]
112: factors=[4,7,4]
121: factors=[11,11]                                wgs=128, runtime_compile
125: factors=[5,5,5]
128: factors=[16,8]                                 wgs=256, tpt=16
160: factors=[4,10,4]                               flavour='wide'
168: factors=[7,6,4]                                wgs=128, half_lds=True
169: factors=[13,13]                                wgs=256, runtime_compile
192: factors=[8,6,4]
200: factors=[5,8,5]
208: factors=[13,16]
216: factors=(6,6,6)                                tpt=36
224: factors=[8,7,4]
240: factors=[8,5,6]
243: factors=[3,3,3,3,3]                            wgs=243
256: factors=[8,4,8]                                flavour='wide'
280: factors=[8,5,7]                                runtime_compile
289: factors=[17,17]                                runtime_compile
336: factors=[6,7,8]
343: factors=[7,7,7]
512: factors=[8,8,8]
```
(`use_3steps_large_twd` per-precision flags omitted above for brevity — they control an unrelated twiddle-table optimization, not the scheme decision.)

`config_sbrc.py` full table:
```
17:   factors=[17]                    wgs=256 tpt=1    runtime_compile
49:   factors=[7,7]                   wgs=196 tpt=7    (block_width=28)
50:   factors=[10,5]                  wgs=50  tpt=5    direct_to_from_reg=False (block_width=10)
64:   factors=[4,4,4]                 wgs=128 tpt=16   (block_width=8)
81:   factors=[3,3,3,3]               wgs=243 tpt=27   (block_width=9)
100:  factors=[5,5,4]                 wgs=100 tpt=25   (block_width=4)
112:  factors=[4,7,4]                 wgs=448 tpt=28   (block_width=16)
121:  factors=[11,11]                 wgs=128 tpt=11   runtime_compile
125:  factors=[5,5,5]                 wgs=250 tpt=25   (block_width=10)
128:  factors=[8,4,4]                 wgs=128 tpt=16   (block_width=8)
169:  factors=[13,13]                 wgs=256 tpt=13   runtime_compile
192:  factors=[6,4,4,2]               wgs=256 tpt=32   (block_width=8)
200:  factors=[8,5,5]                 wgs=400 tpt=40   (block_width=10)
243:  factors=[3,3,3,3,3]             wgs=256 tpt=27   runtime_compile (block_width=10)
256:  factors=[4,4,4,4]               wgs=256 tpt=32   (block_width=8)
289:  factors=[17,17]                 wgs=128 tpt=17   runtime_compile
343:  factors=[7,7,7]                 wgs=256 tpt=49   runtime_compile
512:  factors=[8,8,8]                 wgs=512 tpt=128
625:  factors=[5,5,5,5]               wgs=128 tpt=125  runtime_compile
1331: factors=[11,11,11]              wgs=256 tpt=121  runtime_compile
```
All rows use `scheme='CS_KERNEL_STOCKHAM_BLOCK_RC'`.

**Answer to "does each sub-kernel recursively go through `Decide1DScheme`/`has_function`?": No.** They are created directly via `NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CC/_RC, this)` (a raw node constructor, `node_factory.cpp:409-497`) inside `CC1DNode::BuildTree_internal`, bypassing `DecideNodeScheme` entirely, and their kernel parameters are pulled by a single exact-match `function_pool` lookup keyed on `(length, precision, scheme[, sbrc_transpose_type])`.

---

## 6. `CS_L1D_TRTRT` decomposition — `TRTRT1DNode::BuildTree_internal`

Quoted in full above from source (`library/src/tree_node_1D.cpp:34-138`); summary of the tree it builds for length `L` with `lenFactor1 = divLength1` (the temp length appended by `Decide1DScheme`) and `lenFactor0 = L/lenFactor1`:

1. **Transpose 1** (`CS_KERNEL_TRANSPOSE`): `[lenFactor0, lenFactor1] → [lenFactor1, lenFactor0]`.
2. **Row FFT 1** (recursively built via `NodeFactory::CreateExplicitNode` → `DecideNodeScheme`/`Decide1DScheme` again!): length `lenFactor1`, "batch" `lenFactor0`. **This is the one sub-step that IS recursive** — `row1Plan->RecursiveBuildTree(...)` is called, so a length like `lenFactor1` that itself has no single kernel could itself expand into another `CS_L1D_CC`/`CS_L1D_TRTRT`/Bluestein tree.
3. **Transpose 2**.
4. **Row FFT 2** (`CS_KERNEL_STOCKHAM` directly, **not** recursively decided — created via `CreateNodeFromScheme(CS_KERNEL_STOCKHAM, this)`, so this half **requires** `lenFactor0` to have a compiled single kernel; this is why `get_explicitly_supported_factor`/pow2 largest-length search specifically hunts for a `lenFactor0`/`lenFactor1` split where at least the row-FFT-2 side is a known single-kernel length).
5. **Transpose 3**.

Fuse shims (`FT_TRANS_WITH_STOCKHAM` on transpose1+row1, `FT_STOCKHAM_WITH_TRANS` on row2+transpose3) are attempted to collapse kernel count when possible (this is the standard rocFFT fusion pass, not part of the scheme-decision logic itself).

For **pow2 lengths above 262144**, `divLength1` is computed either as `L / largest` (if `L > largest²`) or as `2^(floor(log2(L)/2))` — a roughly-square-root pow2 split — where `largest = pool.get_largest_pow2_length(precision)`. For **non-pow2** TRTRT fallback, `divLength1` comes from `map1DLengthTRTRT` (only the single `3^18` exception) or from `get_explicitly_supported_factor`/`get_largest_supported_factor` (§4).

`CS_L1D_CRT` (`CRT1DNode::BuildTree_internal`, `tree_node_1D.cpp:679-752`) is a 3-kernel variant (`SBCC col2col` → `CS_KERNEL_STOCKHAM row2row` → `transpose`), used only for the gfx906-262144 special case noted in §3.

---

## 7. The >4096 occupancy/batch heuristic

Exact condition, `library/src/node_factory.cpp:643-660` (already quoted in §3, reproduced isolated here):

```cpp
if(nodeData.length[0] > 4096)
{
    auto kernel = pool.get_kernel(FMKey(nodeData.length[0], nodeData.precision));

    const auto totalBatch
        = product(nodeData.length.begin() + 1, nodeData.length.end()) * nodeData.batch;

    if((parent && parent->scheme == CS_BLUESTEIN)
       || totalBatch / kernel.transforms_per_block
              >= static_cast<size_t>(pool.deviceProp->multiProcessorCount))
        return CS_KERNEL_STOCKHAM;
    // otherwise, fall through to multi-kernel plan
}
```

- `totalBatch` = product of all *other* dimensions (for a pure 1D plan this is just the batch count; for a 1D-within-multi-D node it's outer dims × batch).
- `kernel.transforms_per_block = workgroup_size / threads_per_transform` for the compiled single-kernel entry.
- If `totalBatch / transforms_per_block >= multiProcessorCount` (i.e. the single-kernel launch would already fill or oversubscribe every compute unit with at least one workgroup) **OR** the parent is a Bluestein node (chirp setup must reuse the same single kernel), it **still uses `CS_KERNEL_STOCKHAM`** (the single, non-tiled kernel).
- Otherwise (batch too small to occupy all CUs with the coarse-grained single-kernel launch), it **falls through** past this `if` block into the general multi-kernel `CS_L1D_CC`/`CS_L1D_TRTRT` decision logic below (§3) — i.e. the *same* length may use the compiled single kernel or a 2-kernel `CS_L1D_CC` plan **depending on batch size and target GPU's `multiProcessorCount`**, not on length alone. This is the one place the "static/compile-time only" claim breaks down: the choice is runtime-batch-dependent for lengths in `(4096, 8192]` (single precision, where a compiled single kernel exists) and would also apply to any single-kernel-covered length above 4096 in general.
- Below/at `4096` this whole batch-comparison branch is skipped — `CS_KERNEL_STOCKHAM` is used unconditionally whenever a compiled kernel exists, regardless of batch.

---

## 8. Worked examples

Assumptions unless noted: single precision (FP32), default 64-KiB-LDS architecture (e.g. gfx90a-class), c2c, batch large enough to satisfy the §7 occupancy test where relevant (noted per-row), not a Bluestein sub-node.

| Length | In `config_sbrr.py`? | Scheme | Kernel config (wgs / tpt / factors) | Citation |
|---|---|---|---|---|
| 2 | yes | `CS_KERNEL_STOCKHAM` | wgs=64, tpt=1, factors=(2,) | config_sbrr.py:32 |
| 4 | yes | `CS_KERNEL_STOCKHAM` | wgs=128, tpt=1, factors=(4,) | config_sbrr.py:34 |
| 8 | yes | `CS_KERNEL_STOCKHAM` | wgs=64, tpt=4, factors=(4,2) | config_sbrr.py:38 |
| 16 | yes | `CS_KERNEL_STOCKHAM` | wgs=64, tpt=4, factors=(4,4) | config_sbrr.py:46 |
| 32 | yes | `CS_KERNEL_STOCKHAM` | wgs=128, tpt=16, factors=(8,4) [AOT] | config_sbrr.py:58 |
| 64 | yes | `CS_KERNEL_STOCKHAM` | wgs=64, tpt=16, factors=(4,4,4), half_lds=False, direct_to_from_reg=True | config_sbrr.py:78 |
| 128 | yes | `CS_KERNEL_STOCKHAM` | wgs=256, tpt=16, factors=(16,8) [AOT] | config_sbrr.py:110 |
| 256 | yes | `CS_KERNEL_STOCKHAM` | wgs=64, tpt=64, factors=(4,4,4,4) [AOT] | config_sbrr.py:158 |
| 512 | yes | `CS_KERNEL_STOCKHAM` | wgs=64, tpt=64, factors=(8,8,8) [AOT] | config_sbrr.py:226 |
| 1024 | yes | `CS_KERNEL_STOCKHAM` | wgs=128, tpt=128, factors=(8,8,4,4) [AOT] | config_sbrr.py:322 |
| 2048 | yes | `CS_KERNEL_STOCKHAM` | wgs=256, tpt=256, factors=(16,16,8), runtime_compile | config_sbrr.py:456 |
| 4096 | yes | `CS_KERNEL_STOCKHAM` (unconditional — `length ≤ 4096`, no batch check) | wgs=256, tpt=256, factors=(16,16,16), runtime_compile | config_sbrr.py:485; Decide1DScheme else-branch node_factory.cpp:661-664 |
| 8192 | yes (**sp/hp only**) | Batch-dependent (§7): `CS_KERNEL_STOCKHAM` if `totalBatch/transforms_per_block ≥ multiProcessorCount`, else `CS_L1D_CC` | Single-kernel: wgs=512, tpt=512, factors=(16,4,4,4,8). If falls through: `CS_L1D_CC` with divLength1=64 → SBCC(64): wgs=256,factors=[8,8]; SBRC(128): wgs=128,tpt=16,factors=[8,4,4] | config_sbrr.py:490; node_factory.cpp:46 (`{8192,64}`); node_factory.cpp:643-660 |

**Mixed-radix / non-pow2 examples** (all found directly in `config_sbrr.py`'s single-kernel table, so all resolve to `CS_KERNEL_STOCKHAM` unconditionally — none of 3,5,6,7,9,10,12,15,20,100,105 require `map1DLengthSingle`/TRTRT at all):

| Length | Scheme | Kernel config | Citation |
|---|---|---|---|
| 3 | `CS_KERNEL_STOCKHAM` | wgs=64, tpt=1, factors=(3,) | config_sbrr.py:33 |
| 5 | `CS_KERNEL_STOCKHAM` | wgs=128, tpt=1, factors=(5,) | config_sbrr.py:35 |
| 6 | `CS_KERNEL_STOCKHAM` | wgs=128, tpt=1, factors=(6,) | config_sbrr.py:36 |
| 7 | `CS_KERNEL_STOCKHAM` | wgs=64, tpt=1, factors=(7,) | config_sbrr.py:37 |
| 9 | `CS_KERNEL_STOCKHAM` | wgs=64, tpt=3, factors=(3,3) | config_sbrr.py:39 |
| 10 | `CS_KERNEL_STOCKHAM` | wgs=64, tpt=1, factors=(10,) | config_sbrr.py:40 |
| 12 | `CS_KERNEL_STOCKHAM` | wgs=128, tpt=6, factors=(6,2) | config_sbrr.py:42 |
| 15 | `CS_KERNEL_STOCKHAM` | wgs=128, tpt=5, factors=(3,5) | config_sbrr.py:45 |
| 20 | `CS_KERNEL_STOCKHAM` | wgs=256, tpt=10, factors=(5,4) | config_sbrr.py:49 |
| 100 | `CS_KERNEL_STOCKHAM` | wgs=64, tpt=10, factors=(10,10) [AOT] | config_sbrr.py:97 |
| 105 | `CS_KERNEL_STOCKHAM` | wgs=256, tpt=21, factors=(7,3,5), half_lds=False, runtime_compile | config_sbrr.py:100 |

For a length that genuinely is **not** in `config_sbrr.py` and **not** in `map1DLengthSingle` (e.g. a prime like 1009, or a product like `1009 * 2`) the path would be: `SupportedLength` still returns true if the length factors entirely into {2,3,5,7,11,13,17} radices covered by `sbrr_kernels`' factor set — otherwise `CS_BLUESTEIN`. If supported-but-uncovered, non-pow2 branch fails the map lookup → `CS_L1D_TRTRT` via `get_explicitly_supported_factor`/`get_largest_supported_factor` search over all compiled `CS_KERNEL_STOCKHAM` lengths.

---

## Confidence levels & not-verified items

**High confidence (exact source quoted, line-cited, fetched in full from `develop` HEAD on 2026-09-08):**
- Full `config_sbrr.py` contents and highest defined length (item 1).
- Full `map1DLengthSingle` table and the `Large1DLengthsValid`/`CheckLarge1DMaps` invariant (item 2).
- Full `Decide1DScheme` logic including all thresholds (item 3).
- `get_largest_pow2_length`, `get_explicitly_supported_factor`, `get_largest_supported_factor`, `search_pool` (item 4).
- `CC1DNode`/`TRTRT1DNode`/`CRT1DNode::BuildTree_internal`, and the non-recursive direct `FMKey`-based kernel lookup for SBCC/SBRC leaf nodes (items 5, 6).
- The >4096 occupancy heuristic (item 7).
- `config_sbcc.py` and `config_sbrc.py` full contents.

**Medium confidence / partially inferred:**
- `map1DLengthDouble`'s exact full listing beyond the differences called out (I quoted the differences but did not re-transcribe all ~65 identical rows a second time — see raw fetch at node_factory.cpp:125-207 for bit-exact double-precision table if needed).
- The claim that gfx942/gfx950 (CDNA3/4) are the archs that get the 160-KiB-LDS `config_sbrr.py` rows — the `lds_config.SIZE_160KiB` enum and `supported_arch` list confirm the *mechanism* exists (`config_arch.py`), but I did not trace the exact arch→`max_lds_bytes` wiring (likely in `kernel-generator.py` or CMake build logic) to confirm which specific `supported_arch` entries get 160KiB vs 64KiB at build time.
- `get_largest_pow2_length(single)` = 8192 claim (§4) — logically follows from `config_sbrr.py`'s inclusion of the 8192 pow2 row with `precision=['sp','hp']`, but I did not execute/build rocFFT to confirm this empirically; it's a straightforward reading of `get_lengths()`'s filter logic plus the config table.
- Exact runtime batch/`multiProcessorCount` values were not computed for any specific real GPU (would require querying `hipDeviceProp_t` for a target device) — the worked example for length 8192 is described conditionally rather than resolved to one concrete answer, since it is genuinely batch- and device-dependent per the source code itself (this is not a gap in research, it's how rocFFT actually behaves).

**Not verified / out of scope for this pass:**
- The precise mechanism by which `config_sbrr.py`/`config_sbcc.py`/`config_sbrc.py` Python entries get compiled into the AOT `function_map` (i.e. the `kernel-generator.py`/CMake driver that calls `insert_default_entry()`) — `insert_default_entry()`'s signature was seen (function_pool.h:490-494) but its caller chain (the generator script) was not fetched in this pass.
- Runtime-compiled (RTC) kernel behavior vs AOT — whether `runtime_compile=True` entries produce identical `wgs`/`tpt`/`factors` behavior to what's listed, or whether RTC can retune at first-use, was not investigated (out of scope: this only affects compile strategy, not the planner's scheme/config decision, which is fully static from these tables regardless of AOT vs RTC).
