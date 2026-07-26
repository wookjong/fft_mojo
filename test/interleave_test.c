/* Does sim/interleave.h spread microthreads the way M2NDP-public does?
 *
 *   cc -I.. -o /tmp/t interleave_test.c && /tmp/t interleave.cases
 *
 * The cases come from the reference; see interleave.cases for which commit and
 * which lines of it. This asserts our rule agrees, case by case, packet by
 * packet -- so a change to either side shows up as a failure rather than as a
 * benchmark that quietly runs its work on different cores.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sim/interleave.h"

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "test/interleave.cases";
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path);
        return 2;
    }

    char line[64 * 1024];
    int cases = 0, bad = 0;

    while (fgets(line, sizeof line, f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;

        m2ndp_topology t;
        int used = 0;
        if (sscanf(p, "%li %lu %lu %lu %lu :%n", (long *)&t.base, &t.size,
                   &t.packet, &t.stride, &t.cores, &used) != 5 || !used) {
            fprintf(stderr, "cannot parse: %s", p);
            return 2;
        }

        cases++;
        char *rest = p + used;
        unsigned long u = 0;
        for (;; u++) {
            char *end;
            long want = strtol(rest, &end, 10);
            if (end == rest) break;
            rest = end;

            if (u >= m2ndp_total(&t)) {
                printf("FAIL case %d: more units listed than packets\n", cases);
                bad++;
                break;
            }
            m2ndp_u64 got = m2ndp_core_of(&t, u);
            if ((long)got != want) {
                printf("FAIL case %d: packet %lu -> core %lu, reference says %ld\n",
                       cases, u, (unsigned long)got, want);
                bad++;
                break;
            }
        }
        if (u != m2ndp_total(&t)) {
            printf("FAIL case %d: %lu packets listed, the range holds %lu\n",
                   cases, u, (unsigned long)m2ndp_total(&t));
            bad++;
        }
    }
    fclose(f);

    if (bad) {
        printf("interleave: %d of %d cases disagree with the reference\n", bad, cases);
        return 1;
    }
    printf("interleave: %d cases match the reference\n", cases);
    return 0;
}
