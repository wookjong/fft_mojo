/* Host file access from inside the simulator.
 *
 * Spike's front-end server proxies system calls: the target writes a block
 * describing one into memory, hands the address to the host through tohost,
 * and the host runs it for real and writes the result back. That is how a
 * bare-metal program with no operating system under it opens a file.
 *
 * It is what lets a task read its input and write its output rather than
 * carrying both as constants in the binary. Inputs can then change without a
 * rebuild, results can be compared on the host where the tools are, and
 * neither is bounded by what fits in an assembler literal.
 *
 * The syscall numbers are Linux's, because that is the table fesvr indexes.
 */

#ifndef M2NDP_SIM_HTIF_H
#define M2NDP_SIM_HTIF_H

typedef unsigned long u64;
typedef long i64;

#define HTIF_O_RDONLY 0
#define HTIF_O_WRONLY 01
#define HTIF_O_CREAT  0100
#define HTIF_O_TRUNC  01000

/* The raw interface. Rarely wanted directly. */
i64 htif_syscall(i64 n, i64 a0, i64 a1, i64 a2, i64 a3, i64 a4, i64 a5, i64 a6);

/* Files. Paths are resolved on the host, relative to where spike was run. */
int htif_open(const char *path, int flags);
i64 htif_read(int fd, void *buf, u64 len);
i64 htif_write(int fd, const void *buf, u64 len);
int htif_close(int fd);

/* The whole of a file into a buffer, or -1 if it does not fit. Most tasks
 * want exactly this and nothing else. */
i64 htif_read_file(const char *path, void *buf, u64 cap);

/* A buffer to a file, created or truncated. */
i64 htif_write_file(const char *path, const void *buf, u64 len);

/* The arguments spike was given, so a task can be told which files to use
 * rather than having them compiled in. argv[0] is the ELF. */
int htif_argc(void);
const char *htif_argv(int i);

/* Ends the run. The exit code reaches the shell. */
void htif_exit(int code) __attribute__((noreturn));

/* Text to the host's stdout, for saying what went wrong. */
void htif_print(const char *s);

#endif
