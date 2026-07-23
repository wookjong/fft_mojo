/* Host file access from inside the simulator. See htif.h. */

#include "htif.h"

/* The linker script puts these where fesvr looks for them by name. */
volatile u64 tohost __attribute__((section(".tohost"), aligned(64)));
volatile u64 fromhost __attribute__((section(".tohost"), aligned(64)));

/* Linux numbers: fesvr indexes its table with them. */
#define SYS_openat       56
#define SYS_close        57
#define SYS_read         63
#define SYS_write        64
#define SYS_exit         93
#define SYS_getmainvars 2011

#define AT_FDCWD (-100)

/* One in flight at a time, which is all a single-threaded launcher needs. */
static volatile i64 magic_mem[8] __attribute__((aligned(64)));

i64 htif_syscall(i64 n, i64 a0, i64 a1, i64 a2, i64 a3, i64 a4, i64 a5, i64 a6)
{
    magic_mem[0] = n;
    magic_mem[1] = a0;
    magic_mem[2] = a1;
    magic_mem[3] = a2;
    magic_mem[4] = a3;
    magic_mem[5] = a4;
    magic_mem[6] = a5;
    magic_mem[7] = a6;

    /* The address has to be visible to the host before the doorbell is, and
     * the reply visible to us only after we have seen it. */
    __asm__ volatile("fence" ::: "memory");
    tohost = (u64)(unsigned long)magic_mem;

    /* fromhost is written once the call has been made. Clearing it is part of
     * the protocol -- fesvr will not send another until it is zero. */
    while (fromhost == 0)
        ;
    fromhost = 0;
    __asm__ volatile("fence" ::: "memory");

    return magic_mem[0];
}

int htif_open(const char *path, int flags)
{
    u64 len = 0;
    while (path[len])
        len++;
    /* fesvr wants the length including the terminator. */
    return (int)htif_syscall(SYS_openat, AT_FDCWD, (i64)path, len + 1, flags,
                             0644, 0, 0);
}

i64 htif_read(int fd, void *buf, u64 len)
{
    return htif_syscall(SYS_read, fd, (i64)buf, (i64)len, 0, 0, 0, 0);
}

i64 htif_write(int fd, const void *buf, u64 len)
{
    return htif_syscall(SYS_write, fd, (i64)buf, (i64)len, 0, 0, 0, 0);
}

int htif_close(int fd)
{
    return (int)htif_syscall(SYS_close, fd, 0, 0, 0, 0, 0, 0);
}

i64 htif_read_file(const char *path, void *buf, u64 cap)
{
    int fd = htif_open(path, HTIF_O_RDONLY);
    if (fd < 0)
        return -1;

    i64 total = 0;
    for (;;) {
        /* One byte of headroom, so a file that exactly fills the buffer is
         * told apart from one that overflows it. */
        if ((u64)total >= cap) {
            char probe;
            if (htif_read(fd, &probe, 1) > 0) {
                htif_close(fd);
                return -1;
            }
            break;
        }
        i64 n = htif_read(fd, (char *)buf + total, cap - total);
        if (n <= 0)
            break;
        total += n;
    }
    htif_close(fd);
    return total;
}

i64 htif_write_file(const char *path, const void *buf, u64 len)
{
    int fd = htif_open(path, HTIF_O_WRONLY | HTIF_O_CREAT | HTIF_O_TRUNC);
    if (fd < 0)
        return -1;

    i64 total = 0;
    while ((u64)total < len) {
        i64 n = htif_write(fd, (const char *)buf + total, len - total);
        if (n <= 0) {
            htif_close(fd);
            return -1;
        }
        total += n;
    }
    htif_close(fd);
    return total;
}

/* getmainvars fills a buffer with [argc, argv[0..argc-1], NULL, strings...],
 * the pointers being offsets from the buffer's own address. Read once. */
static u64 mainvars[256];
static int mainvars_read;

static void read_mainvars(void)
{
    if (mainvars_read)
        return;
    mainvars_read = 1;
    if (htif_syscall(SYS_getmainvars, (i64)mainvars, sizeof(mainvars), 0, 0, 0,
                     0, 0) < 0)
        mainvars[0] = 0;
}

int htif_argc(void)
{
    read_mainvars();
    return (int)mainvars[0];
}

const char *htif_argv(int i)
{
    read_mainvars();
    if (i < 0 || i >= (int)mainvars[0])
        return 0;
    return (const char *)mainvars[1 + i];
}

void htif_exit(int code)
{
    htif_syscall(SYS_exit, code, 0, 0, 0, 0, 0, 0);
    for (;;)
        ;
}

void htif_print(const char *s)
{
    u64 len = 0;
    while (s[len])
        len++;
    htif_write(1, s, len);
}
