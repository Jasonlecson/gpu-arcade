#ifndef GPU_ARCADE_COMMON_H
#define GPU_ARCADE_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <locale.h>

#ifdef _WIN32
  #include <windows.h>
  #include <conio.h>
  #define USE_WINCONSOLE 1
#else
  #include <unistd.h>
  #include <sys/ioctl.h>
  #include <curses.h>
#endif

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

/* ======================== Platform: timing ======================== */

static double now_us(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER cnt;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / freq.QuadPart * 1e6;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
#endif
}

/* ======================== Platform: terminal ======================== */

static void get_terminal_size(int *w, int *h) {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info);
    *w = info.srWindow.Right - info.srWindow.Left + 1;
    *h = info.srWindow.Bottom - info.srWindow.Top + 1;
#else
    struct winsize ws;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
    *w = ws.ws_col;
    *h = ws.ws_row;
#endif
    if (*w < 40) *w = 40;
    if (*h < 20) *h = 20;
}

static void platform_sleep_ms(int ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    usleep(ms * 1000);
#endif
}

/* ======================== Windows Console API ======================== */

#ifdef USE_WINCONSOLE

static HANDLE hCon, hConIn;
static CHAR_INFO *g_fb;
static int g_fb_w, g_fb_h;

static void term_init(void) {
    hCon = GetStdHandle(STD_OUTPUT_HANDLE);
    hConIn = GetStdHandle(STD_INPUT_HANDLE);
    SetConsoleOutputCP(65001);
    CONSOLE_CURSOR_INFO ci = {1, FALSE};
    SetConsoleCursorInfo(hCon, &ci);
    get_terminal_size(&g_fb_w, &g_fb_h);
    g_fb = malloc(g_fb_w * g_fb_h * sizeof(CHAR_INFO));
}

static void term_cleanup(void) {
    free(g_fb);
    CONSOLE_CURSOR_INFO ci = {1, TRUE};
    SetConsoleCursorInfo(hCon, &ci);
    system("cls");
}

static WORD win_attr(int color, int bold) {
    WORD a = 0;
    switch (color) {
        case 1: a = FOREGROUND_GREEN; break;
        case 2: a = FOREGROUND_RED; break;
        case 3: a = FOREGROUND_RED | FOREGROUND_GREEN; break;
        case 4: a = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE; break;
        case 5: a = FOREGROUND_GREEN | FOREGROUND_BLUE; break;
        case 6: a = FOREGROUND_RED | FOREGROUND_BLUE; break;
        case 7: a = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE; break;
        default: a = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE; break;
    }
    if (bold) a |= FOREGROUND_INTENSITY;
    return a;
}

static void term_clear(void) {
    int tw, th;
    get_terminal_size(&tw, &th);
    g_fb_w = tw;
    g_fb_h = th;
    g_fb = realloc(g_fb, g_fb_w * g_fb_h * sizeof(CHAR_INFO));
    for (int i = 0; i < g_fb_h * g_fb_w; i++) {
        g_fb[i].Char.UnicodeChar = ' ';
        g_fb[i].Attributes = 0;
    }
}

static void term_refresh(void) {
    COORD buf_sz = {(SHORT)g_fb_w, (SHORT)g_fb_h};
    COORD buf_org = {0, 0};
    SMALL_RECT rc = {0, 0, (SHORT)(g_fb_w - 1), (SHORT)(g_fb_h - 1)};
    WriteConsoleOutputW(GetStdHandle(STD_OUTPUT_HANDLE), g_fb, buf_sz, buf_org, &rc);
}

static void term_puts(int y, int x, const char *s, int color, int bold) {
    if (y < 0 || y >= g_fb_h) return;
    WORD attr = win_attr(color, bold);
    int len = (int)strlen(s);
    for (int i = 0; i < len && x + i < g_fb_w; i++) {
        if (x + i < 0) continue;
        int idx = y * g_fb_w + x + i;
        g_fb[idx].Char.UnicodeChar = s[i];
        g_fb[idx].Attributes = attr;
    }
}

static void term_printf(int y, int x, int color, int bold, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    term_puts(y, x, buf, color, bold);
}

static int win_read_key(INPUT_RECORD *ir) {
    int vk = ir->Event.KeyEvent.wVirtualKeyCode;
    switch (vk) {
        case VK_UP:    return 256 + 1;
        case VK_DOWN:  return 256 + 2;
        case VK_LEFT:  return 256 + 3;
        case VK_RIGHT: return 256 + 4;
        case VK_RETURN: return '\n';
        case VK_ESCAPE: return 27;
        default: return ir->Event.KeyEvent.uChar.AsciiChar;
    }
}

static int term_getch(void) {
    INPUT_RECORD ir;
    DWORD n;
    HANDLE hin = GetStdHandle(STD_INPUT_HANDLE);
    while (1) {
        if (!ReadConsoleInput(hin, &ir, 1, &n)) continue;
        if (ir.EventType != KEY_EVENT) continue;
        if (!ir.Event.KeyEvent.bKeyDown) continue;
        return win_read_key(&ir);
    }
}

static int term_getch_nb(void) {
    INPUT_RECORD ir;
    DWORD n;
    HANDLE hin = GetStdHandle(STD_INPUT_HANDLE);
    while (PeekConsoleInput(hin, &ir, 1, &n) && n > 0) {
        if (ir.EventType == KEY_EVENT && ir.Event.KeyEvent.bKeyDown) {
            ReadConsoleInput(hin, &ir, 1, &n);
            return win_read_key(&ir);
        }
        ReadConsoleInput(hin, &ir, 1, &n);
    }
    return -1;
}

static int term_kbhit(void) {
    INPUT_RECORD ir;
    DWORD n;
    HANDLE hin = GetStdHandle(STD_INPUT_HANDLE);
    while (PeekConsoleInput(hin, &ir, 1, &n) && n > 0) {
        if (ir.EventType == KEY_EVENT && ir.Event.KeyEvent.bKeyDown)
            return 1;
        ReadConsoleInput(hin, &ir, 1, &n);
    }
    return 0;
}

static void term_wait_key(void) { term_getch(); }

#else /* ncurses */

static void term_init(void) {
    setlocale(LC_ALL, "");
    initscr(); cbreak(); noecho(); curs_set(0);
    nodelay(stdscr, TRUE); timeout(50);
    keypad(stdscr, TRUE);
    start_color(); use_default_colors();
    init_pair(1, COLOR_GREEN, -1);
    init_pair(2, COLOR_RED, -1);
    init_pair(3, COLOR_YELLOW, -1);
    init_pair(4, COLOR_WHITE, -1);
    init_pair(5, COLOR_CYAN, -1);
    init_pair(6, COLOR_MAGENTA, -1);
    init_pair(7, COLOR_BLUE, -1);
}

static void term_cleanup(void) { endwin(); }

static void term_puts(int y, int x, const char *s, int color, int bold) {
    int attr = COLOR_PAIR(color);
    if (bold) attr |= A_BOLD;
    mvaddstr(y, x, s);
    /* note: attr must be set before mvaddstr in ncurses;
       we use a wrapper approach */
    (void)attr;
}

static void term_printf(int y, int x, int color, int bold, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    int attr = COLOR_PAIR(color);
    if (bold) attr |= A_BOLD;
    attron(attr);
    mvaddstr(y, x, buf);
    attroff(attr);
}

static void term_clear(void) { erase(); }
static void term_refresh(void) { refresh(); }
static int term_getch(void) { return getch(); }
static int term_kbhit(void) { return 1; } /* ncurses nodelay handles this */
static void term_wait_key(void) { nodelay(stdscr, FALSE); getch(); nodelay(stdscr, TRUE); }

#endif

/* ======================== OpenCL GPU context ======================== */

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue queue;
    char device_name[256];
    char platform_name[256];
    int compute_units;
    int clock_freq;
    int is_gpu;
} gpu_ctx_t;

static void gpu_check(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        term_cleanup();
        fprintf(stderr, "OpenCL error %d: %s\n", err, msg);
        exit(1);
    }
}

static gpu_ctx_t gpu_init(void) {
    gpu_ctx_t g = {0};
    cl_int err;
    cl_uint np;
    clGetPlatformIDs(0, NULL, &np);
    if (np == 0) { term_cleanup(); fprintf(stderr, "No OpenCL platforms\n"); exit(1); }
    cl_platform_id *plats = malloc(np * sizeof(cl_platform_id));
    clGetPlatformIDs(np, plats, NULL);

    for (cl_uint p = 0; p < np && !g.device; p++) {
        cl_uint nd;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, NULL, &nd) == CL_SUCCESS && nd > 0) {
            cl_device_id *devs = malloc(nd * sizeof(cl_device_id));
            clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, nd, devs, NULL);
            g.device = devs[0]; g.platform = plats[p]; g.is_gpu = 1;
            free(devs);
        }
    }
    if (!g.device) {
        for (cl_uint p = 0; p < np && !g.device; p++) {
            cl_uint nd;
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, NULL, &nd) == CL_SUCCESS && nd > 0) {
                cl_device_id *devs = malloc(nd * sizeof(cl_device_id));
                clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, nd, devs, NULL);
                g.device = devs[0]; g.platform = plats[p]; g.is_gpu = 0;
                free(devs);
            }
        }
    }
    free(plats);
    if (!g.device) { term_cleanup(); fprintf(stderr, "No OpenCL devices\n"); exit(1); }

    clGetPlatformInfo(g.platform, CL_PLATFORM_NAME, sizeof(g.platform_name), g.platform_name, NULL);
    clGetDeviceInfo(g.device, CL_DEVICE_NAME, sizeof(g.device_name), g.device_name, NULL);
    clGetDeviceInfo(g.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &g.compute_units, NULL);
    clGetDeviceInfo(g.device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(int), &g.clock_freq, NULL);

    g.ctx = clCreateContext(NULL, 1, &g.device, NULL, NULL, &err);
    gpu_check(err, "context");
#ifdef CL_VERSION_2_0
    g.queue = clCreateCommandQueueWithProperties(g.ctx, g.device,
                (cl_queue_properties[]){CL_QUEUE_PROFILING_ENABLE, 0}, &err);
    if (err != CL_SUCCESS)
#endif
    g.queue = clCreateCommandQueue(g.ctx, g.device, CL_QUEUE_PROFILING_ENABLE, &err);
    gpu_check(err, "queue");
    return g;
}

static cl_program gpu_build(gpu_ctx_t *g, const char *src) {
    cl_int err;
    cl_program prog = clCreateProgramWithSource(g->ctx, 1, &src, NULL, &err);
    gpu_check(err, "program source");
    err = clBuildProgram(prog, 1, &g->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(prog, g->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char *log = malloc(log_sz);
        clGetProgramBuildInfo(prog, g->device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
        term_cleanup();
        fprintf(stderr, "OpenCL build error:\n%s\n", log);
        free(log);
        exit(1);
    }
    return prog;
}

static void gpu_free(gpu_ctx_t *g) {
    clReleaseCommandQueue(g->queue);
    clReleaseContext(g->ctx);
}

/* ======================== Input keys ======================== */

#define KEY_UP_    257
#define KEY_DOWN_  258
#define KEY_LEFT_  259
#define KEY_RIGHT_ 260

static int read_key(void) {
#ifdef USE_WINCONSOLE
    int ch = term_getch_nb();
    return ch; /* -1 if no key, already translated otherwise */
#else
    int ch = term_getch();
    switch (ch) {
        case KEY_UP:    return KEY_UP_;
        case KEY_DOWN:  return KEY_DOWN_;
        case KEY_LEFT:  return KEY_LEFT_;
        case KEY_RIGHT: return KEY_RIGHT_;
        default:        return ch;
    }
#endif
}

#endif /* GPU_ARCADE_COMMON_H */
