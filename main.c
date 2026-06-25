/*
 * GPU Arcade - FC-style multi-game collection
 * All game logic runs on GPU via OpenCL
 */

#include "src/common.h"

/* Game entry point declarations */
extern int game_snake(gpu_ctx_t *gpu);
extern int game_tetris(gpu_ctx_t *gpu);
extern int game_life(gpu_ctx_t *gpu);
extern int game_pong(gpu_ctx_t *gpu);
extern int game_breakout(gpu_ctx_t *gpu);
extern int game_2048(gpu_ctx_t *gpu);
extern int game_mandelbrot(gpu_ctx_t *gpu);
extern int game_sand(gpu_ctx_t *gpu);
extern int game_raycaster(gpu_ctx_t *gpu);

typedef struct {
    const char *name;
    const char *desc;
    int (*run)(gpu_ctx_t *gpu);
} game_entry_t;

static game_entry_t games[] = {
    { "Snake",       "Classic snake - eat food, grow longer",    game_snake     },
    { "Tetris",      "Stack falling blocks, clear lines",        game_tetris    },
    { "Game of Life", "Cellular automaton - watch evolution",    game_life      },
    { "Pong",        "Classic paddle ball game",                 game_pong      },
    { "Breakout",    "Break bricks with a bouncing ball",        game_breakout  },
    { "2048",        "Slide and merge tiles to reach 2048",      game_2048      },
    { "Mandelbrot",  "Explore the fractal - zoom infinitely",    game_mandelbrot },
    { "Falling Sand", "Particle sim - sand/water/fire/stone",    game_sand      },
    { "Raycaster",   "Wolfenstein 3D style 3D rendering",        game_raycaster },
};
#define NUM_GAMES (int)(sizeof(games) / sizeof(games[0]))

static void draw_box(int y, int x, int h, int w, int color) {
    for (int i = 0; i < w; i++) {
        term_printf(y, x + i, color, 0, "-");
        term_printf(y + h - 1, x + i, color, 0, "-");
    }
    for (int i = 0; i < h; i++) {
        term_printf(y + i, x, color, 0, "|");
        term_printf(y + i, x + w - 1, color, 0, "|");
    }
    term_printf(y, x, color, 0, "+");
    term_printf(y, x + w - 1, color, 0, "+");
    term_printf(y + h - 1, x, color, 0, "+");
    term_printf(y + h - 1, x + w - 1, color, 0, "+");
}

static int menu(gpu_ctx_t *gpu) {
    int selected = 0;
    int sw, sh;
    get_terminal_size(&sw, &sh);

    while (1) {
        get_terminal_size(&sw, &sh);
        term_clear();

        int cx = sw / 2;
        int cy = sh / 2;

        /* Title */
        term_printf(cy - 12, cx - 20, 3, 1, "  ____  _   _  _____       _                ");
        term_printf(cy - 11, cx - 20, 3, 1, " / ___|| | | ||  ___|__ _| | ___ __ _ _ __  ");
        term_printf(cy - 10, cx - 20, 3, 1, "| |  _ | |_| || |_ / __| |/ __/ _` | '__| ");
        term_printf(cy -  9, cx - 20, 3, 1, "| |_| ||  _  ||  _| (__| | (_| (_| | |    ");
        term_printf(cy -  8, cx - 20, 3, 1, " \\____|_|_|_||_|  \\___|_|\\___\\__,_|_|    ");

        /* GPU info - truncate device name to fit */
        char short_name[32];
        snprintf(short_name, sizeof(short_name), "%.30s", gpu->device_name);
        term_printf(cy - 6, cx - 25, 5, 0, " GPU: %s | %s | %d CU @ %d MHz ",
                    gpu->platform_name, short_name, gpu->compute_units, gpu->clock_freq);

        /* Game list */
        draw_box(cy - 4, cx - 24, NUM_GAMES + 2, 48, 4);

        for (int i = 0; i < NUM_GAMES; i++) {
            int color = (i == selected) ? 3 : 4;
            int bold = (i == selected) ? 1 : 0;
            const char *cursor = (i == selected) ? " > " : "   ";
            term_printf(cy - 3 + i, cx - 22, color, bold,
                        "%s%d. %-15s %s", cursor, i + 1, games[i].name, games[i].desc);
        }

        /* Footer */
        term_printf(cy + NUM_GAMES - 1, cx - 22, 7, 0, " Up/Down=Select  Enter=Play  Q=Quit ");

        /* Version */
        term_printf(sh - 1, 0, 7, 0, " GPU Arcade v1.0 | github.com/Jasonlecson/gpu-arcade ");
        term_refresh();

        int key = read_key();
        static FILE *mdbg = NULL;
        if (!mdbg) mdbg = fopen("gpu_arcade_menu_debug.log", "w");
        if (mdbg && key != -1) { fprintf(mdbg, "key=%d (sel=%d)\n", key, selected); fflush(mdbg); }
        if (key == KEY_UP_ && selected > 0) selected--;
        else if (key == KEY_DOWN_ && selected < NUM_GAMES - 1) selected++;
        else if (key >= '1' && key <= '0' + NUM_GAMES) selected = key - '1';
        else if (key == '\n' || key == ' ') return selected;
        else if (key == 'q' || key == 'Q' || key == 27) return -1;

        platform_sleep_ms(30);
    }
}

int main(void) {
    srand((unsigned)time(NULL));
    term_init();
    gpu_ctx_t gpu = gpu_init();

    FILE *dbg = fopen("gpu_arcade_debug.log", "w");
    if (dbg) { fprintf(dbg, "main: init done\n"); fflush(dbg); }

    while (1) {
        int choice = menu(&gpu);
        if (dbg) { fprintf(dbg, "main: menu returned %d\n", choice); fflush(dbg); }
        if (choice < 0) break;

        if (dbg) { fprintf(dbg, "main: calling game %d\n", choice); fflush(dbg); }
        int play_again = games[choice].run(&gpu);
        if (dbg) { fprintf(dbg, "main: game returned %d\n", play_again); fflush(dbg); }
        (void)play_again;
    }

    if (dbg) { fprintf(dbg, "main: exiting\n"); fclose(dbg); }
    gpu_free(&gpu);
    term_cleanup();
    printf("Thanks for playing GPU Arcade!\n");
    return 0;
}
