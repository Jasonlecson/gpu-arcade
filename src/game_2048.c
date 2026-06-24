/*
 * 2048 - Slide and merge tiles, grid operations on GPU
 */

#include "src/common.h"

static const char *t2048_kernel_src =
"__kernel void t2048_update(\n"
"    __global int *grid, __global int *score, int W, int H, int dir)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (dir == 0 || dir == 1) {\n"
"        if (id >= W) return;\n"
"        int x = id;\n"
"        int start = (dir == 1) ? H - 1 : 0;\n"
"        int end = (dir == 1) ? -1 : H;\n"
"        int step = (dir == 1) ? -1 : 1;\n"
"        int write = start;\n"
"        int prev = -1;\n"
"        for (int y = start; y != end; y += step) {\n"
"            int v = grid[y * W + x];\n"
"            if (v == 0) continue;\n"
"            if (prev == v) {\n"
"                grid[write * W + x] = v * 2;\n"
"                atomic_add(score, v * 2);\n"
"                prev = -1;\n"
"                write += step;\n"
"            } else {\n"
"                if (prev >= 0) { grid[write * W + x] = prev; write += step; }\n"
"                prev = v;\n"
"            }\n"
"        }\n"
"        if (prev >= 0) grid[write * W + x] = prev;\n"
"        write += step;\n"
"        for (int y = write; y != end; y += step) grid[y * W + x] = 0;\n"
"    } else {\n"
"        if (id >= H) return;\n"
"        int y = id;\n"
"        int start = (dir == 3) ? W - 1 : 0;\n"
"        int end = (dir == 3) ? -1 : W;\n"
"        int step = (dir == 3) ? -1 : 1;\n"
"        int write = start;\n"
"        int prev = -1;\n"
"        for (int x = start; x != end; x += step) {\n"
"            int v = grid[y * W + x];\n"
"            if (v == 0) continue;\n"
"            if (prev == v) {\n"
"                grid[y * W + write] = v * 2;\n"
"                atomic_add(score, v * 2);\n"
"                prev = -1;\n"
"                write += step;\n"
"            } else {\n"
"                if (prev >= 0) { grid[y * W + write] = prev; write += step; }\n"
"                prev = v;\n"
"            }\n"
"        }\n"
"        if (prev >= 0) grid[y * W + write] = prev;\n"
"        write += step;\n"
"        for (int x = write; x != end; x += step) grid[y * W + x] = 0;\n"
"    }\n"
"}\n";

static void spawn_tile(int *grid, int w, int h) {
    int empty[256], ec = 0;
    for (int i = 0; i < w * h && ec < 256; i++)
        if (grid[i] == 0) empty[ec++] = i;
    if (ec > 0) grid[empty[rand() % ec]] = (rand() % 10 < 9) ? 2 : 4;
}

static const int tile_colors[] = {7, 3, 3, 2, 2, 1, 1, 5, 5, 6, 6, 2};

int game_2048(gpu_ctx_t *gpu) {
    int sw, sh;
    get_terminal_size(&sw, &sh);
    int gw = 4, gh = 4;
    int ox = (sw - gw * 6) / 2;
    int oy = (sh - gh * 3) / 2;

    int *grid = calloc(gw * gh, sizeof(int));
    int score = 0;
    spawn_tile(grid, gw, gh);
    spawn_tile(grid, gw, gh);

    cl_int err;
    cl_mem grid_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gw*gh*sizeof(int), grid, &err);
    cl_mem score_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &score, &err);
    cl_program prog = gpu_build(gpu, t2048_kernel_src);
    cl_kernel kern = clCreateKernel(prog, "t2048_update", &err);

    while (1) {
        int key = read_key();
        if (key == 'q' || key == 'Q' || key == 27) break;
        if (key == 'r' || key == 'R') {
            memset(grid, 0, gw*gh*sizeof(int));
            score = 0;
            spawn_tile(grid, gw, gh);
            spawn_tile(grid, gw, gh);
            clEnqueueWriteBuffer(gpu->queue, grid_g, CL_TRUE, 0, gw*gh*sizeof(int), grid, 0, NULL, NULL);
            clEnqueueWriteBuffer(gpu->queue, score_g, CL_TRUE, 0, sizeof(int), &score, 0, NULL, NULL);
            continue;
        }

        int dir = -1;
        if (key == KEY_UP_) dir = 0;
        else if (key == KEY_DOWN_) dir = 1;
        else if (key == KEY_LEFT_) dir = 2;
        else if (key == KEY_RIGHT_) dir = 3;
        if (dir < 0) continue;

        int old_grid[16];
        memcpy(old_grid, grid, gw*gh*sizeof(int));

        clEnqueueWriteBuffer(gpu->queue, score_g, CL_TRUE, 0, sizeof(int), &(int){0}, 0, NULL, NULL);
        size_t gws = (dir <= 1) ? gw : gh;
        int tw = gw, th = gh;
        clSetKernelArg(kern, 0, sizeof(cl_mem), &grid_g);
        clSetKernelArg(kern, 1, sizeof(cl_mem), &score_g);
        clSetKernelArg(kern, 2, sizeof(int), &tw);
        clSetKernelArg(kern, 3, sizeof(int), &th);
        clSetKernelArg(kern, 4, sizeof(int), &dir);
        clEnqueueNDRangeKernel(gpu->queue, kern, 1, NULL, &gws, NULL, 0, NULL, NULL);
        clFinish(gpu->queue);
        clEnqueueReadBuffer(gpu->queue, grid_g, CL_TRUE, 0, gw*gh*sizeof(int), grid, 0, NULL, NULL);
        clEnqueueReadBuffer(gpu->queue, score_g, CL_TRUE, 0, sizeof(int), &score, 0, NULL, NULL);

        int moved = memcmp(old_grid, grid, gw*gh*sizeof(int)) != 0;
        if (moved) spawn_tile(grid, gw, gh);

        /* Check game over */
        int game_over = 1;
        for (int y = 0; y < gh && game_over; y++)
            for (int x = 0; x < gw && game_over; x++) {
                if (grid[y*gw+x] == 0) game_over = 0;
                if (x < gw-1 && grid[y*gw+x] == grid[y*gw+x+1]) game_over = 0;
                if (y < gh-1 && grid[y*gw+x] == grid[(y+1)*gw+x]) game_over = 0;
            }

        /* Render */
        term_clear();
        term_printf(oy - 1, ox, 6, 1, " 2048 | Score: %d | R=Restart Q=Quit ", score);

        for (int y = 0; y < gh; y++) {
            for (int x = 0; x < gw; x++) {
                int v = grid[y*gw+x];
                int ci = 0;
                if (v > 0) { int t = v; while (t > 2) { t >>= 1; ci++; } }
                if (ci > 11) ci = 11;
                int color = (v == 0) ? 7 : tile_colors[ci];
                if (v == 0)
                    term_printf(oy + y*3, ox + x*6, 7, 0, "  .   ");
                else
                    term_printf(oy + y*3, ox + x*6, color, 1, " %4d ", v);
            }
        }

        term_printf(oy + gh*3 + 1, ox, 7, 0, " Arrow Keys=Slide  R=Restart  Q=Quit ");

#ifdef USE_WINCONSOLE
#else
        refresh();
#endif

        if (game_over) {
            term_printf(oy + gh*3/2, ox + gw*3 - 4, 2, 1, " GAME OVER! ");
#ifdef USE_WINCONSOLE
#else
            refresh();
#endif
            nodelay(stdscr, FALSE);
            getch();
            nodelay(stdscr, TRUE);
            break;
        }

        platform_sleep_ms(50);
    }

    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseMemObject(grid_g); clReleaseMemObject(score_g);
    free(grid);
    return 0;
}
