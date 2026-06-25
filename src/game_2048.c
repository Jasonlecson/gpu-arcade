/*
 * 2048 - Slide and merge tiles
 * GPU: parallel game-over detection (scan all cells)
 */

#include "src/common.h"

static const char *t2048_kernel_src =
"__kernel void check_gameover(\n"
"    __global const int *grid, __global int *result, int W, int H)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int total = W * H;\n"
"    if (id >= total) return;\n"
"    int x = id % W;\n"
"    int y = id / W;\n"
"    int v = grid[id];\n"
"    if (v == 0) { result[0] = 0; return; }\n"
"    if (x + 1 < W && grid[id + 1] == v) { result[0] = 0; return; }\n"
"    if (y + 1 < H && grid[id + W] == v) { result[0] = 0; return; }\n"
"}\n";

static void spawn_tile(int *grid, int w, int h) {
    int empty[256], ec = 0;
    for (int i = 0; i < w * h && ec < 256; i++)
        if (grid[i] == 0) empty[ec++] = i;
    if (ec > 0) grid[empty[rand() % ec]] = (rand() % 10 < 9) ? 2 : 4;
}

static int slide_row(int *row, int n) {
    int score = 0;
    int tmp[16], tc = 0;
    for (int i = 0; i < n; i++)
        if (row[i] != 0) tmp[tc++] = row[i];
    int out[16], oc = 0;
    for (int i = 0; i < tc; i++) {
        if (i + 1 < tc && tmp[i] == tmp[i + 1]) {
            int v = tmp[i] * 2;
            out[oc++] = v;
            score += v;
            i++;
        } else {
            out[oc++] = tmp[i];
        }
    }
    for (int i = 0; i < n; i++)
        row[i] = (i < oc) ? out[i] : 0;
    return score;
}

static void slide_left(int *grid, int w, int h, int *score) {
    for (int y = 0; y < h; y++)
        *score += slide_row(&grid[y * w], w);
}

static void slide_right(int *grid, int w, int h, int *score) {
    for (int y = 0; y < h; y++) {
        int row[16];
        for (int x = 0; x < w; x++) row[x] = grid[y * w + w - 1 - x];
        int s = slide_row(row, w);
        for (int x = 0; x < w; x++) grid[y * w + w - 1 - x] = row[x];
        *score += s;
    }
}

static void slide_up(int *grid, int w, int h, int *score) {
    for (int x = 0; x < w; x++) {
        int row[16];
        for (int y = 0; y < h; y++) row[y] = grid[y * w + x];
        int s = slide_row(row, h);
        for (int y = 0; y < h; y++) grid[y * w + x] = row[y];
        *score += s;
    }
}

static void slide_down(int *grid, int w, int h, int *score) {
    for (int x = 0; x < w; x++) {
        int row[16];
        for (int y = 0; y < h; y++) row[y] = grid[(h - 1 - y) * w + x];
        int s = slide_row(row, h);
        for (int y = 0; y < h; y++) grid[(h - 1 - y) * w + x] = row[y];
        *score += s;
    }
}

static const int tile_colors[] = {7, 3, 3, 2, 2, 1, 1, 5, 5, 6, 6, 2, 2};

static void render_2048(int *grid, int gw, int gh, int ox, int oy, int score, int game_over, gpu_ctx_t *gpu,
                         cl_mem grid_g, cl_mem res_g, cl_kernel kern, int fc, double sess) {
    int is_over = 0;
    if (!game_over) {
        clEnqueueWriteBuffer(gpu->queue, grid_g, CL_TRUE, 0, gw*gh*sizeof(int), grid, 0, NULL, NULL);
        int one = 1;
        clEnqueueWriteBuffer(gpu->queue, res_g, CL_TRUE, 0, sizeof(int), &one, 0, NULL, NULL);
        size_t gws = gw * gh;
        int tw = gw, th = gh;
        clSetKernelArg(kern, 0, sizeof(cl_mem), &grid_g);
        clSetKernelArg(kern, 1, sizeof(cl_mem), &res_g);
        clSetKernelArg(kern, 2, sizeof(int), &tw);
        clSetKernelArg(kern, 3, sizeof(int), &th);
        clEnqueueNDRangeKernel(gpu->queue, kern, 1, NULL, &gws, NULL, 0, NULL, NULL);
        clFinish(gpu->queue);
        clEnqueueReadBuffer(gpu->queue, res_g, CL_TRUE, 0, sizeof(int), &is_over, 0, NULL, NULL);
    } else {
        is_over = 1;
    }

    term_clear();
    term_printf(oy - 1, ox, 6, 1, " 2048 | Score: %d | R=Restart Q=Quit ", score);

    for (int y = 0; y < gh; y++)
        for (int x = 0; x < gw; x++) {
            int v = grid[y*gw+x];
            int ci = 0;
            if (v > 0) { int t = v; while (t > 2) { t >>= 1; ci++; } }
            if (ci > 12) ci = 12;
            int color = (v == 0) ? 7 : tile_colors[ci];
            if (v == 0)
                term_printf(oy + y*3, ox + x*6, 7, 0, "  .   ");
            else
                term_printf(oy + y*3, ox + x*6, color, 1, " %4d ", v);
        }

    term_printf(oy + gh*3 + 1, ox, 7, 0, " Arrows=Slide  R=Restart  Q=Quit ");
    draw_metrics(gpu, oy + gh*3, fc, sess, 0);
    term_refresh();

    if (is_over && !game_over) {
        term_printf(oy + gh*3/2, ox + gw*3 - 4, 2, 1, " GAME OVER! ");
        term_printf(oy + gh*3/2 + 1, ox + gw*3 - 10, 4, 0, " R=Restart  Q=Quit ");
        term_refresh();
    }
}

int game_2048(gpu_ctx_t *gpu) {
    int prev_w = 0, prev_h = 0;

restart: ;
    int sw, sh;
    get_terminal_size(&sw, &sh);
    prev_w = sw; prev_h = sh;
    int gw = 4, gh = 4;
    int ox = (sw - gw * 6) / 2;
    int oy = (sh - gh * 3) / 2;

    int *grid = calloc(gw * gh, sizeof(int));
    int score = 0;
    spawn_tile(grid, gw, gh);
    spawn_tile(grid, gw, gh);

    cl_int err;
    cl_mem grid_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gw*gh*sizeof(int), grid, &err);
    cl_mem res_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    cl_program prog = gpu_build(gpu, t2048_kernel_src);
    cl_kernel kern = clCreateKernel(prog, "check_gameover", &err);

    int game_over = 0;
    int fc = 0;
    double sess = now_us();
    render_2048(grid, gw, gh, ox, oy, score, game_over, gpu, grid_g, res_g, kern, fc, sess);

    while (1) {
        term_size_t ts = check_resize(&prev_w, &prev_h);
        if (ts.changed) { ox = (prev_w - gw * 6) / 2; oy = (prev_h - gh * 3) / 2; render_2048(grid, gw, gh, ox, oy, score, game_over, gpu, grid_g, res_g, kern, fc, sess); }

        int key = read_key();
        if (key == 'q' || key == 'Q' || key == 27) break;
        if (key == 'r' || key == 'R') {
            memset(grid, 0, gw*gh*sizeof(int));
            score = 0;
            spawn_tile(grid, gw, gh);
            spawn_tile(grid, gw, gh);
            game_over = 0;
            render_2048(grid, gw, gh, ox, oy, score, game_over, gpu, grid_g, res_g, kern, fc, sess);
            continue;
        }

        if (game_over) { platform_sleep_ms(16); continue; }

        if (key == KEY_UP_ || key == KEY_DOWN_ || key == KEY_LEFT_ || key == KEY_RIGHT_) {
            int old_grid[16] = {0};
            memcpy(old_grid, grid, gw*gh*sizeof(int));

            if (key == KEY_UP_) slide_up(grid, gw, gh, &score);
            else if (key == KEY_DOWN_) slide_down(grid, gw, gh, &score);
            else if (key == KEY_LEFT_) slide_left(grid, gw, gh, &score);
            else if (key == KEY_RIGHT_) slide_right(grid, gw, gh, &score);

            if (memcmp(old_grid, grid, gw*gh*sizeof(int)) != 0) {
                spawn_tile(grid, gw, gh);
            }

            render_2048(grid, gw, gh, ox, oy, score, game_over, gpu, grid_g, res_g, kern, ++fc, sess);
        }

        platform_sleep_ms(16);
    }

    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseMemObject(grid_g); clReleaseMemObject(res_g);
    free(grid);
    return 0;
}
