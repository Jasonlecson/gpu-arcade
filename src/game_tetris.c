/*
 * Tetris - Falling blocks, GPU-accelerated line clearing
 */

#include "src/common.h"

#define TET_W 10
#define TET_H 20

static const char *tetris_kernel_src =
"__kernel void tetris_clear(\n"
"    __global int *board, __global int *lines, int W, int H)\n"
"{\n"
"    int y = get_global_id(0);\n"
"    if (y >= H) return;\n"
"    int full = 1;\n"
"    for (int x = 0; x < W; x++) {\n"
"        if (board[y * W + x] == 0) { full = 0; break; }\n"
"    }\n"
"    lines[y] = full;\n"
"}\n";

static const int tet_shapes[7][4][4][2] = {
    {{{0,0},{0,1},{0,2},{0,3}}, {{0,0},{1,0},{2,0},{3,0}}, {{0,0},{0,1},{0,2},{0,3}}, {{0,0},{1,0},{2,0},{3,0}}},
    {{{0,0},{0,1},{1,0},{1,1}}, {{0,0},{0,1},{1,0},{1,1}}, {{0,0},{0,1},{1,0},{1,1}}, {{0,0},{0,1},{1,0},{1,1}}},
    {{{0,0},{0,1},{0,2},{1,2}}, {{0,0},{1,0},{2,0},{0,1}}, {{0,0},{1,0},{1,1},{1,2}}, {{2,0},{0,1},{1,1},{2,1}}},
    {{{0,2},{1,0},{1,1},{1,2}}, {{0,0},{0,1},{1,1},{2,1}}, {{0,0},{0,1},{0,2},{1,0}}, {{0,0},{1,0},{2,0},{2,1}}},
    {{{0,0},{0,1},{1,1},{1,2}}, {{0,1},{1,0},{1,1},{2,0}}, {{0,0},{0,1},{1,1},{1,2}}, {{0,1},{1,0},{1,1},{2,0}}},
    {{{0,1},{0,2},{1,0},{1,1}}, {{0,0},{1,0},{1,1},{2,1}}, {{0,1},{0,2},{1,0},{1,1}}, {{0,0},{1,0},{1,1},{2,1}}},
    {{{0,0},{0,1},{0,2},{1,1}}, {{0,0},{1,0},{1,1},{2,0}}, {{1,0},{1,1},{1,2},{0,1}}, {{0,1},{1,0},{1,1},{2,1}}},
};
static const int tet_colors[7] = {5, 3, 1, 2, 6, 2, 3};

int game_tetris(gpu_ctx_t *gpu) {
    int sw, sh;
    get_terminal_size(&sw, &sh);
    int ox = (sw - TET_W * 2) / 2;
    int oy = (sh - TET_H) / 2;

    int *board = calloc(TET_W * TET_H, sizeof(int));
    int piece = rand() % 7, rot = 0, px = TET_W / 2 - 1, py = 0;
    int next_piece = rand() % 7;
    int score = 0, level = 1, lines_cleared = 0;
    int drop_timer = 0, drop_interval = 500;

    cl_int err;
    cl_mem board_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE, TET_W*TET_H*sizeof(int), NULL, &err);
    cl_mem lines_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE, TET_H*sizeof(int), NULL, &err);
    cl_program prog = gpu_build(gpu, tetris_kernel_src);
    cl_kernel kern = clCreateKernel(prog, "tetris_clear", &err);

    int game_over = 0;
    double last_drop = now_us();

    while (!game_over) {
        int key = read_key();
        if (key == 'q' || key == 'Q' || key == 27) break;

        int new_rot = rot, new_px = px, new_py = py;
        if (key == KEY_LEFT_) new_px--;
        else if (key == KEY_RIGHT_) new_px++;
        else if (key == KEY_DOWN_) new_py++;
        else if (key == KEY_UP_ || key == ' ') new_rot = (rot + 1) % 4;

        int valid = 1;
        for (int i = 0; i < 4; i++) {
            int bx = new_px + tet_shapes[piece][new_rot][i][1];
            int by = new_py + tet_shapes[piece][new_rot][i][0];
            if (bx < 0 || bx >= TET_W || by < 0 || by >= TET_H) { valid = 0; break; }
            if (board[by * TET_W + bx] > 0) { valid = 0; break; }
        }
        if (valid) { rot = new_rot; px = new_px; py = new_py; }

        double now = now_us();
        drop_timer += (int)(now - last_drop);
        last_drop = now;
        if (drop_timer >= drop_interval) {
            drop_timer = 0;
            int can_drop = 1;
            for (int i = 0; i < 4; i++) {
                int bx = px + tet_shapes[piece][rot][i][1];
                int by = py + 1 + tet_shapes[piece][rot][i][0];
                if (by >= TET_H || (by >= 0 && board[by * TET_W + bx] > 0)) { can_drop = 0; break; }
            }
            if (can_drop) {
                py++;
            } else {
                for (int i = 0; i < 4; i++) {
                    int bx = px + tet_shapes[piece][rot][i][1];
                    int by = py + tet_shapes[piece][rot][i][0];
                    if (by >= 0 && by < TET_H && bx >= 0 && bx < TET_W)
                        board[by * TET_W + bx] = tet_colors[piece];
                }

                clEnqueueWriteBuffer(gpu->queue, board_g, CL_TRUE, 0, TET_W*TET_H*sizeof(int), board, 0, NULL, NULL);
                size_t gws = TET_H;
                clSetKernelArg(kern, 0, sizeof(cl_mem), &board_g);
                clSetKernelArg(kern, 1, sizeof(cl_mem), &lines_g);
                int tw = TET_W, th = TET_H;
                clSetKernelArg(kern, 2, sizeof(int), &tw);
                clSetKernelArg(kern, 3, sizeof(int), &th);
                clEnqueueNDRangeKernel(gpu->queue, kern, 1, NULL, &gws, NULL, 0, NULL, NULL);
                clFinish(gpu->queue);

                int line_flags[TET_H];
                clEnqueueReadBuffer(gpu->queue, lines_g, CL_TRUE, 0, TET_H*sizeof(int), line_flags, 0, NULL, NULL);

                int cleared = 0;
                for (int y = TET_H - 1; y >= 0; y--) {
                    if (line_flags[y]) {
                        cleared++;
                        for (int yy = y; yy > 0; yy--)
                            memcpy(&board[yy * TET_W], &board[(yy-1) * TET_W], TET_W * sizeof(int));
                        memset(board, 0, TET_W * sizeof(int));
                        y++;
                    }
                }
                if (cleared) {
                    lines_cleared += cleared;
                    score += cleared * cleared * 100;
                    level = lines_cleared / 10 + 1;
                    drop_interval = 500 - (level - 1) * 40;
                    if (drop_interval < 50) drop_interval = 50;
                }

                piece = next_piece;
                next_piece = rand() % 7;
                rot = 0; px = TET_W / 2 - 1; py = 0;
                for (int i = 0; i < 4; i++) {
                    int bx = px + tet_shapes[piece][rot][i][1];
                    int by = py + tet_shapes[piece][rot][i][0];
                    if (by >= 0 && by < TET_H && board[by * TET_W + bx] > 0)
                        game_over = 1;
                }
            }
        }

        /* Render */
        term_clear();
        for (int y = 0; y < TET_H; y++) {
            for (int x = 0; x < TET_W; x++) {
                int v = board[y * TET_W + x];
                if (v > 0)
                    term_printf(oy + y, ox + x * 2, v, 1, "[]");
                else
                    term_printf(oy + y, ox + x * 2, 7, 0, " .");
            }
        }
        for (int i = 0; i < 4; i++) {
            int bx = px + tet_shapes[piece][rot][i][1];
            int by = py + tet_shapes[piece][rot][i][0];
            if (by >= 0 && by < TET_H)
                term_printf(oy + by, ox + bx * 2, tet_colors[piece], 1, "[]");
        }

        term_printf(oy - 1, ox, 6, 1, " TETRIS ");
        term_printf(oy, ox + TET_W * 2 + 2, 4, 0, "Next:");
        for (int i = 0; i < 4; i++) {
            int nx = tet_shapes[next_piece][0][i][1];
            int ny = tet_shapes[next_piece][0][i][0];
            term_printf(oy + 1 + ny, ox + TET_W * 2 + 2 + nx * 2, tet_colors[next_piece], 1, "[]");
        }
        term_printf(oy + 6, ox + TET_W * 2 + 2, 4, 0, "Score: %d", score);
        term_printf(oy + 7, ox + TET_W * 2 + 2, 4, 0, "Level: %d", level);
        term_printf(oy + 8, ox + TET_W * 2 + 2, 4, 0, "Lines: %d", lines_cleared);
        term_printf(sh - 1, 0, 7, 0, " Arrows=Move Up=Rotate Q=Quit ");
        term_refresh();

        platform_sleep_ms(50);
    }

    if (game_over) {
        term_printf(oy + TET_H/2, ox + TET_W - 4, 2, 1, " GAME OVER! ");
        term_printf(oy + TET_H/2 + 1, ox + TET_W - 6, 4, 0, " Score: %d ", score);
        term_refresh();
            term_wait_key();
    }

    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseMemObject(board_g); clReleaseMemObject(lines_g);
    free(board);
    return 0;
}
