/*
 * Breakout - Break bricks with a bouncing ball, collision on GPU
 */

#include "src/common.h"

static const char *breakout_kernel_src =
"__kernel void breakout_update(\n"
"    __global int *bricks, __global float *state,\n"
"    int BW, int BH, int W, int H)\n"
"{\n"
"    if (get_global_id(0) != 0) return;\n"
"    float bx = state[0], by = state[1];\n"
"    float dx = state[2], dy = state[3];\n"
"    float spd = state[4];\n"
"    bx += dx * spd; by += dy * spd;\n"
"    if (bx <= 0 || bx >= W - 1) dx = -dx;\n"
"    if (by <= 0) dy = -dy;\n"
"    if (by >= H) { state[5]=1; state[0]=bx; state[1]=by; state[2]=dx; state[3]=dy; return; }\n"
"    int ibx = (int)bx, iby = (int)by;\n"
"    if (iby >= 0 && iby < BH && ibx >= 0 && ibx < BW) {\n"
"        int idx = iby * BW + ibx;\n"
"        if (bricks[idx] > 0) {\n"
"            bricks[idx] = 0;\n"
"            dy = -dy;\n"
"            state[6] += 10;\n"
"        }\n"
"    }\n"
"    float pad = state[7];\n"
"    int pw = (int)state[8];\n"
"    if (by >= H - 2 && bx >= pad && bx <= pad + pw) { dy = -dy; by = H - 3; }\n"
"    state[0]=bx; state[1]=by; state[2]=dx; state[3]=dy;\n"
"    state[5]=0;\n"
"}\n";

int game_breakout(gpu_ctx_t *gpu) {
    int sw, sh;
    get_terminal_size(&sw, &sh);
    int gw = sw - 2, gh = sh - 4;
    if (gw < 20) gw = 20;
    if (gh < 15) gh = 15;
    int bw = gw, bh = gh / 3;
    int pw = gw / 5;

    int *bricks = calloc(bw * bh, sizeof(int));
    for (int y = 0; y < bh; y++)
        for (int x = 0; x < bw; x++)
            bricks[y * bw + x] = (y + 1);

    float state[9] = {gw/2.0f, gh-5.0f, 0.7f, -1.0f, 0.6f, 0, 0, gw/2.0f - pw/2.0f, (float)pw};

    cl_int err;
    cl_mem bk_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bw*bh*sizeof(int), bricks, &err);
    cl_mem st_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(state), state, &err);
    cl_program prog = gpu_build(gpu, breakout_kernel_src);
    cl_kernel kern = clCreateKernel(prog, "breakout_update", &err);

    while (1) {
        int key = read_key();
        if (key == 'q' || key == 'Q' || key == 27) break;
        if (key == KEY_LEFT_ && state[7] > 0) state[7] -= 2.0f;
        if (key == KEY_RIGHT_ && state[7] < gw - pw) state[7] += 2.0f;

        clEnqueueWriteBuffer(gpu->queue, bk_g, CL_TRUE, 0, bw*bh*sizeof(int), bricks, 0, NULL, NULL);
        clEnqueueWriteBuffer(gpu->queue, st_g, CL_TRUE, 0, sizeof(state), state, 0, NULL, NULL);
        size_t gws = 1;
        clSetKernelArg(kern, 0, sizeof(cl_mem), &bk_g);
        clSetKernelArg(kern, 1, sizeof(cl_mem), &st_g);
        clSetKernelArg(kern, 2, sizeof(int), &bw);
        clSetKernelArg(kern, 3, sizeof(int), &bh);
        clSetKernelArg(kern, 4, sizeof(int), &gw);
        clSetKernelArg(kern, 5, sizeof(int), &gh);
        clEnqueueNDRangeKernel(gpu->queue, kern, 1, NULL, &gws, NULL, 0, NULL, NULL);
        clFinish(gpu->queue);
        clEnqueueReadBuffer(gpu->queue, bk_g, CL_TRUE, 0, bw*bh*sizeof(int), bricks, 0, NULL, NULL);
        clEnqueueReadBuffer(gpu->queue, st_g, CL_TRUE, 0, sizeof(state), state, 0, NULL, NULL);

        int bx = (int)state[0], by = (int)state[1];
        int pad = (int)state[7];

        term_clear();
        /* Bricks */
        for (int y = 0; y < bh; y++)
            for (int x = 0; x < bw; x++)
                if (bricks[y * bw + x] > 0)
                    term_printf(y + 1, x + 1, bricks[y * bw + x] % 6 + 1, 0, "#");
        /* Paddle */
        for (int i = 0; i < pw; i++)
            term_printf(gh - 1, pad + i + 1, 5, 1, "=");
        /* Ball */
        if (by >= 0 && by < gh && bx >= 0 && bx < gw)
            term_printf(by + 1, bx + 1, 3, 1, "O");

        term_printf(0, 0, 6, 1, " BREAKOUT | Score: %d | Q=Quit ", (int)state[6]);
        term_printf(sh - 1, 0, 7, 0, " Left/Right=Move Paddle ");
        term_refresh();

        if (state[5] > 0.5f) {
            term_printf(gh/2, gw/2 - 4, 2, 1, " GAME OVER! ");
            term_printf(gh/2 + 1, gw/2 - 6, 4, 0, " Score: %d ", (int)state[6]);
            term_refresh();
            term_wait_key();
            break;
        }

        int remaining = 0;
        for (int i = 0; i < bw * bh; i++) if (bricks[i] > 0) remaining++;
        if (remaining == 0) {
            term_printf(gh/2, gw/2 - 3, 3, 1, " YOU WIN! ");
            term_refresh();
            term_wait_key();
            break;
        }

        platform_sleep_ms(30);
    }

    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseMemObject(bk_g); clReleaseMemObject(st_g);
    free(bricks);
    return 0;
}
