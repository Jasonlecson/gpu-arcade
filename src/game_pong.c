/*
 * Pong - Classic paddle ball game, physics on GPU
 */

#include "src/common.h"

static const char *pong_kernel_src =
"__kernel void pong_update(\n"
"    __global float *state, int W, int H)\n"
"{\n"
"    if (get_global_id(0) != 0) return;\n"
"    float bx = state[0], by = state[1];\n"
"    float dx = state[2], dy = state[3];\n"
"    float p1 = state[4], p2 = state[5];\n"
"    int s1 = (int)state[6], s2 = (int)state[7];\n"
"    int pw = (int)state[8];\n"
"    float spd = state[9];\n"
"    bx += dx * spd; by += dy * spd;\n"
"    if (by <= 0 || by >= H - 1) dy = -dy;\n"
"    if (bx <= 1 && by >= p1 && by <= p1 + pw) { dx = -dx; bx = 2; }\n"
"    if (bx >= W - 2 && by >= p2 && by <= p2 + pw) { dx = -dx; bx = W - 3; }\n"
"    if (bx < 0) { s2++; bx = W/2; by = H/2; dx = 1; }\n"
"    if (bx > W) { s1++; bx = W/2; by = H/2; dx = -1; }\n"
"    state[0]=bx; state[1]=by; state[2]=dx; state[3]=dy;\n"
"    state[4]=p1; state[5]=p2; state[6]=(float)s1; state[7]=(float)s2;\n"
"}\n";

int game_pong(gpu_ctx_t *gpu) {
    int sw, sh;
    get_terminal_size(&sw, &sh);
    int gw = sw - 2, gh = sh - 4;
    if (gw < 20) gw = 20;
    if (gh < 10) gh = 10;
    int pw = gh / 4;

    float state[10] = {gw/2.0f, gh/2.0f, 1.0f, 1.0f, gh/2.0f-pw/2.0f, gh/2.0f-pw/2.0f, 0, 0, (float)pw, 0.5f};

    cl_int err;
    cl_mem st_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(state), state, &err);
    cl_program prog = gpu_build(gpu, pong_kernel_src);
    cl_kernel kern = clCreateKernel(prog, "pong_update", &err);

    while (1) {
        int key = read_key();
        if (key == 'q' || key == 'Q' || key == 27) break;
        if (key == KEY_UP_ && state[4] > 0) state[4] -= 1.5f;
        if (key == KEY_DOWN_ && state[4] < gh - pw) state[4] += 1.5f;

        /* Simple AI for player 2 */
        float target = state[1] - pw / 2.0f;
        if (state[5] < target && state[5] < gh - pw) state[5] += 0.8f;
        if (state[5] > target && state[5] > 0) state[5] -= 0.8f;

        clEnqueueWriteBuffer(gpu->queue, st_g, CL_TRUE, 0, sizeof(state), state, 0, NULL, NULL);
        size_t gws = 1;
        clSetKernelArg(kern, 0, sizeof(cl_mem), &st_g);
        clSetKernelArg(kern, 1, sizeof(int), &gw);
        clSetKernelArg(kern, 2, sizeof(int), &gh);
        clEnqueueNDRangeKernel(gpu->queue, kern, 1, NULL, &gws, NULL, 0, NULL, NULL);
        clFinish(gpu->queue);
        clEnqueueReadBuffer(gpu->queue, st_g, CL_TRUE, 0, sizeof(state), state, 0, NULL, NULL);

        int bx = (int)state[0], by = (int)state[1];
        int p1 = (int)state[4], p2 = (int)state[5];

        term_clear();
        /* Court */
        for (int y = 0; y < gh; y++) {
            term_printf(y + 1, 0, 7, 0, "|");
            term_printf(y + 1, gw + 1, 7, 0, "|");
            if (y % 3 == 0) term_printf(y + 1, gw / 2 + 1, 7, 0, ":");
        }
        /* Paddles */
        for (int i = 0; i < pw; i++) {
            if (p1 + i >= 0 && p1 + i < gh) term_printf(p1 + i + 1, 1, 5, 1, "#");
            if (p2 + i >= 0 && p2 + i < gh) term_printf(p2 + i + 1, gw, 5, 1, "#");
        }
        /* Ball */
        if (by >= 0 && by < gh && bx >= 0 && bx < gw)
            term_printf(by + 1, bx + 1, 3, 1, "O");

        term_printf(0, 0, 6, 1, " PONG | %d : %d | You=Left  AI=Right | Q=Quit ", (int)state[6], (int)state[7]);
        term_printf(sh - 1, 0, 7, 0, " Up/Down=Move Paddle ");
        term_refresh();

        platform_sleep_ms(30);

        if ((int)state[6] >= 11 || (int)state[7] >= 11) {
            int winner = (int)state[6] >= 11 ? 1 : 2;
            term_printf(gh/2, gw/2 - 5, 3, 1, " PLAYER %d WINS! ", winner);
            term_refresh();
            term_wait_key();
            break;
        }
    }

    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseMemObject(st_g);
    return 0;
}
