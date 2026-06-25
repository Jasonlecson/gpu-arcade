/*
 * Game of Life - Conway's cellular automaton, GPU-accelerated
 */

#include "src/common.h"

static const char *life_kernel_src =
"__kernel void life_step(\n"
"    __global const int *in, __global int *out, int W, int H)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x >= W || y >= H) return;\n"
"    int n = 0;\n"
"    for (int dy = -1; dy <= 1; dy++)\n"
"        for (int dx = -1; dx <= 1; dx++) {\n"
"            if (dx == 0 && dy == 0) continue;\n"
"            int nx = (x + dx + W) % W;\n"
"            int ny = (y + dy + H) % H;\n"
"            n += in[ny * W + nx] & 1;\n"
"        }\n"
"    int alive = in[y * W + x] & 1;\n"
"    out[y * W + x] = (alive && (n == 2 || n == 3)) || (!alive && n == 3) ? 1 : 0;\n"
"}\n";

int game_life(gpu_ctx_t *gpu) {
    int sw, sh;
    get_terminal_size(&sw, &sh);
    int gw = sw - 2, gh = sh - 4;
    if (gw < 10) gw = 10;
    if (gh < 5) gh = 5;

    int *grid_a = calloc(gw * gh, sizeof(int));
    int *grid_b = calloc(gw * gh, sizeof(int));

    /* Random initial state */
    for (int i = 0; i < gw * gh; i++)
        grid_a[i] = rand() % 4 == 0 ? 1 : 0;

    cl_int err;
    cl_mem ga = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gw*gh*sizeof(int), grid_a, &err);
    cl_mem gb = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE, gw*gh*sizeof(int), NULL, &err);
    cl_program prog = gpu_build(gpu, life_kernel_src);
    cl_kernel kern = clCreateKernel(prog, "life_step", &err);

    int gen = 0, paused = 0;
    int use_a = 1;

    while (1) {
        int key = read_key();
        if (key == 'q' || key == 'Q' || key == 27) break;
        if (key == ' ') paused = !paused;
        if (key == 'r' || key == 'R') {
            for (int i = 0; i < gw * gh; i++) grid_a[i] = rand() % 4 == 0 ? 1 : 0;
            clEnqueueWriteBuffer(gpu->queue, ga, CL_TRUE, 0, gw*gh*sizeof(int), grid_a, 0, NULL, NULL);
            gen = 0; use_a = 1;
        }

        if (!paused) {
            cl_mem src = use_a ? ga : gb;
            cl_mem dst = use_a ? gb : ga;
            size_t gws[2] = {gw, gh};
            clSetKernelArg(kern, 0, sizeof(cl_mem), &src);
            clSetKernelArg(kern, 1, sizeof(cl_mem), &dst);
            clSetKernelArg(kern, 2, sizeof(int), &gw);
            clSetKernelArg(kern, 3, sizeof(int), &gh);
            clEnqueueNDRangeKernel(gpu->queue, kern, 2, NULL, gws, NULL, 0, NULL, NULL);
            clFinish(gpu->queue);

            int *read_buf = use_a ? grid_b : grid_a;
            clEnqueueReadBuffer(gpu->queue, dst, CL_TRUE, 0, gw*gh*sizeof(int), read_buf, 0, NULL, NULL);
            use_a = !use_a;
            gen++;
        }

        int *display = use_a ? grid_a : grid_b;
        if (!paused) {
            cl_mem src = use_a ? gb : ga;
            clEnqueueReadBuffer(gpu->queue, src, CL_TRUE, 0, gw*gh*sizeof(int), display, 0, NULL, NULL);
        }

        term_clear();
        for (int y = 0; y < gh; y++)
            for (int x = 0; x < gw; x++)
                term_printf(y + 1, x + 1, display[y * gw + x] ? 1 : 7, display[y * gw + x] ? 1 : 0,
                            display[y * gw + x] ? "#" : " ");

        term_printf(0, 0, 6, 1, " GAME OF LIFE | Gen: %d | %s | Q=Quit R=Reset Space=Pause ",
                    gen, paused ? "PAUSED" : "RUNNING");
        term_printf(gh + 2, 0, 5, 0, " Grid: %dx%d | %d threads on GPU ", gw, gh, gw * gh);
        term_refresh();

        platform_sleep_ms(100);
    }

    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseMemObject(ga); clReleaseMemObject(gb);
    free(grid_a); free(grid_b);
    return 0;
}
