/*
 * Falling Sand - Particle simulation (sand/water/fire), physics on GPU
 */

#include "src/common.h"

#define SAND 1
#define WATER 2
#define STONE 3
#define FIRE 4

static const char *sand_kernel_src =
"__kernel void sand_step(\n"
"    __global const int *in, __global int *out,\n"
"    __global int *rng, int W, int H)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x >= W || y >= H) return;\n"
"    int id = y * W + x;\n"
"    int v = in[id];\n"
"    out[id] = v;\n"
"    if (v == 0 || v == 3) return;\n"
"    int seed = rng[id];\n"
"    int r = (seed * 1103515245 + 12345) & 0x7fffffff;\n"
"    rng[id] = r;\n"
"    int rand_dir = (r & 1) ? 1 : -1;\n"
"    if (v == 1) {\n"
"        if (y + 1 < H && in[(y+1)*W+x] == 0) { out[id] = 0; out[(y+1)*W+x] = 1; return; }\n"
"        int d = rand_dir;\n"
"        if (y+1 < H && x+d >= 0 && x+d < W && in[(y+1)*W+x+d] == 0) { out[id] = 0; out[(y+1)*W+x+d] = 1; return; }\n"
"        d = -d;\n"
"        if (y+1 < H && x+d >= 0 && x+d < W && in[(y+1)*W+x+d] == 0) { out[id] = 0; out[(y+1)*W+x+d] = 1; return; }\n"
"    } else if (v == 2) {\n"
"        if (y+1 < H && in[(y+1)*W+x] == 0) { out[id] = 0; out[(y+1)*W+x] = 2; return; }\n"
"        int d = rand_dir;\n"
"        if (y+1 < H && x+d >= 0 && x+d < W && in[(y+1)*W+x+d] == 0) { out[id] = 0; out[(y+1)*W+x+d] = 2; return; }\n"
"        if (x+d >= 0 && x+d < W && in[y*W+x+d] == 0) { out[id] = 0; out[y*W+x+d] = 2; return; }\n"
"        d = -d;\n"
"        if (x+d >= 0 && x+d < W && in[y*W+x+d] == 0) { out[id] = 0; out[y*W+x+d] = 2; return; }\n"
"    } else if (v == 4) {\n"
"        if (y-1 >= 0 && in[(y-1)*W+x] == 0) { out[id] = 0; out[(y-1)*W+x] = 4; return; }\n"
"        int d = rand_dir;\n"
"        if (x+d >= 0 && x+d < W && in[y*W+x+d] == 0) { out[id] = 0; out[y*W+x+d] = 4; return; }\n"
"        if (r % 100 < 5) { out[id] = 0; return; }\n"
"    }\n"
"}\n";

int game_sand(gpu_ctx_t *gpu) {
    int sw, sh;
    get_terminal_size(&sw, &sh);
    int gw = sw - 2, gh = sh - 4;
    if (gw < 10) gw = 10;
    if (gh < 5) gh = 5;

    int *grid_a = calloc(gw * gh, sizeof(int));
    int *grid_b = calloc(gw * gh, sizeof(int));
    int *rng = malloc(gw * gh * sizeof(int));
    for (int i = 0; i < gw * gh; i++) rng[i] = rand();

    /* Add some stone platforms */
    for (int x = gw/4; x < gw*3/4; x++) {
        grid_a[gh*2/3 * gw + x] = STONE;
        grid_a[gh/3 * gw + x] = STONE;
    }

    int brush = SAND, use_a = 1, dirty = 1;

    cl_int err;
    cl_mem ga = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gw*gh*sizeof(int), grid_a, &err);
    cl_mem gb = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE, gw*gh*sizeof(int), NULL, &err);
    cl_mem rg = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gw*gh*sizeof(int), rng, &err);
    cl_program prog = gpu_build(gpu, sand_kernel_src);
    cl_kernel kern = clCreateKernel(prog, "sand_step", &err);

    while (1) {
        int key = read_key();
        if (key == 'q' || key == 'Q' || key == 27) break;
        if (key == '1') brush = SAND;
        if (key == '2') brush = WATER;
        if (key == '3') brush = STONE;
        if (key == '4') brush = FIRE;
        if (key == 'c' || key == 'C') {
            memset(grid_a, 0, gw*gh*sizeof(int));
            clEnqueueWriteBuffer(gpu->queue, ga, CL_TRUE, 0, gw*gh*sizeof(int), grid_a, 0, NULL, NULL);
            dirty = 1;
        }

        /* Place particles with keyboard (simulate mouse with keys) */
        if (key == ' ') {
            int cx = gw/2, cy = gh/2;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2; dx <= 2; dx++) {
                    int px = cx+dx, py = cy+dy;
                    if (px >= 0 && px < gw && py >= 0 && py < gh) {
                        if (brush == STONE || grid_a[py*gw+px] == 0)
                            grid_a[py*gw+px] = brush;
                    }
                }
            clEnqueueWriteBuffer(gpu->queue, ga, CL_TRUE, 0, gw*gh*sizeof(int), grid_a, 0, NULL, NULL);
            dirty = 1;
        }

        /* Simulate */
        cl_mem src = use_a ? ga : gb;
        cl_mem dst = use_a ? gb : ga;
        size_t gws[2] = {gw, gh};
        clSetKernelArg(kern, 0, sizeof(cl_mem), &src);
        clSetKernelArg(kern, 1, sizeof(cl_mem), &dst);
        clSetKernelArg(kern, 2, sizeof(cl_mem), &rg);
        clSetKernelArg(kern, 3, sizeof(int), &gw);
        clSetKernelArg(kern, 4, sizeof(int), &gh);
        clEnqueueNDRangeKernel(gpu->queue, kern, 2, NULL, gws, NULL, 0, NULL, NULL);
        clFinish(gpu->queue);

        int *display = use_a ? grid_b : grid_a;
        clEnqueueReadBuffer(gpu->queue, dst, CL_TRUE, 0, gw*gh*sizeof(int), display, 0, NULL, NULL);
        use_a = !use_a;
        dirty = 1;

        if (dirty) {
            dirty = 0;
            term_clear();
            for (int y = 0; y < gh; y++)
                for (int x = 0; x < gw; x++) {
                    int v = display[y*gw+x];
                    switch (v) {
                        case 0: term_printf(y+1, x+1, 7, 0, " "); break;
                        case SAND:  term_printf(y+1, x+1, 3, 0, "."); break;
                        case WATER: term_printf(y+1, x+1, 5, 0, "~"); break;
                        case STONE: term_printf(y+1, x+1, 4, 0, "#"); break;
                        case FIRE:  term_printf(y+1, x+1, 2, 1, "*"); break;
                    }
                }

            const char *brush_name = "Sand";
            if (brush == WATER) brush_name = "Water";
            if (brush == STONE) brush_name = "Stone";
            if (brush == FIRE) brush_name = "Fire";
            term_printf(0, 0, 6, 1, " FALLING SAND | Brush: %s | 1-4=Select C=Clear Q=Quit ", brush_name);
            term_printf(gh+2, 0, 7, 0, " 1=Sand 2=Water 3=Stone 4=Fire  Space=Place  C=Clear ");
            term_refresh();
        }

        platform_sleep_ms(50);
    }

    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseMemObject(ga); clReleaseMemObject(gb); clReleaseMemObject(rg);
    free(grid_a); free(grid_b); free(rng);
    return 0;
}
