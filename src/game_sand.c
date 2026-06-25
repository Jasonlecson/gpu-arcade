/*
 * Falling Sand - Particle simulation, physics on GPU
 * Logic: 30ms per step | Render: 60 FPS
 * Materials: Sand, Water, Stone, Fire, Lava, Acid, Smoke
 */

#include "src/common.h"

#define SAND    1
#define WATER   2
#define STONE   3
#define FIRE    4
#define BORDER  5
#define LAVA    6
#define ACID    7
#define SMOKE   8
#define SAND_LOGIC_MS 30

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
"    if (v == 0 || v == 3 || v == 5) return;\n"
"    int seed = rng[id];\n"
"    int r = (seed * 1103515245 + 12345) & 0x7fffffff;\n"
"    rng[id] = r;\n"
"    int rd = (r & 1) ? 1 : -1;\n"
"    int d;\n"
"    if (v == 1) {\n"
"        if (y+1 < H && in[(y+1)*W+x] == 0) { out[id] = 0; out[(y+1)*W+x] = 1; return; }\n"
"        d = rd;\n"
"        if (y+1 < H && x+d >= 0 && x+d < W && in[(y+1)*W+x+d] == 0) { out[id] = 0; out[(y+1)*W+x+d] = 1; return; }\n"
"        d = -d;\n"
"        if (y+1 < H && x+d >= 0 && x+d < W && in[(y+1)*W+x+d] == 0) { out[id] = 0; out[(y+1)*W+x+d] = 1; return; }\n"
"    } else if (v == 2) {\n"
"        if (y+1 < H && in[(y+1)*W+x] == 0) { out[id] = 0; out[(y+1)*W+x] = 2; return; }\n"
"        d = rd;\n"
"        if (y+1 < H && x+d >= 0 && x+d < W && in[(y+1)*W+x+d] == 0) { out[id] = 0; out[(y+1)*W+x+d] = 2; return; }\n"
"        if (x+d >= 0 && x+d < W && in[y*W+x+d] == 0) { out[id] = 0; out[y*W+x+d] = 2; return; }\n"
"        d = -d;\n"
"        if (x+d >= 0 && x+d < W && in[y*W+x+d] == 0) { out[id] = 0; out[y*W+x+d] = 2; return; }\n"
"    } else if (v == 4) {\n"
"        if (y-1 >= 0 && in[(y-1)*W+x] == 0) { out[id] = 0; out[(y-1)*W+x] = 4; return; }\n"
"        d = rd;\n"
"        if (x+d >= 0 && x+d < W && in[y*W+x+d] == 0) { out[id] = 0; out[y*W+x+d] = 4; return; }\n"
"        if (r % 100 < 15) { out[id] = 0; return; }\n"
"        if (y == 0) { out[id] = 0; return; }\n"
"    } else if (v == 6) {\n"
"        if (y+1 < H && (in[(y+1)*W+x] == 0 || in[(y+1)*W+x] == 2)) {\n"
"            if (in[(y+1)*W+x] == 2) { out[(y+1)*W+x] = 3; out[id] = 0; return; }\n"
"            out[id] = 0; out[(y+1)*W+x] = 6; return;\n"
"        }\n"
"        d = rd;\n"
"        if (y+1 < H && x+d >= 0 && x+d < W && (in[(y+1)*W+x+d] == 0 || in[(y+1)*W+x+d] == 2)) {\n"
"            if (in[(y+1)*W+x+d] == 2) { out[(y+1)*W+x+d] = 3; out[id] = 0; return; }\n"
"            out[id] = 0; out[(y+1)*W+x+d] = 6; return;\n"
"        }\n"
"        d = -d;\n"
"        if (y+1 < H && x+d >= 0 && x+d < W && (in[(y+1)*W+x+d] == 0 || in[(y+1)*W+x+d] == 2)) {\n"
"            if (in[(y+1)*W+x+d] == 2) { out[(y+1)*W+x+d] = 3; out[id] = 0; return; }\n"
"            out[id] = 0; out[(y+1)*W+x+d] = 6; return;\n"
"        }\n"
"    } else if (v == 7) {\n"
"        if (y+1 < H) {\n"
"            int below = in[(y+1)*W+x];\n"
"            if (below == 1 || below == 8) { out[(y+1)*W+x] = 0; out[id] = 0; return; }\n"
"            if (below == 0) { out[id] = 0; out[(y+1)*W+x] = 7; return; }\n"
"        }\n"
"        d = rd;\n"
"        if (x+d >= 0 && x+d < W) {\n"
"            int adj = in[y*W+x+d];\n"
"            if (adj == 1 || adj == 8) { out[y*W+x+d] = 0; out[id] = 0; return; }\n"
"            if (adj == 0) { out[id] = 0; out[y*W+x+d] = 7; return; }\n"
"        }\n"
"        d = -d;\n"
"        if (x+d >= 0 && x+d < W) {\n"
"            int adj = in[y*W+x+d];\n"
"            if (adj == 1 || adj == 8) { out[y*W+x+d] = 0; out[id] = 0; return; }\n"
"            if (adj == 0) { out[id] = 0; out[y*W+x+d] = 7; return; }\n"
"        }\n"
"    } else if (v == 8) {\n"
"        if (y-1 >= 0 && in[(y-1)*W+x] == 0) { out[id] = 0; out[(y-1)*W+x] = 8; return; }\n"
"        d = rd;\n"
"        if (x+d >= 0 && x+d < W && in[y*W+x+d] == 0) { out[id] = 0; out[y*W+x+d] = 8; return; }\n"
"        if (r % 100 < 8) { out[id] = 0; return; }\n"
"        if (y == 0) { out[id] = 0; return; }\n"
"    }\n"
"}\n";

static const char *sand_materials[] = {"Sand", "Water", "Stone", "Fire", "Lava", "Acid", "Smoke"};
static const int sand_ids[] = {1, 2, 3, 4, 6, 7, 8};
#define NUM_MATERIALS 7

int game_sand(gpu_ctx_t *gpu) {
    int prev_w = 0, prev_h = 0;

restart: ;
    int sw, sh;
    get_terminal_size(&sw, &sh);
    prev_w = sw; prev_h = sh;
    int gw = sw - 2, gh = sh - 8;
    if (gw < 10) gw = 10;
    if (gh < 5) gh = 5;

    int *grid_a = calloc(gw * gh, sizeof(int));
    int *grid_b = calloc(gw * gh, sizeof(int));
    int *rng = malloc(gw * gh * sizeof(int));
    for (int i = 0; i < gw * gh; i++) rng[i] = rand();

    for (int x = 0; x < gw; x++) {
        grid_a[x] = BORDER;
        grid_a[(gh - 1) * gw + x] = BORDER;
    }
    for (int x = gw/4; x < gw*3/4; x++)
        grid_a[(gh*2/3) * gw + x] = STONE;
    for (int x = gw/5; x < gw*4/5; x++)
        grid_a[(gh/3) * gw + x] = STONE;

    int brush_idx = 0, use_a = 1;
    int cx = gw / 2, cy = gh / 2;

    cl_int err;
    cl_mem ga = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gw*gh*sizeof(int), grid_a, &err);
    cl_mem gb = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE, gw*gh*sizeof(int), NULL, &err);
    cl_mem rg = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gw*gh*sizeof(int), rng, &err);
    cl_program prog = gpu_build(gpu, sand_kernel_src);
    cl_kernel kern = clCreateKernel(prog, "sand_step", &err);

    int *display = grid_a;
    double next_logic = now_us();
    int fc = 0;
    double sess = now_us();

    while (1) {
        term_size_t ts = check_resize(&prev_w, &prev_h);
        if (ts.changed) goto cleanup_restart;

        int key = read_key();
        if (key == 'q' || key == 'Q' || key == 27) break;
        if (key >= '1' && key <= '0' + NUM_MATERIALS) brush_idx = key - '1';

        if (key == 'c' || key == 'C') {
            for (int i = 0; i < gw * gh; i++)
                if (grid_a[i] != BORDER) grid_a[i] = 0;
            for (int i = 0; i < gw * gh; i++)
                if (grid_b[i] != BORDER) grid_b[i] = 0;
            clEnqueueWriteBuffer(gpu->queue, ga, CL_TRUE, 0, gw*gh*sizeof(int), grid_a, 0, NULL, NULL);
            clEnqueueWriteBuffer(gpu->queue, gb, CL_TRUE, 0, gw*gh*sizeof(int), grid_b, 0, NULL, NULL);
            display = use_a ? grid_a : grid_b;
        }

        if (key == KEY_UP_ && cy > 0) cy--;
        if (key == KEY_DOWN_ && cy < gh - 1) cy++;
        if (key == KEY_LEFT_ && cx > 0) cx--;
        if (key == KEY_RIGHT_ && cx < gw - 1) cx++;

        if (key == ' ') {
            int brush = sand_ids[brush_idx];
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2; dx <= 2; dx++) {
                    int px = cx+dx, py = cy+dy;
                    if (px >= 0 && px < gw && py >= 0 && py < gh) {
                        if (brush == STONE || brush == BORDER || grid_a[py*gw+px] == 0)
                            grid_a[py*gw+px] = brush;
                    }
                }
            clEnqueueWriteBuffer(gpu->queue, ga, CL_TRUE, 0, gw*gh*sizeof(int), grid_a, 0, NULL, NULL);
        }

        double now = now_us();
        if (now >= next_logic) {
            next_logic = now + SAND_LOGIC_MS * 1000.0;

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

            display = use_a ? grid_b : grid_a;
            clEnqueueReadBuffer(gpu->queue, dst, CL_TRUE, 0, gw*gh*sizeof(int), display, 0, NULL, NULL);
            use_a = !use_a;
        }

        term_clear();
        for (int y = 0; y < gh; y++)
            for (int x = 0; x < gw; x++) {
                int v = display[y*gw+x];
                switch (v) {
                    case 0: term_printf(y+1, x+1, 7, 0, " "); break;
                    case SAND:   term_printf(y+1, x+1, 3, 0, "."); break;
                    case WATER:  term_printf(y+1, x+1, 5, 0, "~"); break;
                    case STONE:  term_printf(y+1, x+1, 4, 0, "#"); break;
                    case FIRE:   term_printf(y+1, x+1, 2, 1, "*"); break;
                    case BORDER: term_printf(y+1, x+1, 4, 1, "="); break;
                    case LAVA:   term_printf(y+1, x+1, 1, 1, "@"); break;
                    case ACID:   term_printf(y+1, x+1, 2, 0, "&"); break;
                    case SMOKE:  term_printf(y+1, x+1, 7, 0, "^"); break;
                }
            }

        term_printf(0, 0, 6, 1, " FALLING SAND | [%d]%s | Cursor(%d,%d) | Q=Quit ",
                    brush_idx + 1, sand_materials[brush_idx], cx, cy);
        term_printf(gh+2, 0, 7, 0, " 1=Sand 2=Water 3=Stone 4=Fire 5=Lava 6=Acid 7=Smoke | Arrows=Move Space=Place C=Clear");

        if (cy >= 0 && cy < gh && cx >= 0 && cx < gw)
            term_printf(cy+1, cx+1, 3, 1, "+");

        draw_metrics(gpu, gh, fc++, sess, 0);
        term_refresh();

        platform_sleep_ms(16);
    }

    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseMemObject(ga); clReleaseMemObject(gb); clReleaseMemObject(rg);
    free(grid_a); free(grid_b); free(rng);
    return 0;

cleanup_restart:
    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseMemObject(ga); clReleaseMemObject(gb); clReleaseMemObject(rg);
    free(grid_a); free(grid_b); free(rng);
    goto restart;
}
