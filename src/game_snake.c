/*
 * Snake - Classic snake game, all logic on GPU
 * Logic: 80ms per step | Render: 60 FPS
 */

#include "src/common.h"

#define SNAKE_FOOD 5
#define SNAKE_LOGIC_MS 80

static const char *snake_kernel_src =
"__kernel void snake_update(\n"
"    __global int *grid, __global int *head, __global int *dir,\n"
"    __global int *len, __global int *status, __global int *rng,\n"
"    int W, int H)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int total = W * H;\n"
"    if (id >= total) return;\n"
"    int frame = len[0] + 1;\n"
"    if (grid[id] > 0) { grid[id]++; if (grid[id] > frame) grid[id] = 0; }\n"
"    barrier(CLK_GLOBAL_MEM_FENCE);\n"
"    if (id == 0) {\n"
"        int hy = head[0], hx = head[1];\n"
"        int d = dir[0];\n"
"        int ny = hy + (d == 1 ? 1 : (d == 0 ? -1 : 0));\n"
"        int nx = hx + (d == 3 ? 1 : (d == 2 ? -1 : 0));\n"
"        status[0] = 0;\n"
"        if (ny < 0 || ny >= H || nx < 0 || nx >= W) { status[0] = 2; return; }\n"
"        int target = ny * W + nx;\n"
"        int val = grid[target];\n"
"        if (val > 0) { status[0] = 2; return; }\n"
"        if (val == -1) { status[0] = 1; len[0]++; }\n"
"        grid[target] = 1;\n"
"        head[0] = ny; head[1] = nx;\n"
"    }\n"
"    barrier(CLK_GLOBAL_MEM_FENCE);\n"
"    if (id == 0 && status[0] == 1) {\n"
"        unsigned int seed = rng[0];\n"
"        int ec = 0;\n"
"        for (int i = 0; i < total; i++) if (grid[i] == 0) ec++;\n"
"        if (ec > 0) {\n"
"            seed = seed * 1103515245u + 12345u;\n"
"            int pick = seed % ec;\n"
"            int cnt = 0;\n"
"            for (int i = 0; i < total; i++) {\n"
"                if (grid[i] == 0) { if (cnt == pick) { grid[i] = -1; break; } cnt++; }\n"
"            }\n"
"            rng[0] = (int)seed;\n"
"        }\n"
"    }\n"
"}\n";

int game_snake(gpu_ctx_t *gpu) {
    FILE *dbg = fopen("gpu_arcade_debug.log", "w"); if(dbg){fprintf(dbg,"1: entered game_snake\n");fflush(dbg);}
    int sw, sh;
    get_terminal_size(&sw, &sh);
    int gw = sw - 2, gh = sh - 6;
    if (gw < 10) gw = 10;
    if (gh < 5) gh = 5;
    if(dbg){fprintf(dbg,"2: grid=%dx%d\n",gw,gh);fflush(dbg);}

    int *grid = calloc(gw * gh, sizeof(int));
    int sy = gh / 2, sx = gw / 2;
    for (int i = 0; i < 4 && sx - i >= 0; i++)
        grid[sy * gw + (sx - i)] = i + 1;

    int head[2] = {sy, sx}, dir[1] = {3}, len[1] = {4}, status[1] = {0};
    int rng[1] = {(int)(time(NULL) * 1000) & 0x7FFFFFFF};
    for (int f = 0; f < SNAKE_FOOD; f++) {
        int fy, fx;
        do { fy = rand() % gh; fx = rand() % gw; } while (grid[fy * gw + fx] != 0);
        grid[fy * gw + fx] = -1;
    }

    if(dbg){fprintf(dbg,"3: creating buffers\n");fflush(dbg);}
    cl_int err;
    cl_mem grid_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gw*gh*sizeof(int), grid, &err);
    if(dbg){fprintf(dbg,"4: grid_g err=%d\n",err);fflush(dbg);}
    cl_mem head_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 2*sizeof(int), head, &err);
    cl_mem dir_g  = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), dir, &err);
    cl_mem len_g  = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), len, &err);
    cl_mem st_g   = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), status, &err);
    cl_mem rng_g  = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), rng, &err);

    if(dbg){fprintf(dbg,"5: building kernel\n");fflush(dbg);}
    cl_program prog = gpu_build(gpu, snake_kernel_src);
    if(dbg){fprintf(dbg,"6: kernel built OK\n");fflush(dbg);}
    cl_kernel kern = clCreateKernel(prog, "snake_update", &err);
    if(dbg){fprintf(dbg,"7: kernel created err=%d\n",err);fflush(dbg);}

    int fc = 0;
    double sess = now_us();
    double next_logic = now_us();
    if(dbg){fprintf(dbg,"8: entering loop\n");fflush(dbg);}

    while (1) {
        int key = read_key();
        if (key == 'q' || key == 'Q' || key == 27) { if(dbg)fprintf(dbg,"quit\n"); break; }
        if (key == KEY_UP_ && dir[0] != 1) dir[0] = 0;
        else if (key == KEY_DOWN_ && dir[0] != 0) dir[0] = 1;
        else if (key == KEY_LEFT_ && dir[0] != 3) dir[0] = 2;
        else if (key == KEY_RIGHT_ && dir[0] != 2) dir[0] = 3;

        double now = now_us();
        if (now >= next_logic && status[0] != 2) {
            next_logic = now + SNAKE_LOGIC_MS * 1000.0;
            if(dbg && fc==0){fprintf(dbg,"9: first logic\n");fflush(dbg);}

            clEnqueueWriteBuffer(gpu->queue, dir_g, CL_TRUE, 0, sizeof(int), dir, 0, NULL, NULL);
            size_t gws = gw * gh;
            clSetKernelArg(kern, 0, sizeof(cl_mem), &grid_g);
            clSetKernelArg(kern, 1, sizeof(cl_mem), &head_g);
            clSetKernelArg(kern, 2, sizeof(cl_mem), &dir_g);
            clSetKernelArg(kern, 3, sizeof(cl_mem), &len_g);
            clSetKernelArg(kern, 4, sizeof(cl_mem), &st_g);
            clSetKernelArg(kern, 5, sizeof(cl_mem), &rng_g);
            clSetKernelArg(kern, 6, sizeof(int), &gw);
            clSetKernelArg(kern, 7, sizeof(int), &gh);
            cl_event ev;
            clEnqueueNDRangeKernel(gpu->queue, kern, 1, NULL, &gws, NULL, 0, NULL, &ev);
            clFinish(gpu->queue);

            clEnqueueReadBuffer(gpu->queue, head_g, CL_TRUE, 0, 2*sizeof(int), head, 0, NULL, NULL);
            clEnqueueReadBuffer(gpu->queue, st_g, CL_TRUE, 0, sizeof(int), status, 0, NULL, NULL);
            clEnqueueReadBuffer(gpu->queue, len_g, CL_TRUE, 0, sizeof(int), len, 0, NULL, NULL);
            clEnqueueReadBuffer(gpu->queue, grid_g, CL_TRUE, 0, gw*gh*sizeof(int), grid, 0, NULL, NULL);
            clReleaseEvent(ev);
            fc++;
        }

        double elapsed = (now_us() - sess) / 1e6;

        term_clear();
        for (int y = 0; y < gh; y++) {
            for (int x = 0; x < gw; x++) {
                int v = grid[y * gw + x];
                if (y == head[0] && x == head[1])
                    term_printf(y + 1, x + 1, 3, 1, "@");
                else if (v == -1)
                    term_printf(y + 1, x + 1, 2, 1, "*");
                else if (v > 0)
                    term_printf(y + 1, x + 1, 1, 0, "o");
                else
                    term_printf(y + 1, x + 1, 7, 0, ".");
            }
        }
        term_printf(0, 0, 6, 1, " SNAKE | Score:%d Length:%d | Q=Quit ", len[0]-4, len[0]);
        term_printf(gh + 2, 0, 7, 0, " Arrows=Move  Q=Back to Menu | %d FPS ", fc > 0 ? (int)(fc / elapsed) : 0);
        term_refresh();
        if(dbg && fc==0){fprintf(dbg,"10: first refresh done\n");fflush(dbg);}

        if (status[0] == 2) {
            term_printf(gh/2, gw/2 - 4, 2, 1, " GAME OVER! ");
            term_printf(gh/2 + 1, gw/2 - 7, 4, 0, " Press any key ");
            term_refresh();
            term_wait_key();
            break;
        }

        platform_sleep_ms(16);
    }

    if(dbg){fprintf(dbg,"11: exiting loop\n");fclose(dbg);}
    clReleaseKernel(kern);
    clReleaseProgram(prog);
    clReleaseMemObject(grid_g); clReleaseMemObject(head_g);
    clReleaseMemObject(dir_g); clReleaseMemObject(len_g);
    clReleaseMemObject(st_g); clReleaseMemObject(rng_g);
    free(grid);
    return 0;
}
