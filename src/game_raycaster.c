/*
 * Raycaster - Wolfenstein 3D-style raycasting, every column on GPU
 */

#include "src/common.h"
#include <math.h>

static const char *ray_kernel_src =
"__kernel void raycast(\n"
"    __global int *screen, __global const int *map,\n"
"    float px, float py, float pa, int W, int H, int MW, int MH)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    if (x >= W) return;\n"
"    float fov = 3.14159f / 3.0f;\n"
"    float angle = pa - fov / 2.0f + (float)x / W * fov;\n"
"    float dx = cos(angle), dy = sin(angle);\n"
"    float dist = 0;\n"
"    int hit = 0;\n"
"    for (int i = 0; i < 60; i++) {\n"
"        dist += 0.1f;\n"
"        int mx = (int)(px + dx * dist);\n"
"        int my = (int)(py + dy * dist);\n"
"        if (mx < 0 || mx >= MW || my < 0 || my >= MH) break;\n"
"        if (map[my * MW + mx] > 0) { hit = map[my * MW + mx]; break; }\n"
"    }\n"
"    if (dist < 0.01f) dist = 0.01f;\n"
"    int wall_h = (int)(H / dist);\n"
"    if (wall_h > H) wall_h = H;\n"
"    int wall_top = (H - wall_h) / 2;\n"
"    int wall_bot = wall_top + wall_h;\n"
"    for (int y = 0; y < H; y++) {\n"
"        int idx = y * W + x;\n"
"        if (y < wall_top) {\n"
"            screen[idx] = 0;\n"
"        } else if (y < wall_bot) {\n"
"            float shade = 1.0f - dist / 6.0f;\n"
"            if (shade < 0.2f) shade = 0.2f;\n"
"            int c = hit > 0 ? hit : 4;\n"
"            screen[idx] = c * 10 + (int)(shade * 9);\n"
"        } else {\n"
"            float shade = 0.3f + (float)(y - H/2) / H;\n"
"            screen[idx] = 70 + (int)(shade * 9);\n"
"        }\n"
"    }\n"
"}\n";

int game_raycaster(gpu_ctx_t *gpu) {
    int sw, sh;
    get_terminal_size(&sw, &sh);
    int gw = sw - 2, gh = sh - 4;
    if (gw < 20) gw = 20;
    if (gh < 10) gh = 10;

    int mw = 16, mh = 16;
    int map[256] = {
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
        1,0,0,2,2,0,0,0,0,0,3,3,3,0,0,1,
        1,0,0,2,0,0,0,0,0,0,0,0,3,0,0,1,
        1,0,0,2,0,0,0,0,0,0,0,0,3,0,0,1,
        1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,4,4,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,4,4,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
        1,0,3,0,0,0,0,0,0,0,0,0,2,0,0,1,
        1,0,3,0,0,0,0,0,0,0,0,0,2,0,0,1,
        1,0,3,3,0,0,0,0,0,0,2,2,2,0,0,1,
        1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    };

    float px = 8.0f, py = 8.0f, pa = 0.0f;
    int *screen = malloc(gw * gh * sizeof(int));

    cl_int err;
    cl_mem scr_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE, gw*gh*sizeof(int), NULL, &err);
    cl_mem map_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 256*sizeof(int), map, &err);
    cl_program prog = gpu_build(gpu, ray_kernel_src);
    cl_kernel kern = clCreateKernel(prog, "raycast", &err);

    while (1) {
        int key = read_key();
        if (key == 'q' || key == 'Q' || key == 27) break;

        float move_spd = 0.2f, rot_spd = 0.1f;
        if (key == KEY_UP_) { px += cos(pa)*move_spd; py += sin(pa)*move_spd; }
        if (key == KEY_DOWN_) { px -= cos(pa)*move_spd; py -= sin(pa)*move_spd; }
        if (key == KEY_LEFT_) pa -= rot_spd;
        if (key == KEY_RIGHT_) pa += rot_spd;

        /* Collision */
        if (map[(int)py * mw + (int)px] > 0) {
            px -= cos(pa)*move_spd; py -= sin(pa)*move_spd;
        }

        size_t gws = gw;
        int iw = gw, ih = gh, imw = mw, imh = mh;
        clSetKernelArg(kern, 0, sizeof(cl_mem), &scr_g);
        clSetKernelArg(kern, 1, sizeof(cl_mem), &map_g);
        clSetKernelArg(kern, 2, sizeof(float), &px);
        clSetKernelArg(kern, 3, sizeof(float), &py);
        clSetKernelArg(kern, 4, sizeof(float), &pa);
        clSetKernelArg(kern, 5, sizeof(int), &iw);
        clSetKernelArg(kern, 6, sizeof(int), &ih);
        clSetKernelArg(kern, 7, sizeof(int), &imw);
        clSetKernelArg(kern, 8, sizeof(int), &imh);
        clEnqueueNDRangeKernel(gpu->queue, kern, 1, NULL, &gws, NULL, 0, NULL, NULL);
        clFinish(gpu->queue);
        clEnqueueReadBuffer(gpu->queue, scr_g, CL_TRUE, 0, gw*gh*sizeof(int), screen, 0, NULL, NULL);

        term_clear();
        for (int y = 0; y < gh; y++)
            for (int x = 0; x < gw; x++) {
                int v = screen[y*gw+x];
                int color = v / 10;
                int bright = v % 10;
                if (color == 0) {
                    term_printf(y+1, x+1, 7, 0, " ");
                } else if (color >= 7) {
                    int ci = (bright > 5) ? 7 : 4;
                    term_printf(y+1, x+1, ci, 0, ".");
                } else {
                    char ch = (bright > 6) ? '#' : (bright > 3) ? '=' : '.';
                    int bold = (bright > 6) ? 1 : 0;
                    term_printf(y+1, x+1, color, bold, "%c", ch);
                }
            }

        term_printf(0, 0, 6, 1, " RAYCASTER | Pos(%.1f,%.1f) Angle:%.0f | %dx%d GPU ",
                    px, py, pa*180/3.14159f, gw, gh);
        term_printf(gh+2, 0, 7, 0, " Arrows=Move/Turn  Q=Quit ");
        term_refresh();

        platform_sleep_ms(30);
    }

    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseMemObject(scr_g); clReleaseMemObject(map_g);
    free(screen);
    return 0;
}
