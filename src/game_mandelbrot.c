/*
 * Mandelbrot Explorer - Fractal rendering, every pixel computed on GPU
 */

#include "src/common.h"

static const char *mandel_kernel_src =
"__kernel void mandel_render(\n"
"    __global int *buf, int W, int H,\n"
"    double cx, double cy, double zoom, int max_iter)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x >= W || y >= H) return;\n"
"    double px = cx + (x - W / 2.0) / (zoom * W / 4.0);\n"
"    double py = cy + (y - H / 2.0) / (zoom * H / 4.0);\n"
"    double zx = 0, zy = 0;\n"
"    int iter = 0;\n"
"    while (zx * zx + zy * zy < 4.0 && iter < max_iter) {\n"
"        double tmp = zx * zx - zy * zy + px;\n"
"        zy = 2.0 * zx * zy + py;\n"
"        zx = tmp;\n"
"        iter++;\n"
"    }\n"
"    buf[y * W + x] = iter;\n"
"}\n";

int game_mandelbrot(gpu_ctx_t *gpu) {
    int prev_w = 0, prev_h = 0;

restart: ;
    int sw, sh;
    get_terminal_size(&sw, &sh);
    prev_w = sw; prev_h = sh;
    int gw = sw - 2, gh = sh - 6;
    if (gw < 20) gw = 20;
    if (gh < 10) gh = 10;

    double cx = -0.5, cy = 0.0, zoom = 1.0;
    int max_iter = 64;
    int *buf = malloc(gw * gh * sizeof(int));

    cl_int err;
    cl_mem buf_g = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE, gw*gh*sizeof(int), NULL, &err);
    cl_program prog = gpu_build(gpu, mandel_kernel_src);
    cl_kernel kern = clCreateKernel(prog, "mandel_render", &err);

    int dirty = 1;
    int fc = 0;
    double sess = now_us();

    while (1) {
        term_size_t ts = check_resize(&prev_w, &prev_h);
        if (ts.changed) goto cleanup_restart;
        int key = read_key();
        if (key == 'q' || key == 'Q' || key == 27) break;
        if (key == KEY_UP_) { cy -= 0.3 / zoom; dirty = 1; }
        if (key == KEY_DOWN_) { cy += 0.3 / zoom; dirty = 1; }
        if (key == KEY_LEFT_) { cx -= 0.3 / zoom; dirty = 1; }
        if (key == KEY_RIGHT_) { cx += 0.3 / zoom; dirty = 1; }
        if (key == '+' || key == '=') { zoom *= 1.5; dirty = 1; }
        if (key == '-' || key == '_') { zoom /= 1.5; dirty = 1; }
        if (key == 'i') { max_iter += 32; dirty = 1; }
        if (key == 'r' || key == 'R') { cx = -0.5; cy = 0; zoom = 1; max_iter = 64; dirty = 1; }

        if (!dirty) { platform_sleep_ms(30); continue; }
        dirty = 0;

        size_t gws[2] = {gw, gh};
        int iw = gw, ih = gh;
        clSetKernelArg(kern, 0, sizeof(cl_mem), &buf_g);
        clSetKernelArg(kern, 1, sizeof(int), &iw);
        clSetKernelArg(kern, 2, sizeof(int), &ih);
        clSetKernelArg(kern, 3, sizeof(double), &cx);
        clSetKernelArg(kern, 4, sizeof(double), &cy);
        clSetKernelArg(kern, 5, sizeof(double), &zoom);
        clSetKernelArg(kern, 6, sizeof(int), &max_iter);
        clEnqueueNDRangeKernel(gpu->queue, kern, 2, NULL, gws, NULL, 0, NULL, NULL);
        clFinish(gpu->queue);
        clEnqueueReadBuffer(gpu->queue, buf_g, CL_TRUE, 0, gw*gh*sizeof(int), buf, 0, NULL, NULL);

        term_clear();
        for (int y = 0; y < gh; y++)
            for (int x = 0; x < gw; x++) {
                int v = buf[y*gw+x];
                if (v >= max_iter)
                    term_printf(y + 1, x + 1, 0, 0, " ");
                else {
                    int ci = (v * 7 / max_iter) + 1;
                    if (ci > 7) ci = 7;
                    term_printf(y + 1, x + 1, ci, 1, "#");
                }
            }

        term_printf(0, 0, 6, 1, " MANDELBROT | Zoom: %.1fx | Iter: %d | (%.4f, %.4f) ",
                    zoom, max_iter, cx, cy);
        term_printf(gh + 2, 0, 7, 0, " Arrows=Pan +/-=Zoom I=Iter R=Reset Q=Quit | %dx%d GPU ", gw, gh);
        draw_metrics(gpu, gh, fc++, sess, 0);
        term_refresh();

        platform_sleep_ms(16);
    }

    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseMemObject(buf_g);
    free(buf);
    return 0;

cleanup_restart:
    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseMemObject(buf_g);
    free(buf);
    goto restart;
}
