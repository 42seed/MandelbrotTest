import pyray as ray
import array
import numba
import numpy
import sys
import os

os.add_dll_directory(f"{sys.prefix}/Library/bin")  # tbb dll

WIDTH, HEIGHT = 900, 900
RUN_TIME = 30  # sec
MAX_ITER = 50
SCALE = 1.0 / 300.0
OFFSET_X = -2.25
OFFSET_Y = -1.5
CHUNK_SIZE = 2


class Test:
    frame_count: int = 0
    avg_fps: float = 0.0
    start_time: float = 0.0
    elapsed_time: float = 1e-6


def draw_avg_fps(test: Test) -> None:

    if test.elapsed_time < RUN_TIME:
        test.frame_count += 1
        test.elapsed_time = ray.get_time() - test.start_time
        test.avg_fps = test.frame_count / test.elapsed_time

        ray.draw_text(f"FPS: {int(test.avg_fps)}", 10, 10, 30, ray.RAYWHITE)
    else:
        ray.draw_rectangle(0, 0, WIDTH, HEIGHT, ray.BLACK)
        ray.draw_text(f"Average FPS:", 180, 150, 80, ray.RAYWHITE)

        col = ray.RED if test.avg_fps < 30 else ray.ORANGE if test.avg_fps < 60 else ray.GREEN
        ray.draw_text(f"{test.avg_fps:.1f}", 370, 250, 80, col)


@numba.njit(fastmath=True, inline='always')
def mandelbrot(c_re, c_im):
    #
    c_norm = c_re ** 2 + c_im ** 2

    # skip calculation M1 (main cardioid)
    if 256.0 * c_norm ** 2 - 96.0 * c_norm + 32.0 * c_re < 3.0:
        return MAX_ITER

    # skip calculation M2 (main bulb)
    if 16.0 * c_norm + 32.0 * c_re < -15.0:
        return MAX_ITER

    # classic calculations
    z_re, z_im = 0.0, 0.0
    for i in range(MAX_ITER):

        z_re_sqr, z_im_sqr = z_re ** 2, z_im ** 2
        z_im = 2.0 * z_re * z_im + c_im
        z_re = z_re_sqr - z_im_sqr + c_re

        z_norm = z_re_sqr + z_im_sqr
        if z_norm > 16.0:
            return i - numpy.log2(numpy.log10(z_norm))  # smooth count
    return MAX_ITER


@numba.njit(parallel=True, fastmath=True, inline='always')
def fill_framebuffer(frame_buffer, time):

    for x in numba.prange(WIDTH):
        for y in range(HEIGHT):

            c_re = x * SCALE + OFFSET_X
            c_im = y * SCALE + OFFSET_Y

            res = mandelbrot(c_re, c_im)

            if res < MAX_ITER:
                idx = (y * WIDTH + x) * 4

                frame_buffer[idx + 0] = int(127.5 * numpy.cos(res * 0.3 + time) + 127.5)
                frame_buffer[idx + 1] = int(127.5 * numpy.cos(res * 0.7 + time) + 127.5)
                frame_buffer[idx + 2] = int(127.5 * numpy.cos(res * 0.9 + time) + 127.5)


def main():
    # tbb - Intel Threading Building Blocks
    numba.config.THREADING_LAYER = 'tbb'
    numba.set_parallel_chunksize(CHUNK_SIZE)

    ray.set_trace_log_level(ray.TraceLogLevel.LOG_NONE)
    ray.init_window(WIDTH, HEIGHT, "Mandelbrot Fractal")

    # framebuffer (image) on the CPU side
    frame_buffer = numpy.array([*ray.BLACK] * (WIDTH * HEIGHT), dtype=numpy.uint8)

    # framebuffer pointer
    frame_buffer_ptr = ray.ffi.cast("uint8_t*", frame_buffer.ctypes.data)

    # texture on the GPU side
    texture = ray.load_texture_from_image(ray.gen_image_color(WIDTH, HEIGHT, ray.BLACK))

    test = Test()
    test.start_time = ray.get_time()

    while not ray.window_should_close():
        fill_framebuffer(frame_buffer, ray.get_time())

        ray.update_texture(texture, frame_buffer_ptr)  # write framebuffer from CPU to GPU

        ray.begin_drawing()
        ray.draw_texture(texture, 0, 0, ray.WHITE)  # draw the texture on the GPU side
        draw_avg_fps(test)
        ray.draw_text("Python", 10, 45, 30, ray.GREEN)
        ray.end_drawing()

    ray.close_window()


if __name__ == "__main__":
    main()
