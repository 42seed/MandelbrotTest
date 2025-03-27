#include <iostream>
#include <vector>
#include "raylib.h"

constexpr int WIDTH {900}, HEIGHT {900};
constexpr double RUN_TIME {30.0};  // sec
constexpr int MAX_ITER {50};
constexpr double SCALE {1.0 / 300.0};
constexpr double OFFSET_X {-2.25};
constexpr double OFFSET_Y {-1.5};
constexpr int CHUNK_SIZE {2};


struct Test
{
    int frameCount {0};
    double avgFPS {0.0};
    double startTime {0.0};
    double elapsedTime {1e-6};
};


void DrawAvgFPS(Test& test)
{
    if (test.elapsedTime < RUN_TIME)
    {
        test.elapsedTime = GetTime() - test.startTime;
        test.avgFPS = ++test.frameCount / test.elapsedTime;
        DrawText(TextFormat("FPS: %.0f", test.avgFPS), 10, 10, 30, RAYWHITE);
    }
    else
    {
        DrawRectangle(0, 0, WIDTH, HEIGHT, BLACK);
        DrawText(TextFormat("Average FPS:"), 180, 150, 80, RAYWHITE);

        const auto col = test.avgFPS < 30.0 ? RED : test.avgFPS < 60.0 ? ORANGE : GREEN;
        DrawText(TextFormat("%.1f", test.avgFPS), 370, 250, 80, col);
    }

}


double mandelbrot(const double c_re, const double c_im)
{
    const double c_norm = c_re * c_re + c_im * c_im;

    // skip calculation M1 (main cardioid)
    if (256.0 * c_norm * c_norm - 96.0 * c_norm + 32.0 * c_re < 3.0)
        return MAX_ITER;

    // skip calculation M2 (main bulb)
    if (16.0 * c_norm + 32.0 * c_re < -15.0)
        return MAX_ITER;

    // classic calculations
    double z_re = 0, z_im = 0;
    for (int i = 0; i < MAX_ITER; ++i)
    {
        const double z_re_sqr = z_re * z_re, z_im_sqr = z_im * z_im;
        z_im = 2 * z_re * z_im + c_im;
        z_re = z_re_sqr - z_im_sqr + c_re;

        const double z_norm = z_re * z_re + z_im * z_im;
        if (z_norm > 16)
            return i - log2(log10(z_norm));  // smooth count
    }
    return MAX_ITER;
}


void fillFramebuffer(std::vector<Color>& frameBuffer, double time)
{
    // ----------------------------------------
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    // ----------------------------------------
    for (int x = 0; x < WIDTH; ++x)
    {
        for (int y = 0; y < HEIGHT; ++y)
        {
            const double c_re = x * SCALE + OFFSET_X;
            const double c_im = y * SCALE + OFFSET_Y;

            const double res = mandelbrot(c_re, c_im);
            if (res < MAX_ITER)
            {
                int idx = y * WIDTH + x;

                frameBuffer[idx].r = static_cast<uint8_t>(127.5 * cos(res * 0.3 + time) + 127.5);
                frameBuffer[idx].g = static_cast<uint8_t>(127.5 * cos(res * 0.7 + time) + 127.5);
                frameBuffer[idx].b = static_cast<uint8_t>(127.5 * cos(res * 0.9 + time) + 127.5);
            }
        }
    }
}


int main()
{
    SetTraceLogLevel(LOG_NONE);
    InitWindow(WIDTH, HEIGHT, "Mandelbrot Fractal");

    // framebuffer (image) on the CPU side
    auto frameBuffer = std::vector{WIDTH * HEIGHT, BLACK};

    // texture on the GPU side
    auto texture = LoadTextureFromImage(GenImageColor(WIDTH, HEIGHT, BLACK));

    Test test;
    test.startTime = GetTime();

    while (!WindowShouldClose())
    {
        fillFramebuffer(frameBuffer, GetTime());

        UpdateTexture(texture, frameBuffer.data());  // write framebuffer from CPU to GPU

        BeginDrawing();
        DrawTexture(texture, 0, 0, WHITE);  // draw the texture on the GPU side
        DrawAvgFPS(test);
        DrawText("C++", 10, 45, 30, GREEN);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
