#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "STBImage.h"
#include "meanshift.h"

static void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <image> [bandwidth] [max_iter] [brute|grid] [--pbar]" << std::endl;
    std::cerr << "  bandwidth : float, default 150" << std::endl;
    std::cerr << "  max_iter  : int,   default 100" << std::endl;
    std::cerr << "  algorithm : brute or grid, default grid" << std::endl;
    std::cerr << "  --pbar    : show per-iteration progress bar on stderr" << std::endl;
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::time_t now_t = std::time(nullptr);
    std::tm* tm_info = std::localtime(&now_t);
    char ts_buf[16];
    std::strftime(ts_buf, sizeof(ts_buf), "%y%m%d_%H%M%S", tm_info);
    std::string timestamp(ts_buf);

    // Strip --pbar from argv before positional parsing
    bool show_pbar = false;
    std::vector<const char*> pos_args;
    pos_args.push_back(argv[0]);
    for(int i = 1; i < argc; ++i) {
        if(std::string(argv[i]) == "--pbar")
            show_pbar = true;
        else
            pos_args.push_back(argv[i]);
    }
    const int pos_argc = static_cast<int>(pos_args.size());

    const char* image_path = pos_args[1];
    float bandwidth = 150.0f;
    int max_iter = 100;
    std::string algorithm = "grid";

    if(pos_argc >= 3) bandwidth = std::stof(pos_args[2]);
    if(pos_argc >= 4) max_iter = std::stoi(pos_args[3]);
    if(pos_argc >= 5) algorithm = pos_args[4];

    if(algorithm != "brute" && algorithm != "grid") {
        std::cerr << "Unknown algorithm: " << algorithm << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    using clock = std::chrono::steady_clock;
    auto t_total_start = clock::now();

    auto t_load_start = clock::now();
    STBImage image;
    if(!image.loadImage(image_path)) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }
    auto t_load_end = clock::now();

    std::cout << "Image: " << image.width << "x" << image.height
              << " (" << image.width * image.height << " pixels)" << std::endl;
    std::cout << "Algorithm: " << algorithm
              << "  bandwidth=" << bandwidth
              << "  max_iter=" << max_iter << std::endl;

    auto t_conv_start = clock::now();
    std::vector<uint8_t> data = STBImageToStdVector(image);
    auto t_conv_end = clock::now();

    auto t_ms_start = clock::now();
    MeanShiftResult result{};
    if(algorithm == "brute")
        result = meanShift(data, bandwidth, max_iter, 1e-3f, show_pbar);
    else
        result = meanShiftOptimized(data, bandwidth, max_iter, 1e-3f, show_pbar);
    auto t_ms_end = clock::now();

    auto t_out_start = clock::now();
    cv::Mat image_ref = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat result_mat = vectorToCVMat(data, image.width, image.height);

    namespace fs = std::filesystem;
    fs::path input_p(image_path);
    fs::path out_dir = input_p.parent_path();
    std::string stem = input_p.stem().string();
    std::string output_path = (out_dir / (stem + "_" + timestamp + "_result.png")).string();
    std::string log_path    = (out_dir / (stem + "_" + timestamp + "_result.log")).string();
    cv::imwrite(output_path, result_mat);
    auto t_out_end = clock::now();

    auto t_total_end = clock::now();

    auto ms = [](auto start, auto end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    std::cout << "\n[Timing]" << std::endl;
    std::cout << "  Image load:        " << ms(t_load_start, t_load_end) << " ms" << std::endl;
    std::cout << "  Data conversion:   " << ms(t_conv_start, t_conv_end) << " ms" << std::endl;
    std::cout << "  Mean shift total:  " << ms(t_ms_start, t_ms_end) << " ms" << std::endl;
    if(algorithm == "grid")
        std::cout << "    Grid build:      " << result.grid_build_ms << " ms" << std::endl;
    std::cout << "    Pixel shifting:  " << result.pixel_shift_ms << " ms" << std::endl;
    std::cout << "  Result conversion: " << ms(t_out_start, t_out_end) << " ms" << std::endl;
    std::cout << "  Total:             " << ms(t_total_start, t_total_end) << " ms" << std::endl;
    std::cout << "  Iterations:        " << result.iterations << std::endl;
    std::cout << "\nResult saved to: " << output_path << std::endl;

    {
        std::ofstream log(log_path);
        log << "[Image]\n";
        log << "  file:    " << image_path << "\n";
        log << "  size:    " << image.width << "x" << image.height
            << "  (" << image.width * image.height << " pixels)\n";
        log << "\n[Algorithm]\n";
        log << "  method:     " << algorithm << "\n";
        log << "  bandwidth:  " << bandwidth << "\n";
        log << "  max_iter:   " << max_iter << "\n";
        log << "  tolerance:  " << 1e-3f << "\n";
        log << "\n[Output]\n";
        log << "  image:   " << output_path << "\n";
        log << "\n[Timing]\n";
        log << "  Image load:        " << ms(t_load_start, t_load_end) << " ms\n";
        log << "  Data conversion:   " << ms(t_conv_start, t_conv_end) << " ms\n";
        log << "  Mean shift total:  " << ms(t_ms_start, t_ms_end) << " ms\n";
        if(algorithm == "grid")
            log << "    Grid build:      " << result.grid_build_ms << " ms\n";
        log << "    Pixel shifting:  " << result.pixel_shift_ms << " ms\n";
        log << "  Result conversion: " << ms(t_out_start, t_out_end) << " ms\n";
        log << "  Total:             " << ms(t_total_start, t_total_end) << " ms\n";
        log << "  Iterations:        " << result.iterations << "\n";
    }
    std::cout << "Log saved to:    " << log_path << std::endl;

    if(image_ref.empty()) {
        std::cerr << "OpenCV could not load reference image for display" << std::endl;
        return 0;
    }

    std::vector<cv::Mat> images = {image_ref, result_mat};
    cv::Mat output;
    cv::hconcat(images, output);
    cv::imshow("MeanShift (right)", output);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}