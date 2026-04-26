#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "STBImage.h"
#include "meanshift_seq.h"
#include "meanshift_soa.h"
#include "meanshift_baseline.h"
#include "meanshift_omp.h"
#include "meanshift_omp_soa.h"

static void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <image> [bandwidth] [max_iter] [baseline|seq|soa|omp|omp_soa] [--pbar] [--no-display] [--no-output] [--kernel flat|gaussian|epanechnikov]" << std::endl;
    std::cerr << "  bandwidth    : float, default 150" << std::endl;
    std::cerr << "  max_iter     : int,   default 100" << std::endl;
    std::cerr << "  algorithm    : baseline, seq, soa, omp or omp_soa, default seq" << std::endl;
    std::cerr << "  --pbar       : show per-iteration progress bar on stderr" << std::endl;
    std::cerr << "  --no-display : skip OpenCV image display window" << std::endl;
    std::cerr << "  --no-output  : skip writing result PNG and log file" << std::endl;
    std::cerr << "  --kernel     : kernel function: flat (default), gaussian, epanechnikov" << std::endl;
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

    // Strip flags from argv before positional parsing
    bool show_pbar = false;
    bool no_display = false;
    bool no_output = false;
    std::string kernel_name = "flat";
    std::vector<const char*> pos_args;
    pos_args.push_back(argv[0]);
    for(int i = 1; i < argc; ++i) {
        if(std::string(argv[i]) == "--pbar")
            show_pbar = true;
        else if(std::string(argv[i]) == "--no-display")
            no_display = true;
        else if(std::string(argv[i]) == "--no-output")
            no_output = true;
        else if(std::string(argv[i]) == "--kernel" && i + 1 < argc)
            kernel_name = argv[++i];
        else
            pos_args.push_back(argv[i]);
    }
    const int pos_argc = static_cast<int>(pos_args.size());

    const char* image_path = pos_args[1];
    float bandwidth = 150.0f;
    int max_iter = 100;
    std::string algorithm = "seq";

    if(pos_argc >= 3) bandwidth = std::stof(pos_args[2]);
    if(pos_argc >= 4) max_iter = std::stoi(pos_args[3]);
    if(pos_argc >= 5) algorithm = pos_args[4];

    if(algorithm != "baseline" && algorithm != "seq" && algorithm != "soa" &&
       algorithm != "omp" && algorithm != "omp_soa") {
        std::cerr << "Unknown algorithm: " << algorithm << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    KernelFn kernel = makeKernel(kernel_name);

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
              << "  max_iter=" << max_iter
              << "  kernel=" << kernel_name << std::endl;

    auto t_conv_start = clock::now();
    std::vector<uint8_t> data;
    if(algorithm != "baseline")
        data = STBImageToStdVector(image);
    auto t_conv_end = clock::now();

    auto t_ms_start = clock::now();
    MeanShiftResult result{};
    if(algorithm == "baseline")
        result = meanShiftBaseline(image, bandwidth, max_iter, 1e-3f, show_pbar, kernel);
    else if(algorithm == "soa")
        result = meanShiftSoA(data, image.width, bandwidth, max_iter, 1e-3f, show_pbar, kernel);
    else if(algorithm == "omp")
        result = meanShiftOMP(data, image.width, bandwidth, max_iter, 1e-3f, show_pbar, kernel);
    else if(algorithm == "omp_soa")
        result = meanShiftSoAOMP(data, image.width, bandwidth, max_iter, 1e-3f, show_pbar, kernel);
    else
        result = meanShift(data, image.width, bandwidth, max_iter, 1e-3f, show_pbar, kernel);
    auto t_ms_end = clock::now();

    auto t_out_start = clock::now();
    std::string output_path;
    std::string log_path;

    if(!no_output) {
        namespace fs = std::filesystem;
        fs::path input_p(image_path);
        fs::path out_dir = input_p.parent_path();
        std::string stem = input_p.stem().string();
        output_path = (out_dir / (stem + "_" + timestamp + "_result.png")).string();
        log_path    = (out_dir / (stem + "_" + timestamp + "_result.log")).string();
        // Save result using stb_image_write — no OpenCV needed
        if(algorithm == "baseline")
            image.saveImage(output_path);
        else
            saveSTB(output_path, image.width, image.height, data);
    }
    auto t_out_end = clock::now();

    auto t_total_end = clock::now();

    auto ms = [](auto start, auto end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    std::cout << "\n[Timing]" << std::endl;
    std::cout << "  Image load:        " << ms(t_load_start, t_load_end) << " ms" << std::endl;
    std::cout << "  Data conversion:   " << ms(t_conv_start, t_conv_end) << " ms" << std::endl;
    std::cout << "  Mean shift total:  " << ms(t_ms_start, t_ms_end) << " ms" << std::endl;
    std::cout << "    Pixel shifting:  " << result.pixel_shift_ms << " ms" << std::endl;
    std::cout << "  Result conversion: " << ms(t_out_start, t_out_end) << " ms" << std::endl;
    std::cout << "  Total:             " << ms(t_total_start, t_total_end) << " ms" << std::endl;
    std::cout << "  Iterations:        " << result.iterations << std::endl;

    // Per-iteration table
    double avg_iter_ms = 0.0;
    if(!result.iter_details.empty()) {
        std::cout << "\n[Per-iteration]" << std::endl;
        std::cout << "  " << std::setw(5) << "Iter"
                  << std::setw(14) << "Time (ms)"
                  << std::setw(14) << "Max Change" << std::endl;
        for(const auto& info : result.iter_details) {
            std::cout << "  " << std::setw(5) << info.iteration
                      << std::setw(14) << std::fixed << std::setprecision(2) << info.time_ms
                      << std::setw(14) << std::fixed << std::setprecision(3) << info.max_change
                      << std::endl;
            avg_iter_ms += info.time_ms;
        }
        avg_iter_ms /= result.iter_details.size();
        std::cout << "  " << std::setw(5) << "Avg:"
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_iter_ms
                  << std::endl;
    }

    if(!no_output) {
        std::cout << "\nResult saved to: " << output_path << std::endl;

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
        log << "    Pixel shifting:  " << result.pixel_shift_ms << " ms\n";
        log << "  Result conversion: " << ms(t_out_start, t_out_end) << " ms\n";
        log << "  Total:             " << ms(t_total_start, t_total_end) << " ms\n";
        log << "  Iterations:        " << result.iterations << "\n";

        if(!result.iter_details.empty()) {
            log << "\n[Per-iteration]\n";
            log << "  " << std::setw(5) << "Iter"
                << std::setw(14) << "Time (ms)"
                << std::setw(14) << "Max Change" << "\n";
            for(const auto& info : result.iter_details) {
                log << "  " << std::setw(5) << info.iteration
                    << std::setw(14) << std::fixed << std::setprecision(2) << info.time_ms
                    << std::setw(14) << std::fixed << std::setprecision(3) << info.max_change
                    << "\n";
            }
            log << "  " << std::setw(5) << "Avg:"
                << std::setw(14) << std::fixed << std::setprecision(2) << avg_iter_ms
                << "\n";
        }
    }

    if(!no_display) {
        cv::Mat image_ref = cv::imread(image_path, cv::IMREAD_COLOR);
        if(image_ref.empty()) {
            std::cerr << "OpenCV could not load reference image for display" << std::endl;
            return 0;
        }
        cv::Mat result_mat;
        if(algorithm == "baseline")
            result_mat = STBImageToCVMat(image);
        else
            result_mat = vectorToCVMat(data, image.width, image.height);

        std::vector<cv::Mat> images = {image_ref, result_mat};
        cv::Mat output;
        cv::hconcat(images, output);
        cv::imshow("MeanShift (right)", output);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}
