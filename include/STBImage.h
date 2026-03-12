#ifndef STBIMAGE_H
#define STBIMAGE_H

#include <cstdint>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct STBImage {
    int width{0}, height{0}, channels{0};
    uint8_t* rgb_image{nullptr};
    std::string filename{};

    ~STBImage();

    bool loadImage(const std::string& name);
    void saveImage(const std::string& newName) const;
};

cv::Mat STBImageToCVMat(const STBImage& image);
std::vector<uint8_t> STBImageToStdVector(const STBImage& image);
cv::Mat vectorToCVMat(const std::vector<uint8_t>& image, int width, int height);

#endif // STBIMAGE_H
