#include "STBImage.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

STBImage::~STBImage() {
    if(rgb_image) {
        stbi_image_free(rgb_image);
        rgb_image = nullptr;
    }
}

bool STBImage::loadImage(const std::string& name) {
    rgb_image = stbi_load(name.c_str(), &width, &height, &channels, 3);
    if(channels > 3) {
        channels = 3;
    }
    if(!rgb_image)
        return false;

    filename = name;
    return true;
}

void STBImage::saveImage(const std::string& newName) const {
    stbi_write_png(newName.c_str(), width, height, channels, rgb_image, width * channels);
}

cv::Mat STBImageToCVMat(const STBImage& image) {
    cv::Mat opencv_image(cv::Size(image.width, image.height), CV_8UC3);
    for(int y = 0; y < image.height; y++) {
        for(int x = 0; x < image.width; x++) {
            uint8_t* stb_pixel = &image.rgb_image[(y * image.width + x) * 3];
            // RGB (stb) -> BGR (OpenCV)
            opencv_image.at<cv::Vec3b>(y, x) = cv::Vec3b(stb_pixel[2], stb_pixel[1], stb_pixel[0]);
        }
    }
    return opencv_image;
}

std::vector<uint8_t> STBImageToStdVector(const STBImage& image) {
    int dim = image.height * image.width * 3;
    std::vector<uint8_t> output_vec(dim);
    for(int i = 0; i < dim; i++) {
        output_vec[i] = image.rgb_image[i];
    }
    return output_vec;
}

cv::Mat vectorToCVMat(const std::vector<uint8_t>& image, int width, int height) {
    cv::Mat opencv_image(cv::Size(width, height), CV_8UC3);
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            const uint8_t* pixel = &image[(y * width + x) * 3];
            // RGB (data) -> BGR (OpenCV)
            opencv_image.at<cv::Vec3b>(y, x) = cv::Vec3b(pixel[2], pixel[1], pixel[0]);
        }
    }
    return opencv_image;
}

bool saveSTB(const std::string& path, int width, int height,
             const std::vector<uint8_t>& data) {
    int stride = width * 3;
    return stbi_write_png(path.c_str(), width, height, 3, data.data(), stride) != 0;
}
