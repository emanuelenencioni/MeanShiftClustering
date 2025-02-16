

#ifndef STBIMAGE_H
#define STBIMAGE_H
#include <opencv2/opencv.hpp>
/*
* Wrapper around STBImage lib.
*
*/
struct STBImage {
    int width{0}, height{0}, channels{0};
    uint8_t *rgb_image{nullptr};
    std::string filename{};

    bool loadImage(const std::string &name);

    void saveImage(const std::string &newName) const;
};

cv::Mat STBImageToCVMat(STBImage image);


#endif // STBIMAGE_H