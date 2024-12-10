#include "STBImage.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

bool STBImage::loadImage(const std::string &name) {
        rgb_image = stbi_load(name.c_str(), &width, &height, &channels, 3);
        if(channels > 3){
            channels = 3;
        }
        if (!rgb_image)
            return false;
        else {
            filename = name;
            return true;
        }
    }
void STBImage::saveImage(const std::string &newName) const {
        stbi_write_jpg(newName.c_str(), width, height, channels, rgb_image, width * channels);
}

cv::Mat STBImageToCVMat(STBImage image) {
    unsigned char* stb_pixel;
    cv::Mat opencv_image(cv::Size(image.width, image.height), CV_8UC3);
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            stb_pixel = &image.rgb_image[(y * image.width*image.channels) + (x * image.channels)];
            
                unsigned char r, g, b;
                r = stb_pixel[0];
                g = stb_pixel[1];
                b = stb_pixel[2];

                cv::Vec3b opencv_pixel(r, g, b);
                opencv_image.at<cv::Vec3b>(y, x) = opencv_pixel;
        }
    }   return opencv_image;
}
