#include <vector>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "STBImage.h"




cv::Mat getOpencvImage1c(unsigned char* image, uint16_t height, uint16_t width) {
    cv::Mat opencv_image(cv::Size(width, height), CV_8UC1); 

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
                opencv_image.at<unsigned char>(y, x) = image[y*width + x];
            }
        }
    return opencv_image;
}


int main(int argc, char* argv[]){
    std::string rgb = {'R', 'G', 'B'};
    STBImage image;
    if(image.loadImage(argv[1])){
        std::cout<<"Image loaded successfully"<<std::endl;
    }else{
        std::cout<<"Failed to load image"<<std::endl;
    }

    std::cout<<image.width<<" "<<image.height<<" "<<image.channels<<std::endl;

    cv::imshow("Red Channel", STBImageToCVMat(image));
    cv::waitKey(0);
    cv::Mat image_ref = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::imshow("Red Channel", image_ref); // bgr
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}