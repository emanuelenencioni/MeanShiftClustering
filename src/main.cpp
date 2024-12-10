#include <vector>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "STBImage.h"






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