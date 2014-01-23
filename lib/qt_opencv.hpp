#ifndef QT_OPENCV_HPP
#define QT_OPENCV_HPP

#include <QImage>
#include <opencv2/core.hpp>
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>

// tracker state info
struct trackerinfo{
    cv::Mat image;
    bool selectObject;
    bool startSelection;
    cv::Rect bbox;
};

QImage mat2QImage(cv::Mat &mat);

#endif 
