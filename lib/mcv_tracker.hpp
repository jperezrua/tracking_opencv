/*
    Juan Manuel PEREZ RUA
    Jilliam Maria DIAZ BARROS
*/

#ifndef MCV_TRACKER_HPP
#define MCV_TRACKER_HPP

#include <opencv2/core.hpp>
#include <string>

class mcv_tracker {
public:
    mcv_tracker();
    virtual ~mcv_tracker() {}
    virtual void init(const cv::Mat& image, const cv::Rect& roi) = 0;
    virtual void update(const cv::Mat& image, cv::Rect& roi) = 0;

    virtual std::string getName() const = 0;
    bool started;
};

#endif
