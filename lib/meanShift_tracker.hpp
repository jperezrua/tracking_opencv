/*
    Juan Manuel PEREZ RUA
    Jilliam Maria DIAZ BARROS
*/

#ifndef MEANSHIFT_TRACKER_HPP
#define MEANSHIFT_TRACKER_HPP

#include "mcv_tracker.hpp"
#include <opencv2/core.hpp>

class meanShiftTracker : public mcv_tracker{
public:
    meanShiftTracker( );
    meanShiftTracker( std::vector<int> channels, std::vector<int> histSize, std::vector<float> ranges);
    ~meanShiftTracker();

    virtual void init(const cv::Mat& image, const cv::Rect& roi);
    virtual void update(const cv::Mat& image, cv::Rect& roi);
    virtual std::string getName() const;

private:
    cv::Mat objectImage;
    cv::MatND objectModel;
    cv::Rect objectPos;

    std::vector<int> channels;
    std::vector<int> histSize;
    std::vector<float> ranges;

    cv::MatND computeModel(const cv::Mat &im);
};

cv::Ptr<mcv_tracker> createMeanShiftTracker();
cv::Ptr<mcv_tracker> createMeanShiftTracker(std::vector<int>channels,
                                            std::vector<int>histSize, std::vector<float>ranges);
#endif
