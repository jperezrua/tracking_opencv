/*
    Juan Manuel PEREZ RUA
    Jilliam Maria DIAZ BARROS
*/

#ifndef SPATIO_TRACKER_HPP
#define SPATIO_TRACKER_HPP

#include "mcv_tracker.hpp"
#include "spatiogram.hpp"
#include <opencv2/core.hpp>

class spatiogramTracker : public mcv_tracker{
public:
    spatiogramTracker( int nBins=8, int nIter=10 );
    ~spatiogramTracker();

    virtual void init(const cv::Mat& image, const cv::Rect& roi);
    virtual void update(const cv::Mat& image, cv::Rect& roi);

    virtual std::string getName() const;
private:
    cv::Mat imPatch;
    cv::Rect boundBox;
    cv::Rect prev_boundBox;
    spatiogram targetModel;
    int nBins;
    int nIter;
};

cv::Ptr<mcv_tracker> createSpatiogramTracker( int nBins, int nIter );

#endif
