/*
    Juan Manuel PEREZ RUA
    Jilliam Maria DIAZ BARROS
*/

#ifndef SPATIAL_PF_TRACKER_HPP
#define SPATIAL_PF_TRACKER_HPP

#include "mcv_tracker.hpp"
#include "spatiogram.hpp"
#include <opencv2/core.hpp>
#include <time.h>

class spatialPFTracker : public mcv_tracker{
public:
    spatialPFTracker(int nParticles=75, int nBins=8, double sigX=7.5, double sigY=7.5,
                      double sigS=7.5 );
    ~spatialPFTracker();

    virtual void init(const cv::Mat& image, const cv::Rect& roi);
    virtual void update(const cv::Mat& image, cv::Rect& roi);

    virtual std::string getName() const;

    void getParticles(std::vector<cv::Rect2d>& xpPred){xpPred = this->xpPred;}
    bool useSizeStates;
private:
    cv::Mat imPatch;
    cv::Rect boundBox;
    spatiogram targetModel;
    int nBins;
    int nIter;

    int nParticles;
    double sigX, sigY, sigS;
    double lambda;
    cv::RNG *rng;
    std::vector<cv::Rect2d> xpPred;
    std::vector<double> weights;
};

cv::Ptr<mcv_tracker> createSpatialPFTracker(int nParticles=75, int nBins=8,
                                            double sigX=2.0f, double sigY=2.0f, double sigS=2.0f);


#endif // SPATIAL_PF_TRACKER_HPP
