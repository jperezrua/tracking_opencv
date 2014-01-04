/*
    Juan Manuel PEREZ RUA
    Jilliam Maria DIAZ BARROS
*/

#ifndef SPATIOGRAM_HPP
#define SPATIOGRAM_HPP

#include <opencv2/core.hpp>

struct spatiogram{
    cv::Mat cd; // color distribution
    cv::Mat mu; // mean vectors
    cv::Mat cm; // cov matrices
    double C;   // normalization value
    int bins;   // number of bins
};

void computeSpatiogram( const cv::Mat &imagePatch, int m, spatiogram &sg );
double compareSpatiograms( const spatiogram &p, const spatiogram &q, cv::Mat &w, cv::Mat &v );
void computeWeights( const cv::Mat &imagePatch, const spatiogram &qTarget, const spatiogram &pCurrent,
                     const cv::Mat &w, cv::Mat &weights );
cv::Point2d computeMeanshiftVector(const cv::Rect &patch, const cv::Mat &weights, const cv::Mat &v);

#endif
