#include "spatiogram_tracker.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>

spatiogramTracker::spatiogramTracker( int nBins, int nIter ){
    this->started=false;
    this->nBins = nBins;
    this->nIter = nIter;
}

spatiogramTracker::~spatiogramTracker(){

}

static cv::Rect checkRect(const cv::Mat &image, const cv::Rect &roi){
    cv::Rect out=roi;
    if ( out.x+out.width > image.cols ) out.x = out.x - (out.width - (image.cols - out.x)) - 1;
    if ( out.y+out.height > image.rows ) out.y = out.y - (out.height - (image.rows - out.y)) - 1;
    if ( out.x < 0 ) out.x=0;
    if ( out.y < 0 ) out.y=0;
    return out;
}

void spatiogramTracker::init(const cv::Mat &image, const cv::Rect &roi){
    imPatch = image(roi).clone();
    boundBox = roi;
    prev_boundBox = boundBox;
    computeSpatiogram(imPatch, nBins, targetModel);
    started=true;
}

void spatiogramTracker::update(const cv::Mat &image, cv::Rect &roi){

    for (int i=0; i<nIter; i++){
        // vars
        spatiogram colorModel;
        double rho_0;
        cv::Mat w, v, weights;

        // STEP 1
        // Calculate the pdf of the previous position
        imPatch = image(prev_boundBox).clone();
        computeSpatiogram(imPatch, nBins, colorModel);
        // Evaluate the similarity coefficient
        rho_0 = compareSpatiograms(targetModel, colorModel, w, v);
        //std::cout<<"iter_"<<i<<": "<<rho_0<<std::endl;

        // STEP 2
        // Derive the weights
        computeWeights(imPatch, targetModel, colorModel, w, weights);
        //cv::Mat nw=weights.clone();
        //cv::normalize(nw,nw,0,255,cv::NORM_MINMAX);
        //cv::imshow("we",nw);

        // STEP 3
        // Compute the mean-shift vector using Epanechnikov kernel
        cv::Point2d z = computeMeanshiftVector(prev_boundBox, weights, v);
        roi = prev_boundBox;
        roi.x = int(z.x);
        roi.y = int(z.y);
        roi=checkRect(image, roi);

        // STEPS 4 & 5
        double rho_1 = 0;
        for ( int cont=0; (cont<0 && rho_1 < rho_0); cont++ ){
            // Calculate the pdf of the new position
            cv::Mat newImPatch = image(roi).clone();
            spatiogram newColorModel;
            computeSpatiogram(newImPatch, nBins, newColorModel);

            // Evaluate the Bhattacharyya coefficient
            cv::Mat ww, vv;
            rho_1 = compareSpatiograms(targetModel, newColorModel, ww, vv);

            // Update center (correct)
            if (rho_1 < rho_0){
                roi.x = (0.5*prev_boundBox.x + 0.5*roi.x);
                roi.y = (0.5*prev_boundBox.y + 0.5*roi.y);
                roi=checkRect(image, roi);
            }
        }

        // STEP 6
        double norm1 = fabs(roi.x-prev_boundBox.x)+fabs(roi.y-prev_boundBox.y);
        if ( norm1 < 5 )
            break;
        //std::cout<<"norm: "<<norm1<<std::endl;
        prev_boundBox = roi;
    }
}

std::string spatiogramTracker::getName() const{
    return "mcv_tracker.spatiogram";
}

cv::Ptr<mcv_tracker> createSpatiogramTracker( int nBins, int nIter ){
    return cv::Ptr<mcv_tracker>( new spatiogramTracker( nBins, nIter ) );
}
