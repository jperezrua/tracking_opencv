#include "meanShift_tracker.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <vector>


meanShiftTracker::meanShiftTracker(){
    std::vector<int> channels, histSize;
    std::vector<float> ranges;
    channels.push_back(0);
    histSize.push_back(8);
    ranges.push_back(0);
    ranges.push_back(255);

    meanShiftTracker(channels, histSize, ranges);
}

meanShiftTracker::meanShiftTracker(std::vector<int>channels,std::vector<int>histSize, std::vector<float>ranges){
    started=false;
    this->channels = channels;
    this->histSize = histSize;
    this->ranges = ranges;
}

meanShiftTracker::~meanShiftTracker(){

}

void meanShiftTracker::init(const cv::Mat &image, const cv::Rect &roi){
    objectImage = image(roi).clone();
    objectPos = roi;
    objectModel = computeModel(objectImage);
    started=true;
}

void meanShiftTracker::update(const cv::Mat &image, cv::Rect &roi){
    cv::MatND backproj;
    float** ranges1 = new float*[2];
    for(int i = 0; i < 2; i++)
        ranges1[i] = new float[(int)channels.size()];
    for(int i = 0; i < (int)channels.size(); i++){
        ranges1[i][0]=ranges[0];
        ranges1[i][1]=ranges[1];
    }

    cv::calcBackProject(&image, 1, channels.data(), objectModel, backproj, (const float**)ranges1, 1);
    cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER,10,0.01);
    cv::meanShift(backproj, objectPos, criteria);
    //cv::Rect buf=objectPos;
    //cv::RotatedRect rect = cv::CamShift(backproj, buf, criteria);
    //cv::Mat dr=image.clone();
    //cv::ellipse(dr, rect, cv::Scalar(255,0,255));
    //cv::imshow("camshift",dr );
    roi=objectPos;
}

std::string meanShiftTracker::getName() const{
    return "mcv_tracker.meanShift";
}

cv::MatND meanShiftTracker::computeModel(const cv::Mat& im){
    cv::MatND model;
    float** ranges1 = new float*[2];
    for(int i = 0; i < 2; i++)
        ranges1[i] = new float[(int)channels.size()];
    for(int i = 0; i < (int)channels.size(); i++){
        ranges1[i][0]=ranges[0];
        ranges1[i][1]=ranges[1];
    }
    cv::calcHist(&im, 1, channels.data(), cv::Mat(), model, (int)channels.size(), histSize.data(), (const float**)ranges1);
    cv::normalize(model, model, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    return model;
}

/* constructor caller */
cv::Ptr<mcv_tracker> createMeanShiftTracker(std::vector<int>channels,std::vector<int>histSize, std::vector<float>ranges){
    return cv::Ptr<mcv_tracker>( new meanShiftTracker(channels,histSize,ranges) );
}
cv::Ptr<mcv_tracker> createMeanShiftTracker(){
    return cv::Ptr<mcv_tracker>( new meanShiftTracker() );
}
