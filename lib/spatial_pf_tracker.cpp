#include "spatial_pf_tracker.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <limits>
#include <functional>
#include <algorithm>

spatialPFTracker::spatialPFTracker( int nParticles, int nBins, double sigX,
                                    double sigY, double sigS ){
    this->started=false;
    this->nBins = nBins;

    this->sigX = sigX;
    this->sigY = sigY;
    this->sigS = sigS;
    lambda = 100;
    rng = new cv::RNG(time(NULL));
    xpPred.resize(size_t(nParticles));
    weights.resize(size_t(nParticles));
    useSizeStates=false;
}

spatialPFTracker::~spatialPFTracker(){
    delete rng;
}


static void linspace(std::vector<double> &vec, double start, double end, double space){
    double delta=space;
    double val=start;

    vec.clear();
    int bins = fabs((end-start)/delta);
    for (int i=0; i<=bins; i++){
        vec.push_back(val);
        val += delta;
    }
}

static cv::Rect2d checkRect(const cv::Mat &image, const cv::Rect2d &roi, int mins=3){
    cv::Rect2d out=roi;

    if ( out.width<mins ) out.width=3;
    if ( out.height<mins ) out.height=3;
    if ( out.width>image.cols*0.85 ) out.width=image.cols*0.85;
    if ( out.height>image.rows*0.85 ) out.height=image.rows*0.85;

    if ( out.x+out.width > (double)image.cols ) out.x = out.x - (out.width - ((double)image.cols - out.x)) - 1;
    if ( out.y+out.height > (double)image.rows ) out.y = out.y - (out.height - ((double)image.rows - out.y)) - 1;
    if ( out.x < 0 ) out.x=0;
    if ( out.y < 0 ) out.y=0;

    return out;
}

static cv::Rect2d propagEstimate(std::vector<cv::Rect2d> &particles, const std::vector<double> &weights, cv::RNG* rng, bool use=false){
    cv::Rect2d estim;
    for (size_t i=0; i<particles.size(); i++){
        estim.x+=particles[i].x;
        estim.y+=particles[i].y;
        estim.width+=particles[i].width;
        estim.height+=particles[i].height;
    }
    estim.x/=particles.size();
    estim.y/=particles.size();
    estim.width/=particles.size();
    estim.height/=particles.size();

    //std::cout<<"estima="<<estim.x<<","<<estim.y<<","<<estim.width<<","<<estim.height<<std::endl;

    // PURPOSE : Performs the resampling stage of the SIR
    //           in order(number of samples) steps. It uses Liu's
    //           residual resampling algorithm and Niclas' magic line.


    int S = (int)particles.size();
    std::vector<int> N_babies(S);
    std::vector<double> q_res = weights;

    std::transform( q_res.begin(), q_res.end(), q_res.begin(),
                    std::bind1st(std::multiplies<double>(), double(S)) );

    int sum_N_babies=0;
    for (size_t i=0; i<N_babies.size(); i++){
        if (q_res[i]>0) N_babies[i] = (int)std::floor(q_res[i]);
        else N_babies[i] = (int)std::ceil(q_res[i]);

        sum_N_babies+=N_babies[i];
    }

    int N_res = S - sum_N_babies;

    std::vector<double> cumDist(S);
    std::vector<double> cumProd(N_res);
    std::vector<double> expos;
    linspace(expos, N_res, 1, -1);

    if (N_res>0){
        for (size_t i=0; i<q_res.size(); i++){
            q_res[i]=(q_res[i]-N_babies[i])/N_res;
            if (i==0) cumDist[i]=q_res[i];
            else cumDist[i]=cumDist[i-1]+q_res[i];
        }
        std::vector<double> rands(N_res);
        for (int i=0; i<N_res; i++){
            rands[i] = std::pow( rng->uniform(0.0f,1.0f), 1/expos[i] );
            if (i==0){
                cumProd[(N_res-1)-i]=rands[i];
            }else{
                cumProd[(N_res-1)-i]=cumProd[(N_res-1)-i+1]*rands[i];
            }
        }
        int j=0;
        for (int i=0; i<N_res; i++){
            while (cumProd[i]>cumDist[j]){
                j=j+1;
            }
            if (j<N_babies.size())
                N_babies[j]+=1;
        }
        // propagate
        std::vector<cv::Rect2d> particles_=particles;
        int index=0;
        for (int i=0; i<S; i++){
          for (int j=index; j<index+N_babies[i]/*-1*/; j++){
              if (j<particles.size())
                particles[j] = particles_[i];
          }
          index = index+N_babies[i];
        }

        // estimate
        estim.x=0;
        estim.y=0;
        if (use){
            estim.width=0;
            estim.height=0;
        }
        for (size_t i=0;i<particles.size();i++){
            estim.x += weights[i]*particles[i].x;
            estim.y += weights[i]*particles[i].y;
            if (use){
                estim.width += weights[i]*particles[i].width;
                estim.height += weights[i]*particles[i].height;
            }
        }
    }

    //std::cout<<"estimb="<<estim.x<<","<<estim.y<<","<<estim.width<<","<<estim.height<<std::endl;

    return estim;
}

void spatialPFTracker::init(const cv::Mat &image, const cv::Rect &roi){
    imPatch = image(roi).clone();
    boundBox = roi;
    computeSpatiogram(imPatch, nBins, targetModel);
    started=true;

    // initialize particles
    for (size_t i=0; i<xpPred.size(); i++){
        cv::Rect2d rect_;
        rect_ = boundBox;
        rect_.x += rng->gaussian(sigX);
        rect_.y += rng->gaussian(sigY);

        if (useSizeStates){
            rect_.width += rng->gaussian(sigS);
            rect_.height += rng->gaussian(sigS);
        }

        xpPred[i] = checkRect(image, rect_);
    }

}

void spatialPFTracker::update(const cv::Mat &image, cv::Rect &roi){

    double sum_w=0.0;
    for (size_t i=0; i<xpPred.size(); i++){
        //prediction
        cv::Rect2d rect_ = xpPred[i];
        rect_.x += rng->gaussian(sigX);
        rect_.y += rng->gaussian(sigY);

        if (useSizeStates){
            rect_.width += rng->gaussian(sigS);
            rect_.height += rng->gaussian(sigS);
        }
        xpPred[i] = checkRect(image, rect_);

        /*std::cout<<"------------------------------"<<std::endl;
        std::cout<<"im="<<image.rows<<","<<image.cols<<std::endl;
        std::cout<<"xp.x="<<xpPred[i].x<<std::endl;
        std::cout<<"xp.y="<<xpPred[i].y<<std::endl;
        std::cout<<"xp.height="<<xpPred[i].height<<std::endl;
        std::cout<<"xp.width="<<xpPred[i].width<<std::endl;*/

        // evaluate weights
        spatiogram model_;
        cv::Mat patch;
        //try{
            patch = image(xpPred[i]);
        /*}catch(...){
            weights[i]=0.0; continue;
        }*/
        if (xpPred[i].width<=0 && xpPred[i].height<=0){
            weights[i]=0.0; continue;
        }
        computeSpatiogram(patch, nBins, model_);
        //double rho_1 = 1-compareSpatiograms(targetModel, model_);
        double rho_1 = 1-compareSpatiograms2(targetModel, model_);
        weights[i] = std::exp(-lambda*(rho_1)*(rho_1));
        sum_w += weights[i];
    }

    // normalize weights
    std::transform( weights.begin(), weights.end(), weights.begin(),
                    std::bind1st(std::multiplies<double>(),1/(sum_w+std::numeric_limits<double>::epsilon())) );

    // program the function!!!!
    cv::Rect2d roi2d = propagEstimate( xpPred, weights, rng, useSizeStates );
    roi2d = checkRect(image, roi2d);
    roi.x = int(std::floor(roi2d.x));
    roi.y = int(std::floor(roi2d.y));
    roi.height = int(std::floor(roi2d.height));
    roi.width = int(std::floor(roi2d.width));
}

std::string spatialPFTracker::getName() const{
    return "mcv_tracker.spatialpf";
}

cv::Ptr<mcv_tracker> createSpatialPFTracker( int nParticles, int nBins, double sigX,
                                             double sigY, double sigS ){
    return cv::Ptr<mcv_tracker>( new spatialPFTracker( nParticles, nBins, sigX, sigY, sigS ) );
}
