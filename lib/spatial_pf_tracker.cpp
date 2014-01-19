#include "spatial_pf_tracker.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <limits>

spatialPFTracker::spatialPFTracker( int nParticles, int nBins, double sigX,
                                    double sigY, double sigS ){
    this->started=false;
    this->nBins = nBins;

    this->sigX = sigX;
    this->sigY = sigY;
    this->sigS = sigS;
    lambda = 200;
    rng = new cv::RNG(time(NULL));
    xpPred.resize(size_t(nParticles));
    weights.resize(size_t(nParticles));
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

static cv::Rect2d checkRect(const cv::Mat &image, const cv::Rect2d &roi){
    cv::Rect2d out=roi;
    //std::cout<<"roi="<<out.x<<","<<out.y<<","<<out.width<<","<<out.height<<std::endl;
    if ( out.x+out.width > (double)image.cols ) out.x = out.x - (out.width - ((double)image.cols - out.x)) - 1;
    if ( out.y+out.height > (double)image.rows ) out.y = out.y - (out.height - ((double)image.rows - out.y)) - 1;
    if ( out.x < 0 ) out.x=0;
    if ( out.y < 0 ) out.y=0;
    //std::cout<<"roi="<<out.x<<","<<out.y<<","<<out.width<<","<<out.height<<std::endl;
    return out;
}

static cv::Rect2d propagEstimate(std::vector<cv::Rect2d> &particles, const std::vector<double> &weights, cv::RNG* rng){
    cv::Rect2d estim=particles[0];
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
    //std::cout<<"N_res: "<<N_res<<std::endl;
    //std::cout<<"S: "<<S<<std::endl;
    //std::cout<<"sum_N_babies: "<<sum_N_babies<<std::endl;

    std::vector<double> cumDist(S);
    std::vector<double> cumProd(N_res);
    std::vector<double> expos;
    linspace(expos, N_res, 1, -1);

    if (N_res!=0){
        for (size_t i=0; i<q_res.size(); i++){
            q_res[i]=(q_res[i]-N_babies[i])/N_res;
            if (i==0) cumDist[i]=q_res[i];
            else cumDist[i]=cumDist[i-1]+q_res[i];
        }
        //std::cout<<"cumDist=";
        //for (int i=0; i<cumDist.size(); i++) std::cout<<" "<<cumDist[i]<<" ";
        //std::cout<<std::endl;
        std::vector<double> rands(N_res);
        //std::cout<<"idx=";
        for (int i=0; i<N_res; i++){
            rands[i] = std::pow( rng->uniform(0.0f,1.0f), 1/expos[i] );
            if (i==0){
                cumProd[(N_res-1)-i]=rands[i];
                //std::cout<<(N_res-1)-i<<" ";
            }else{
                cumProd[(N_res-1)-i]=cumProd[(N_res-1)-i+1]*rands[i];
                //std::cout<<(N_res-1-i+1)<<" ";
            }
        }
        //std::cout<<std::endl;
        //std::cout<<"1/expos=";
        //for (int i=0; i<expos.size(); i++) std::cout<<" "<<1/expos[i]<<" ";
        //std::cout<<std::endl;
        //std::cout<<"rands=";
        //for (int i=0; i<cumProd.size(); i++) std::cout<<" "<<cumProd[i]<<" ";
        //std::cout<<std::endl;
        //std::cout<<"cumProd=";
        //for (int i=0; i<cumProd.size(); i++) std::cout<<" "<<cumProd[i]<<" ";
        //std::cout<<std::endl;
        int j=0;
        for (int i=0; i<N_res; i++){
            while (cumProd[i]>cumDist[j]){
                j=j+1;
            }
            N_babies[j]+=1;
        }
        //std::cout<<"N_babies=";
        //for (int i=0; i<N_babies.size(); i++) std::cout<<" "<<N_babies[i]<<" ";
        //std::cout<<std::endl;
    }
    // propagate
    std::vector<cv::Rect2d> particles_=particles;
    int index=0;
    for (int i=0; i<S; i++){
      for (int j=index; j<index+N_babies[i]-1; j++){
         particles[j] = particles_[i];
      }
      index = index+N_babies[i];
    }

    // estimate
    estim.x=0; estim.y=0;
    for (size_t i=0;i<particles.size();i++){
        estim.x += weights[i]*particles[i].x;
        estim.y += weights[i]*particles[i].y;
    }

    std::cout<<"estimb="<<estim.x<<","<<estim.y<<","<<estim.width<<","<<estim.height<<std::endl;

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
        xpPred[i] = checkRect(image, rect_);

        // evaluate weights
        spatiogram model_;
        cv::Mat patch = image(xpPred[i]);
        if (xpPred[i].width<=0 && xpPred[i].height<=0){
            weights[i]=0.0; continue;
        }
        computeSpatiogram(patch, nBins, model_);
        double rho_1 = 1-compareSpatiograms(targetModel, model_);
        weights[i] = std::exp(-lambda*(rho_1)*(rho_1));
        sum_w += weights[i];
    }

    // normalize weights
    std::transform( weights.begin(), weights.end(), weights.begin(),
                    std::bind1st(std::multiplies<double>(),1/(sum_w+std::numeric_limits<double>::epsilon())) );

    // program the function!!!!
    //std::cout<<"SIZE IM: "<<image.rows<<","<<image.cols<<std::endl;
    //std::cout<<"preda="<<xpPred[0].x<<","<<xpPred[0].y<<","<<xpPred[0].width<<","<<xpPred[0].height<<std::endl;
    cv::Rect2d roi2d = propagEstimate( xpPred, weights, rng );
    //std::cout<<"predb="<<xpPred[0].x<<","<<xpPred[0].y<<","<<xpPred[0].width<<","<<xpPred[0].height<<std::endl;
    //std::cout<<"pred="<<roi2d.x<<","<<roi2d.y<<","<<roi2d.width<<","<<roi2d.height<<std::endl;
    roi2d = checkRect(image, roi2d);
    //std::cout<<"pred="<<roi2d.x<<","<<roi2d.y<<","<<roi2d.width<<","<<roi2d.height<<std::endl;
    roi.x = int(roi2d.x);
    roi.y = int(roi2d.y);
    roi.height = int(roi2d.height);
    roi.width = int(roi2d.width);

    /*cv::Mat im1=image.clone();
    std::cout<<"SIZE PRED: "<<xpPred.size()<<std::endl;
    for (size_t i=0; i<xpPred.size(); i++){
        cv::rectangle( im1, xpPred[i], cv::Scalar( 255, 255, 255 ), 2, 1 );
        cv::imshow("a",im1);
    }
    cv::waitKey(0);*/
}

std::string spatialPFTracker::getName() const{
    return "mcv_tracker.spatialpf";
}

cv::Ptr<mcv_tracker> createSpatialPFTracker( int nParticles, int nBins, double sigX,
                                             double sigY, double sigS ){
    return cv::Ptr<mcv_tracker>( new spatialPFTracker( nParticles, nBins, sigX, sigY, sigS ) );
}
