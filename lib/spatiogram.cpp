#include "spatiogram.hpp"
#include <limits>
#include <iostream>


static void meshgrid(const cv::Range &xgv, const cv::Range &ygv,
                         cv::Mat &X, cv::Mat &Y)
{
  std::vector<int> t_x, t_y;
  for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
  for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);

  cv::Mat Tx=cv::Mat(t_x), Ty=cv::Mat(t_y);
  cv::repeat(Tx.reshape(1,1), Ty.total(), 1, X);
  cv::repeat(Ty.reshape(1,1).t(), 1, Tx.total(), Y);
}

static void robustnorm(const cv::Mat &X, const cv::Mat &Y, cv::Mat &D2)
{
    cv::Mat tX, tY;
    X.convertTo(tX,CV_64FC1);
    Y.convertTo(tY,CV_64FC1);
    D2 = tX.mul(tX) + tY.mul(tY);
    cv::sqrt(D2, D2);
    //cv::normalize(D2, D2, 0, 1, cv::NORM_RELATIVE, CV_64FC1);
    D2 = D2 / *std::max_element(D2.begin<double>(), D2.end<double>());
}

static void epanechnikov(const cv::Mat &D2, cv::Mat &K)
{
    K=2/M_PI-(2/M_PI)*D2;
}

static void linspace(std::vector<double> &vec, double start, double end, double space)
{
    double delta=space;
    double val=start;

    vec.clear();
    int bins = (end-start)/delta;
    for (int i=0; i<=bins; i++){
        vec.push_back(val);
        val += delta;
    }
}

static void binelements(const cv::Mat &imagePatch, const std::vector<double> bins, int channel, int bin, cv::Mat &mask){

    mask = cv::Mat::zeros(imagePatch.rows,imagePatch.cols, CV_64FC1);

    for (int i=0; i<imagePatch.rows; i++) {
        for (int j=0; j<imagePatch.cols; j++) {
            switch (imagePatch.channels()){
                case 3:
                if ( (double)imagePatch.at<cv::Vec3b>(i,j)[channel] >= bins[bin] &&
                     (double)imagePatch.at<cv::Vec3b>(i,j)[channel]  < bins[bin+1]){
                    mask.at<double>(i,j) = 1; // is in the correct bin, mark it
                }break;
                case 1:
                if ( (double)imagePatch.at<uchar>(i,j) >= bins[bin] &&
                     (double)imagePatch.at<uchar>(i,j)  < bins[bin+1]){
                    mask.at<double>(i,j) = 1; // is in the correct bin, mark it
                }break;
            }
        }
    }
}

static void mat3min(const std::vector<cv::Mat> &im, cv::Mat &m3m){

    m3m = im[0];

    if (im.size()>1){
        for (int i=0; i<im[0].rows; i++) {
            for (int j=0; j<im[0].cols; j++) {
                switch (im.size()){
                    case 3:
                    {
                        double val = std::min(im[0].at<double>(i,j),im[1].at<double>(i,j));
                        val = std::min(val,im[2].at<double>(i,j));
                        m3m.at<double>(i,j) = val;
                    }break;
                    case 2:
                    {
                        double val = std::min(im[0].at<double>(i,j),im[1].at<double>(i,j));
                        m3m.at<double>(i,j) = val;
                    }break;
                }
            }
        }
    }
}

void computeSpatiogram(const cv::Mat &imagePatch, int m, spatiogram &sg){
    int n = 256/m;
    std::vector<double> bins;
    linspace(bins, 0, 256, n);
    int height = imagePatch.rows;
    int width = imagePatch.cols;

    cv::Mat X,Y, Xm, Ym, dist2, K;
    meshgrid(cv::Range(1,width), cv::Range(1,height), X, Y);
    meshgrid(cv::Range(-width/2,width/2), cv::Range(-height/2,height/2), Xm, Ym);
    X.convertTo(X, CV_64FC1); Y.convertTo(Y, CV_64FC1);
    robustnorm(Xm,Ym,dist2);
    dist2 = dist2(cv::Rect(0,0,width,height));
    epanechnikov(dist2, K);

    // initialize spatiogram
    sg.C = 1/cv::sum(K)[0];
    sg.C = sg.C/(imagePatch.channels());
    sg.cd = cv::Mat::zeros(1, m*imagePatch.channels(), CV_64FC1);
    sg.mu = cv::Mat::zeros(2, m*imagePatch.channels(), CV_64FC1);
    sg.cm = cv::Mat::zeros(2, m*imagePatch.channels(), CV_64FC1);
    sg.bins = m*imagePatch.channels();

    for (int l=0;l<imagePatch.channels();l++){
        for (int j=0;j<m;j++){
            cv::Mat temp;
            binelements(imagePatch, bins, l, j, temp);
            // zeroth order spatiogram
            sg.cd.at<double>(0,l*m+j) = cv::sum( K.mul(temp) )[0];
            double den = cv::sum(temp)[0];
            if (den==0.0) den=std::numeric_limits<double>::epsilon();
            if (sg.cd.at<double>(0,l*m+j)==0.0) sg.cd.at<double>(0,l*m+j)=std::numeric_limits<double>::epsilon();

            // Mean vector of the pixels' coordinates mu: first order spatiogram
            double mu_x = cv::sum( X.mul(temp) )[0]/den;
            double mu_y = cv::sum( Y.mul(temp) )[0]/den;
            sg.mu.at<double>(0, l*m+j) = mu_x;
            sg.mu.at<double>(1, l*m+j) = mu_y;

            // Covariance matrix of the pixels' coordinates Cm[2x2]: 2nd order spatiogram
            cv::Mat C11; cv::pow(X - mu_x, 2, C11); //(x-mu_x)^2
            cv::Mat C22; cv::pow(Y - mu_y, 2, C22); //(y-mu_y)^2
            sg.cm.at<double>(0, l*m+j) = cv::sum(C11.mul(temp))[0]/den; //Cov(1,1)
            sg.cm.at<double>(1, l*m+j) = cv::sum(C22.mul(temp))[0]/den; //Cov(2,2)
        }
    }
    //normalize color distribution
    sg.cd = sg.C*sg.cd;
}

double compareSpatiograms(const spatiogram &p, const spatiogram &q, cv::Mat &w, cv::Mat &v){
    CV_Assert(p.bins==q.bins);

    cv::Mat temp = cv::Mat::zeros(1, p.bins, CV_64FC1);
    w = cv::Mat::zeros(1, p.bins, CV_64FC1);
    v = cv::Mat::zeros(2, p.bins, CV_64FC1);

    for (int i=0; i<p.bins; i++){
        // Means
        cv::Mat ub1(2,1,CV_64FC1);
        ub1.at<double>(0,0)=p.mu.at<double>(0,i);
        ub1.at<double>(1,0)=p.mu.at<double>(1,i);
        cv::Mat ub2(2,1,CV_64FC1);
        ub2.at<double>(0,0)=q.mu.at<double>(0,i);
        ub2.at<double>(1,0)=q.mu.at<double>(1,i);
        // covariances in matrix format and clipped to 1
        cv::Mat cm1=cv::Mat::ones(2,2,CV_64FC1);
        cm1.at<double>(0,0) += p.cm.at<double>(0,i);
        cm1.at<double>(1,1) += p.cm.at<double>(1,i);
        cv::Mat cm2=cv::Mat::ones(2,2,CV_64FC1);
        cm2.at<double>(0,0) += q.cm.at<double>(0,i);
        cm2.at<double>(1,1) += q.cm.at<double>(1,i);
        // weightening factor
        cv::Mat invS = cm1.inv(cv::DECOMP_LU) + cm2.inv(cv::DECOMP_LU);
        cv::Mat diff = ub2-ub1;
        cv::Mat expo = (-0.5)*diff.t()*invS*diff;
        w.at<double>(0,i) = std::exp( expo.at<double>(0,0) );
        // compute v
        cv::Mat V = w.at<double>(0,i)*std::sqrt(p.cd.at<double>(0,i)*q.cd.at<double>(0,i))*invS*(ub1-ub2);
        v.at<double>(0,i) = V.at<double>(0,0);
        v.at<double>(1,i) = V.at<double>(1,0);
        // Bhattacharyya coefficient
        double rho = sqrt( p.cd.at<double>(0,i)*q.cd.at<double>(0,i) );
        temp.at<double>(0,i) = (w.at<double>(0,i)) * rho;
    }
    return cv::sum(temp)[0];
}

void computeWeights(const cv::Mat &imagePatch, const spatiogram &qTarget, const spatiogram &pCurrent,
                    const cv::Mat &w, cv::Mat &weights){

    CV_Assert(qTarget.bins==pCurrent.bins);
    cv::Mat sqco;
    cv::divide(qTarget.cd,pCurrent.cd,sqco);
    cv::sqrt(sqco, sqco);
    cv::Mat rel = w.mul( sqco );
    std::vector<cv::Mat> weightC;
    cv::Mat tc =cv::Mat::zeros(imagePatch.rows, imagePatch.cols, CV_64FC1);

    int n = 256/qTarget.bins;
    std::vector<double> bins;
    linspace(bins, 0, 256, n);
    int m = pCurrent.bins/imagePatch.channels();
    for (int l=0; l<imagePatch.channels(); l++){
        for (int j=0; j<m; j++){
            cv::Mat temp;
            binelements(imagePatch, bins, l, j, temp);
            tc = tc + (rel.at<double>(0,l*m+j))*temp;
        }
        weightC.push_back(qTarget.C*tc);
    }
    mat3min( weightC, weights );
    //weights=weightC[0];
}

cv::Point2d computeMeanshiftVector(const cv::Rect &patch, const cv::Mat &weights, const cv::Mat &v){
    int height = patch.height;
    int width = patch.width;
    cv::Mat X,Y;
    meshgrid(cv::Range(1,width), cv::Range(1,height), X, Y);
    X.convertTo(X, CV_64FC1);
    Y.convertTo(Y, CV_64FC1);

    double den = cv::sum(weights)[0];
    if (den==0.0) den=std::numeric_limits<double>::epsilon();

    cv::Point2d z;
    z.x = ( cv::sum(X.mul(weights))[0] - cv::sum(v.row(0))[0] )/den;
    z.y = ( cv::sum(Y.mul(weights))[0] - cv::sum(v.row(1))[0] )/den;

    z.x = patch.x-patch.width/2 + z.x;
    z.y = patch.y-patch.height/2 + z.y;

    return z;
}


