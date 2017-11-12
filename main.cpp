#include<opencv2/opencv.hpp>
#include<opencv2/tracking.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<cmath>
using namespace std;
using namespace cv;
#define Gamma 2
vector<float> hist(int& bin,int& x,int& y,int& cell_sz,Mat& mag,Mat& ang){

    vector<float> mag_plus(bin,0);
    vector<float> block_mag_plus;
    vector<float> block_mag_plus_norm;

    double step = floor(180/bin);
    for(int cols=0;cols<2;cols++)//每个block有四个cell，所以cols和rows最大取值为２
    {
        for(int rows=0;rows<2;rows++)
        {
            for(int i=(x+cell_sz*cols);i<(x+cell_sz*(cols+1));i++)
            {
                for(int j=(y+cell_sz*rows);j<(y+cell_sz*(rows+1));j++)
                {
                    float per_ang = ang.at<float>(i,j);//取出每个位置上的角度
                    float per_mag = mag.at<float>(i,j);//取出每个位置上的幅值

                    if(per_ang<180)
                    {
                        double n = floor(per_ang/step);//角度除区间，得到保存的下标

                        mag_plus[n] +=  per_mag;//将落在区间中的幅值进行累加

                    }else{
                        double n = floor((per_ang-180)/step);
                        mag_plus[n] +=per_mag;
                    }//因为是将３６０度分为９个区间，所以对角的区间合并为同一个
                }

            }


            auto iter = mag_plus.begin();
            for(;iter!=mag_plus.end();iter++)
            {
                //cout<<"*it0"<<*it0<<endl;
                float mag_value = *iter;
                //cout<<"mag_value = "<<mag_value<<endl;
                block_mag_plus.push_back(mag_value);
                *iter = 0;
            }
        }
    }
    /**
    double add_norm = 0;
    for(int ii=0;ii<block_mag_plus.size();ii++)
    {
        add_norm += pow(block_mag_plus[ii],2);
    }
    double add_norm_sqrt=0;
    add_norm_sqrt = sqrt(add_norm);
    for(int jj=0;jj<block_mag_plus.size();jj++)
    {
        block_mag_plus[jj] = block_mag_plus[jj]/add_norm_sqrt;
    }
   **/

    normalize(block_mag_plus,block_mag_plus_norm,1,0,NORM_L2);//对block的光流直方图进行归一化
    block_mag_plus = block_mag_plus_norm;

    return block_mag_plus;//返回一个block的光流统计结果

}

Mat correctGamma( Mat& img, double gamma ) {
    double inverse_gamma = 1.0 / gamma;

    Mat lut_matrix(1, 256, CV_8UC1 );
    uchar * ptr = lut_matrix.ptr();
    for( int i = 0; i < 256; i++ )
        ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );

    Mat result;
    LUT( img, lut_matrix, result );

    return result;
}//伽马校正

int main(int argc,char* argv[]){

    Mat img0;
    Mat img1;

    img0 = imread("../data/basketball01.jpg");
    img1 = imread("../data/basketball04.jpg");//读取图片

    cvtColor(img0,img0,CV_BGR2GRAY);
    cvtColor(img1,img1,CV_BGR2GRAY);//转化为灰度图

    resize(img1,img1,cv::Size(640,480));//固定图像img1大小
    resize(img0,img0,cv::Size(640,480));//固定图像img0大小

    img0 = correctGamma(img0,Gamma);//对img0进行gamma校正
    img1 = correctGamma(img1,Gamma);//对img1进行gamma校正

    Mat flow;//光流

    calcOpticalFlowFarneback(img0, img1, flow, 0.5, 3,15, 3, 5, 1.2, 0);//计算光流

    for(int y=0; y<img0.rows; y+=10 )
    {
        for(int x=0; x<img1.cols; x+=10)
        {
            Point2f fxy = flow.at<Point2f>(y, x);
            line(img0, Point(x, y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), CV_RGB(0,255,0));
        }

    }//这部分循环将光流画在图上和hof无关

    Mat flow_x(img0.rows,img0.cols,CV_32F);//存放每个像素x方向的光流
    Mat flow_y(img0.rows,img0.cols,CV_32F);//存放每个像素y方向的光流

    for(int i=0; i<flow.rows; i++)
    {
        for(int j=0; j<flow.cols; j++)
        {
            Point2f flow_p = flow.at<Point2f>(i,j);

            flow_x.at<float>(i,j)=flow_p.x;

            flow_y.at<float>(i,j)=flow_p.y;

        }
    }//对flow_x和flow_y进行赋值

    Mat img = imread("../data/0001.jpg");
    img.convertTo(img, CV_32F, 1/255.0);
    
    resize(img,img,Size(64,128));
    // Calculate gradients gx, gy
    //Mat gx, gy;
    //Mat flow_x,flow_y;
    //Sobel(img, flow_x, CV_32F, 1, 0, 1);
    //Sobel(img, flow_y, CV_32F, 0, 1, 1);
    Mat mag;//每个像素点的幅值
    Mat angle;//每个像素点的梯度
    cartToPolar(flow_x,flow_y,mag,angle,1);//计算每个像素点的幅值和角度
   //cartToPolar(gx,gy,mag,angle,1);
   /**
    for(int ii = 0;ii<angle.cols;ii++)
    {
        for(int jj=0;jj<angle.rows;jj++){

            cout<<"angle = "<<angle.at<float>(ii,jj)<<endl;
        }
    }
    **/
    vector<vector<float> > hist_accumulate;//将n个block的统计图放在hist_accumulate
    int cell_sz = 8;//每个cell的宽
    int hist_bin = 30;//每个cell统计的bin的大小
    int block_stride = 8;//取block的步长
    //int block_num = 0;
    for(int i=0;i<(mag.cols-2*cell_sz+1);i+= block_stride)
    {
        for(int j=0;j<(mag.rows-2*cell_sz+1);j+=block_stride)
        {   //block_num++;
            //cout<<"i"<<i<<endl;
            //cout<<"j"<<j<<endl;
            //cout<<"block_num"<<block_num<<endl;
            hist_accumulate.push_back(hist( hist_bin, i, j, cell_sz, mag, angle));//hist函数每次会返回每个block中的一个３６维度的统计结果
        }
        //cout<<"i = "<<i<<endl;
    }//这部分先取出不同的block,然后在每个block中进行直方图统计
    cout<<"mag.rows = "<<mag.rows<<endl;
    cout<<"mag.cols = "<<mag.cols<<endl;
    //cout<<"block_num = "<<block_num<<endl;
    cout<<"hist_accumulate = "<<hist_accumulate.size()<<endl;
    cout<<"hist_accumulate[0] = "<<hist_accumulate[0].size()<<endl;

    vector<float> descriptors;//hof向量
    vector<float> per_hist;
    double hist_sz = hist_accumulate.size();
    cout<<"hist_sz = "<<hist_sz<<endl;
    for(int i = 0; i<hist_sz;i++)
    {
        per_hist = hist_accumulate[i];
        for(int j = 0;j<36;j++)
        {
            descriptors.push_back(per_hist[j]);
        }

    }//这部分将ＨＯＦ拉直保存到descriptors中

    cout<<"descriptors = "<<descriptors.size()<<endl;

    for(int ii = 0;ii<descriptors.size();ii++)
    {
        cout<<"descriptors "<<ii<<"="<<descriptors[ii]<<endl;
    }

    imshow("deer",img0);

    cvWaitKey(0);

    return 0;

}
