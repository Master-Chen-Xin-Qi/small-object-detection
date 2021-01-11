#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <ncnn/net.h>
#include "mobile_sim.id.h"
#define INT_MIN (-INT_MAX - 1)
using namespace std;
//这个函数是官方提供的用于打印输出的tensor
void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}
//打印预测标签图
void print_predict_label(const ncnn::Mat& m)
{
    vector<vector<int>> max_pixel(m.h, vector<int>(m.w, INT_MIN));
    vector<vector<int>> label(m.h, vector<int>(m.w, INT_MIN));
    ofstream OutFile("Test.txt");
    for(int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for(int y=0; y<m.h; y++)
        {
            for(int x=0; x<m.w; x++)
            {
                if(ptr[x]>max_pixel[y][x])
                {
                    max_pixel[y][x] = ptr[x];
                    label[y][x] = q;
                }
            }
            ptr += m.w;
        }
    }
    for(int i=0; i<m.h; i++)
    {
        for(int j=0; j<m.w; j++)
        {
            cout<<label[i][j]<<" ";
            OutFile << label[i][j];
        }
        cout<<endl;
    }
    OutFile.close();
    printf("------------------------\n");
    
}
//main函数模板
int main(int argc, const char* argv[]){
    string rgb_img_path;
    string depth_img_path;
    if(argc == 3)
    {
        rgb_img_path = argv[1];
        depth_img_path = argv[2];
    }
    else
    {
        rgb_img_path = "/home/xinqichen/Downloads/Firefox-Downloads/rgbd_cap/out/rgb/1.png";
        depth_img_path = "/home/xinqichen/Downloads/Firefox-Downloads/rgbd_cap/out/depth/1.png";
    }
    
    cv::Mat rgb_img = cv::imread(rgb_img_path, cv::IMREAD_COLOR);
    cv::Mat depth_img = cv::imread(depth_img_path, cv::IMREAD_COLOR);
    cout<<depth_img<<endl;
    cv::Mat rgb_img2;
    cv::Mat depth_img2;
    int input_width = 320;//转onnx时指定的输入大小
    int input_height = 240;
    // resize
    cv::resize(rgb_img, rgb_img2, cv::Size(input_width, input_height));
    cv::resize(depth_img, depth_img2, cv::Size(input_width, input_height));
    //cout<<"44 done"<<endl;
    //cout<<depth_img2<<endl;
    // 加载转换并且量化后的alexnet网络
    ncnn::Net net;
    //net.opt.num_threads=1;
    net.load_param_bin("/home/xinqichen/Desktop/tools/onnx/mobile_sim.param.bin");
    net.load_model("/home/xinqichen/Desktop/tools/onnx/mobile_sim.bin");
    // 把opencv的mat转换成ncnn的mat
    ncnn::Mat input1 = ncnn::Mat::from_pixels(rgb_img2.data, ncnn::Mat::PIXEL_BGR, rgb_img2.cols, rgb_img2.rows);
    ncnn::Mat input2 = ncnn::Mat::from_pixels(depth_img2.data, ncnn::Mat::PIXEL_BGR, depth_img2.cols, depth_img2.rows);
    cout<<rgb_img2.cols<<endl<<rgb_img2.rows;
    // ncnn前向计算
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(true);
    extractor.input(mobile_sim_param_id::BLOB_rgb_inputs, input1);
    extractor.input(mobile_sim_param_id::BLOB_depth_inputs, input2);
    ncnn::Mat output0;//取决于模型的输出有几个
    extractor.extract(mobile_sim_param_id::BLOB_y, output0);
    print_predict_label(output0);
    //pretty_print(output0);
    
    // 或者展平后输出

    // ncnn::Mat out_flatterned = output0.reshape(output0.w * output0.h * output0.c);
    // std::vector<float> scores;
    // scores.resize(out_flatterned.w);
    // for (int j=0; j<out_flatterned.w; j++)
    // {
    //     scores[j] = out_flatterned[j];
    //     cout<<scores[j]<<" ";
    // }
    cout<<"done"<<endl;
    cout<<output0.c<<endl;
    cout<<output0.w<<endl;
    cout<<output0.h<<endl;
    return 0;
}