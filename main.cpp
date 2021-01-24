#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/video/tracking.hpp"


#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv/cv_image.h>


#include <vector>
#include<math.h>

using namespace dlib;
using namespace std;
using namespace cv;


void rect_from_face_image(cv::Mat temp, std::vector<cv::Rect>& faces_rect);
void draw_points(Mat& img, Mat landmarks_points, Scalar color = Scalar(255, 0, 0));
void draw_rect(cv::Mat& src, cv::Rect faces);

dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
std::vector<cv::Rect> faces_rect;

int main()
{
    Mat img = imread("image.jpg");

    dlib::shape_predictor pose_model;
    string landmarkFile =  "shape_predictor_68_face_landmarks.dat";
    dlib::deserialize(landmarkFile) >> pose_model;
    try
    {
        if (img.channels() > 3)
            cvtColor(img, img, CV_BGRA2BGR);

        faces_rect.clear();
        rect_from_face_image(img,faces_rect);
        cout<<"No of faces found: "<<faces_rect.size()<<endl;
        if (faces_rect.size() == 0)
        {
            cout << "face_not detected" << endl;
            return 0;
        }


        std::vector<cv::Mat> shapes;
        std::vector<cv::Mat > crop_imgs;
        for(int i=0; i<faces_rect.size(); i++)
        {

            crop_imgs.push_back(img(faces_rect[i]));
            imshow("crop_img",crop_imgs[0]);
            // waitKey(0);
            cout<<" "<<faces_rect[i].x<<" "<<faces_rect[i].y<<" "<<faces_rect[i].width<<" "<<faces_rect[i].height<<endl;
            dlib::rectangle rectan(faces_rect[i].x, faces_rect[i].y, faces_rect[i].x+faces_rect[i].width, faces_rect[i].y+faces_rect[i].height);

            dlib::full_object_detection shape = pose_model(dlib::cv_image<bgr_pixel>(img), rectan);

            Mat *curr_shape1 = new Mat(68, 2, CV_64FC1);
            for (int i = 0; i < 68; i++)
            {
                curr_shape1->at<double>(i, 0) = shape.part(i).x();
                curr_shape1->at<double>(i, 1) = shape.part(i).y();
            }

            curr_shape1->convertTo(*curr_shape1, CV_32FC1);

            Mat landmarks_points = Mat::zeros(68, 2, CV_32FC1);
            landmarks_points = *curr_shape1;
            shapes.push_back(landmarks_points);
            draw_rect(img, faces_rect[i]);
            draw_points(img,landmarks_points,Scalar(255, 0, 0));
            imshow("points",img);
            waitKey(0);
        }

    }
    catch (const char *msg)
    {
        cout << msg << endl;
    }
    catch (std::exception &e)
    {
        cout << "exception occured" << endl;
        cout << e.what() << endl;
    }
}

void draw_rect(cv::Mat& src, cv::Rect faces)
{
    Point p1(faces.x + faces.width, faces.y + faces.height);
    Point p2(faces.x, faces.y);
    Scalar color = cv::Scalar(255, 10, 255);
   // putText(src, "Face",p2,FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0,200,200), 1);
    cv::rectangle(src, p1, p2, color,1,8,0);

}


void draw_points(Mat& img, Mat landmarks_points,Scalar color)
{
    for (int i = 0; i < landmarks_points.rows; ++i)
        circle(img, Point2f(round(landmarks_points.at<float>(i, 0)), round(landmarks_points.at<float>(i, 1))),1, color, -1);
}


void rect_from_face_image(cv::Mat temp, std::vector<cv::Rect>& faces_rect)
{
    dlib::cv_image<dlib::bgr_pixel> cimg(temp);
    /// Detect faces
    //cout<<cimg.size()<<endl;

    std::vector<dlib::rectangle> faces = detector(cimg);
    cv::Rect temp1;
    int rows = temp.rows;
    int cols = temp.cols;
    for (int i = 0; i < faces.size(); ++i)
    {

        temp1.x = faces[i].left();
        temp1.y = faces[i].top();
        temp1.width = faces[i].right() - faces[i].left();
        temp1.height = faces[i].bottom() - faces[i].top();


        faces_rect.push_back(temp1);
    }

}


