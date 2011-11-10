//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


#define IPL_GAUSSIAN_5x5 5
//____________________________________________________________________________________

int g_slider_position = 0;
CvCapture *g_capture  = NULL;

//____________________________________________________________________________________

IplImage* doPyrDown(
		IplImage* in,
		int filter = IPL_GAUSSIAN_5x5
) {

	// Best to make sure input image is divisible by two.
	//
	assert( in->width%2 == 0 && in->height%2 == 0 );

	IplImage* out = cvCreateImage(
			cvSize( in->width/2, in->height/2 ),
			in->depth,
			in->nChannels
	);
	cvPyrDown( in, out );
	return( out );
}

//____________________________________________________________________________________

IplImage* doCanny(
		IplImage* in,
		double    lowThresh,
		double    highThresh,
		double    aperture
) {
	if(in->nChannels != 1)
		return(0); //Canny only handles gray scale images

	IplImage* out = cvCreateImage(
			cvGetSize( in ),
			in->depth, //IPL_DEPTH_8U,
			1);

	cvCanny( in, out, lowThresh, highThresh, aperture );
	return( out );
}

//____________________________________________________________________________________

void onTrackbarSlide(int pos)
{
	cvSetCaptureProperty(
			g_capture,
			CV_CAP_PROP_POS_FRAMES,
			pos
	);
}

//____________________________________________________________________________________

String cascadeName = "haarcascade_frontalface_alt.xml";
String nestedCascadeName = "haarcascade_eye_tree_eyeglasses.xml";

//____________________________________________________________________________________

void detectAndDraw( Mat& img,
		CascadeClassifier& cascade, CascadeClassifier& nestedCascade,
		double scale)
{
	int i = 0;
	double t = 0;
	vector<Rect> faces;
	const static Scalar colors[] =  { CV_RGB(0,0,255),
			CV_RGB(0,128,255),
			CV_RGB(0,255,255),
			CV_RGB(0,255,0),
			CV_RGB(255,128,0),
			CV_RGB(255,255,0),
			CV_RGB(255,0,0),
			CV_RGB(255,0,255)} ;
	Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

	cvtColor( img, gray, CV_BGR2GRAY );
	resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
	equalizeHist( smallImg, smallImg );

	t = (double)cvGetTickCount();
	cascade.detectMultiScale( smallImg, faces,
			1.1, 2, 0
			//|CV_HAAR_FIND_BIGGEST_OBJECT
			//|CV_HAAR_DO_ROUGH_SEARCH
			|CV_HAAR_SCALE_IMAGE
			,
			Size(30, 30) );
	t = (double)cvGetTickCount() - t;
	printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
	for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
	{
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i%8];
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);
		circle( img, center, radius, color, 3, 8, 0 );
		if( nestedCascade.empty() )
			continue;
		smallImgROI = smallImg(*r);
		nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
				1.1, 2, 0
				//|CV_HAAR_FIND_BIGGEST_OBJECT
				//|CV_HAAR_DO_ROUGH_SEARCH
				//|CV_HAAR_DO_CANNY_PRUNING
				|CV_HAAR_SCALE_IMAGE
				,
				Size(30, 30) );
		for( vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ )
		{
			center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
			center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
			radius = cvRound((nr->width + nr->height)*0.25*scale);
			circle( img, center, radius, color, 3, 8, 0 );
		}
	}
	cv::imshow( "result", img );
}

//____________________________________________________________________________________

int main(int argc, char *argv[])
{
	cvNamedWindow("Video1");
	cvNamedWindow("Post");

	CascadeClassifier cascade, nestedCascade;
	double scale = 1;

	//    IplImage *out = NULL;

	//    g_capture = cvCreateCameraCapture(0);
	g_capture = cvCreateFileCapture(argv[1]);

	int frames = (int) cvGetCaptureProperty(
			g_capture,
			CV_CAP_PROP_FRAME_COUNT
	);

	int width = (int) cvGetCaptureProperty(
			g_capture,
			CV_CAP_PROP_FRAME_WIDTH
	);
	int height = (int) cvGetCaptureProperty(
			g_capture,
			CV_CAP_PROP_FRAME_HEIGHT
	);

	double fps = cvGetCaptureProperty(g_capture,
			CV_CAP_PROP_FPS
	);

	if(frames != 0)
	{
		cvCreateTrackbar(
				"Position",
				"Video1",
				&g_slider_position,
				frames,
				onTrackbarSlide
		);
	}
	if( !cascade.load( cascadeName ) )
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}
	if( !nestedCascade.load( nestedCascadeName ) )
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
	//    CvVideoWriter* writer = cvCreateVideoWriter("record.avi", CV_FOURCC('I', 'Y', 'U', 'V'), fps, cvSize( width, height ));
	IplImage *frame;
	while(1)
	{
		frame = cvQueryFrame(g_capture);
		if(!frame) break;
		IplImage *out;
		//	cvSmooth( frame, out, CV_GAUSSIAN, 3, 3 );
		out = doPyrDown( frame, IPL_GAUSSIAN_5x5 );
		//	out = doPyrDown( out, IPL_GAUSSIAN_5x5 );
		//	out = doCanny( out, 10, 100, 3 );

		// if( frame.empty() )
		//   break;
		// if( iplImg->origin == IPL_ORIGIN_TL )
		//   frame.copyTo( frameCopy );
		// else
		//   flip( frame, frameCopy, 0 );

		Mat frameCopy = out;
		detectAndDraw(frameCopy, cascade, nestedCascade, scale);
		//CV_FOURCC('D', 'M', '4', 'V')

		// if(writer == NULL)
		// {
		//     cout<<"Nothing to write"<<endl;
		//     exit(0);
		// }
		//	IplImage w = frameCopy;
		//	cout<<"Before writing"<<endl;
		//	cvWriteFrame(writer,  &w);
		//	cvShowImage("Video1", frame);
		//	cvShowImage("Post", out);
		char c = cvWaitKey(33);
		cvReleaseImage(&out);
		if(c == 27) break;
		g_slider_position++;
		//	cvSetTrackbarPos("Position", "Video1", g_slider_position);
	}
	cvReleaseCapture(&g_capture);
	//    cvReleaseVideoWriter(&writer);
	cvDestroyWindow("Video1");
	cvDestroyWindow("Post");

	return 0;
}




/*
  Getting into a building - vibrate
  Other heads - vibrate
  optic flow
  Screen brightness  
 */


