#include "CarSegmenter.h"

using namespace cv;
using namespace std;

Mat CarSegmenter::gammaCorrection (const Mat& img, const double gamma) {
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    Mat correctedImg = img.clone();
    LUT(img, lookUpTable, correctedImg);

    return correctedImg;
}

Mat CarSegmenter::convertToGrayscale (const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    return gray;
}

Mat CarSegmenter::equalization (const Mat& img) {
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    Mat equalizedImg;
    clahe->apply(img, equalizedImg);

    return equalizedImg;
}

vector<Mat> CarSegmenter::loadImages(const int sequence) {
    vector<String> fileNames;
    if (sequence == 0)
        glob("../data/sequence0/frames", fileNames);
    else if (sequence == 1)
        glob("../data/sequence1/frames", fileNames);
    else if (sequence == 2)
        glob("../data/sequence2/frames", fileNames);
    else if (sequence == 3)
        glob("../data/sequence3/frames", fileNames);
    else if (sequence == 4)
        glob("../data/sequence4/frames", fileNames);
    else if (sequence == 5)
        glob("../data/sequence5/frames", fileNames);

    vector<Mat> imgs;
    for (const auto& file : fileNames) {
        Mat img = imread(file);
        if (img.empty()) {
            cerr << "Error loading image: " << file << endl;
            continue;
        }
        imgs.push_back(img);
    }

    return imgs;
}

Mat CarSegmenter::createAvgImg (const vector<Mat>& imgVector) {
    // Mat avgImg = Mat::zeros(imgVector[0].size(), CV_32FC3);
    // for (const auto& img : imgVector) {
    //     Mat imgFloat;
    //     img.convertTo(imgFloat, CV_32FC3);
    //     avgImg += imgFloat / imgVector.size();
    // }

    // avgImg.convertTo(avgImg, CV_8UC3);
    
    // return avgImg;

    // Crea un array 3D per immagazzinare i valori pixel per pixel
    vector<Mat> channels[3];
    for(int i = 0; i < 3; i++) 
        channels[i].resize(imgVector.size());

    // Separiamo i canali BGR di ogni immagine
    for(size_t i = 0; i < imgVector.size(); i++) {
        vector<Mat> bgr;
        split(imgVector[i], bgr);
        for(int j = 0; j < 3; j++) {
            channels[j][i] = bgr[j]; // Salviamo i canali separati
        }
    }

    // Calcoliamo la mediana pixel per pixel
    Mat medianImage(imgVector[0].size(), CV_8UC3);
    for (int row = 0; row < medianImage.rows; row++) {
        for (int col = 0; col < medianImage.cols; col++) {
            Vec3b medianPixel;
            for (int ch = 0; ch < 3; ch++) {
                vector<uchar> pixelValues;
                for (size_t i = 0; i < imgVector.size(); i++) {
                    pixelValues.push_back(imgVector[i].at<Vec3b>(row, col)[ch]);
                }
                sort(pixelValues.begin(), pixelValues.end());
                medianPixel[ch] = pixelValues[pixelValues.size() / 2];
            }
            medianImage.at<Vec3b>(row, col) = medianPixel;
        }
    }

    return medianImage;
}

Mat CarSegmenter::preprocessImage (const Mat& img, String type) {
    Mat gray = convertToGrayscale(img);
    if (type == "gamma") return gammaCorrection(gray, 0.5);
    else if (type == "equalize") return equalization(gray);
    else return gray;
}

Mat CarSegmenter::differenceImage (const Mat& empty, const Mat& img) {
    Mat diff;
    absdiff(empty, img, diff);
    Mat thresh;
    threshold(diff, thresh, 15, 255, THRESH_BINARY_INV);

    return thresh;
}

Mat CarSegmenter::analyzeImage (const Mat& img) {
   Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
//    morphologyEx(img, img, MORPH_OPEN, kernel);
   morphologyEx(img, img, MORPH_CLOSE, kernel);

   return img;
}

pair<vector<vector<Point>>, vector<Vec4i>> CarSegmenter::findContoursImg(const Mat& img) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    return {contours, hierarchy};
}

void CarSegmenter::drawContoursImg(Mat& img, const vector<vector<Point>>& contours, const vector<Vec4i>& hierarchy) {
    for (size_t i = 0; i < contours.size(); i++) {
        if (hierarchy[i][3] == -1) {
            drawContours(img, contours, i, Scalar(0, 255, 0), 2, LINE_8, hierarchy);
        }
    }
}


void CarSegmenter::showImages(const Mat& img) {
    imshow("Image", img);
    waitKey(0);
}
