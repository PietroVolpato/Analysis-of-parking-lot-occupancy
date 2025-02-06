#include "ParkingSpaceClassifier.h"

using namespace cv;

// Function to create a bounding box
Mat createBoundingBox(const Mat &parkingLotImage, const RotatedRect &rotated_rect) {
    Mat rotation_matrix = getRotationMatrix2D(rotated_rect.center, rotated_rect.angle, 1.0);
        // Rotate the entire image around the bounding box center
        Mat rotated_image;
        warpAffine(parkingLotImage, rotated_image, rotation_matrix, parkingLotImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
        
        // Extract the ROI from the rotated image
        Size rect_size = rotated_rect.size;
        
        // Create a 3x3 full rotation matrix from the 2x3 rotation matrix
        cv::Rect bounding_box(
            static_cast<int>(rotated_rect.center.x - rect_size.width / 2),
            static_cast<int>(rotated_rect.center.y - rect_size.height / 2),
            static_cast<int>(rect_size.width),
            static_cast<int>(rect_size.height)
        );

        // Ensure the bounding box is within the image bounds
        bounding_box &= cv::Rect(0, 0, rotated_image.cols, rotated_image.rows);
        Mat cropped_image;

        if (bounding_box.width > 0 && bounding_box.height > 0) {
            cropped_image = rotated_image(bounding_box);
        }
    return cropped_image;
}

void classifyParkingSpaces(const Mat &parkingLotImage, const Mat &parkingLotEmpty, 
                           std::vector<RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus) {

    Ptr<SIFT> sift = SIFT::create();  // Create SIFT detector
    FlannBasedMatcher matcher;  // FLANN Matcher for SIFT

    Mat grayEmpty, grayCurrent;
    cvtColor(parkingLotEmpty, grayEmpty, COLOR_BGR2GRAY);
    cvtColor(parkingLotImage, grayCurrent, COLOR_BGR2GRAY);

    // Apply CLAHE for better contrast
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(grayEmpty, grayEmpty);
    clahe->apply(grayCurrent, grayCurrent);

    occupancyStatus.clear();

    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        std::cout << "Roi #" << i << std::endl;
        RotatedRect &rotated_rect = parkingSpaces[i];

        // Scale the bounding box by 80%
        Size2f newSize(rotated_rect.size.width * 0.7, rotated_rect.size.height * 0.7);
        RotatedRect scaledRect(rotated_rect.center, newSize, rotated_rect.angle);
        
        Mat roiEmpty = createBoundingBox(grayEmpty, const_cast<const RotatedRect&>(scaledRect));
        Mat roiCurrent = createBoundingBox(grayCurrent, const_cast<const RotatedRect&>(scaledRect));
        //imshow("roi", roiCurrent);
        //waitKey(0);
        std::vector<KeyPoint> keypointsEmpty, keypointsCurrent;
        Mat descriptorsEmpty, descriptorsCurrent;

        // SIFT Feature Detection
        sift->detectAndCompute(roiEmpty, noArray(), keypointsEmpty, descriptorsEmpty);
        sift->detectAndCompute(roiCurrent, noArray(), keypointsCurrent, descriptorsCurrent);

        // Debugging: Print number of detected keypoints
        std::cout << "Parking space: " << keypointsEmpty.size() << " (empty) vs " 
             << keypointsCurrent.size() << " (current)" << std::endl;

        // If no keypoints are found, assume occupied
        if (descriptorsEmpty.empty() || descriptorsCurrent.empty()) {
            std::cout << "No keypoints found in one of the spaces, assuming occupied." << std::endl;
            occupancyStatus.push_back(true);
            continue;
        }


        // KNN Matching (k=2)
        std::vector<std::vector<DMatch>> knnMatches;
        matcher.knnMatch(descriptorsEmpty, descriptorsCurrent, knnMatches, 2);

        // Apply NNDR (Lowe's Ratio Test)
        int goodMatches = 0;
        for (const auto& match : knnMatches) {
            if (match.size() == 2 && match[0].distance < 0.75 * match[1].distance) {
                goodMatches++;
            }
        }

        // Debugging: Show number of good matches
        std::cout << "Good matches: " << goodMatches << std::endl;

        // Classification: Too few matches = occupied
        bool occupied = goodMatches < 2;  // Adjust threshold as needed
        occupancyStatus.push_back(occupied);
    }
}

// Function to draw the parking spaces on the image
void drawParkingSpaces(Mat &image, const std::vector<RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        Point2f vertices[4];
        parkingSpaces[i].points(vertices);
        
        Scalar color = occupancyStatus[i] ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

        for (int j = 0; j < 4; j++)
            line(image, vertices[j], vertices[(j+1)%4], color, 2);
    }
}

void contrastStretching(cv::Mat& input, cv::Mat& output) {
    output = input.clone();  // Clone the input to output
    // Define the (r1, s1) and (r2, s2) points for contrast stretching
    int r1 = 70, s1 = 30;
    int r2 = 170, s2 = 220;
    int L = 256;  // Assuming 8-bit image (0 to 255 intensity levels)

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            int r = input.at<uchar>(i, j);
            int s = 0;

            // Apply the piecewise linear transformation
            if (r <= r1) {
                s = (s1 / (float)r1) * r;
            } else if (r <= r2) {
                s = ((s2 - s1) / (float)(r2 - r1)) * (r - r1) + s1;
            } else {
                s = ((L - 1 - s2) / (float)(L - 1 - r2)) * (r - r2) + s2;
            }

            output.at<uchar>(i, j) = cv::saturate_cast<uchar>(s);
        }
    }
}

