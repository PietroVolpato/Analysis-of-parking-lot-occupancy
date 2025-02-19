#include "BboxSorter.h"
#include <algorithm>
#include <cmath>
#include <cfloat>

using namespace cv;
using namespace std;

// Constructor
BboxSorter::BboxSorter(vector<RotatedRect> rects) : rectangles(std::move(rects)) {}

// Find the bottom-left bounding box
RotatedRect BboxSorter::findBottomLeft() {
    return *min_element(rectangles.begin(), rectangles.end(), [](const RotatedRect& a, const RotatedRect& b) {
        return (a.center.y > b.center.y) || ((a.center.y == b.center.y) && (a.center.x < b.center.x));
    });
}

// Find the next bounding box in the same row
RotatedRect* BboxSorter::findNextInRow(RotatedRect& current, vector<RotatedRect*>& remaining) {
    RotatedRect* next = nullptr;
    float minDist = FLT_MAX;
    const float angleThreshold = 10.0f;

    for (auto& rect : remaining) {
        float dx = rect->center.x - current.center.x;
        float dy = rect->center.y - current.center.y;
        float distance = sqrt(dx * dx + dy * dy);

        // Ensure the rectangle is horizontally aligned within a small vertical margin
        if (abs(dy) < angleThreshold && dx > 0 && distance < minDist) {
            next = rect;
            minDist = distance;
        }
    }

    return next;
}

// Find the first bounding box of the next row
RotatedRect* BboxSorter::findNextRowStart(RotatedRect& current, vector<RotatedRect*>& remaining) {
    RotatedRect* next = nullptr;
    float minDist = FLT_MAX;
    const float perpendicularThreshold = 20.0f;

    for (auto& rect : remaining) {
        float dx = rect->center.x - current.center.x;
        float dy = rect->center.y - current.center.y;
        float distance = sqrt(dx * dx + dy * dy);

        // Ensure it's below the current row and not too far horizontally
        if (dy < 0 && abs(dx) < perpendicularThreshold && distance < minDist) {
            next = rect;
            minDist = distance;
        }
    }

    return next;
}

// Sort the bounding boxes row-wise
vector<RotatedRect> BboxSorter::sort() {
    vector<RotatedRect> sortedRects;
    if (rectangles.empty()) return sortedRects;

    // Find bottom-left starting rectangle
    RotatedRect first = findBottomLeft();
    sortedRects.push_back(first);

    // Convert to pointer list for easier modification
    vector<RotatedRect*> remaining;
    for (auto& r : rectangles) {
        if (!(r.center == first.center)) {
            remaining.push_back(&r);
        }
    }

    RotatedRect* current = &sortedRects.back();

    while (!remaining.empty()) {
        RotatedRect* next = findNextInRow(*current, remaining);

        if (next) {
            sortedRects.push_back(*next);
            remaining.erase(remove(remaining.begin(), remaining.end(), next), remaining.end());
        } else {
            // Move to the next row
            current = findNextRowStart(*sortedRects.rbegin(), remaining);
            if (current) {
                sortedRects.push_back(*current);
                remaining.erase(remove(remaining.begin(), remaining.end(), current), remaining.end());
            } else {
                break; // No more rectangles left
            }
        }

        if (current) current = &sortedRects.back();
    }

    return sortedRects;
}
