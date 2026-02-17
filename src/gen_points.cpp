#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cstdlib>
#include <sys/stat.h>
#include <iomanip>

constexpr size_t VECTOR_SIZE_BYTES  = 8192;
constexpr size_t VECTOR_SIZE_FLOATS = VECTOR_SIZE_BYTES / sizeof(float);
constexpr size_t POINT_FLOATS       = 2;
constexpr size_t POINTS_PER_VECTOR  = VECTOR_SIZE_FLOATS / POINT_FLOATS;

size_t file_point_count(const std::string &fname) {
    struct stat st;
    if (stat(fname.c_str(), &st) != 0) return 0; // file doesn't exist
    size_t numVectors = st.st_size / VECTOR_SIZE_BYTES;
    return numVectors * POINTS_PER_VECTOR;
}

size_t get_file_size(const std::string &fname) {
    struct stat st;
    if (stat(fname.c_str(), &st) != 0) return 0;
    return st.st_size;
}

std::string format_size(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <number_of_points> [output.bin]\n"
                  << "  output.bin defaults to data/points.bin\n";
        return 1;
    }

    size_t totalPoints = std::atoll(argv[1]);
    if (totalPoints == 0) {
        std::cerr << "Error: number_of_points must be > 0\n";
        return 1;
    }

    std::string outPath = (argc >= 3) ? argv[2] : "data/points.bin";

    size_t already = file_point_count(outPath);
    if (already >= totalPoints) {
        size_t fileSize = get_file_size(outPath);
        std::cout << "File already contains " << already << " points ("
                  << format_size(fileSize) << "). Nothing to do.\n";
        return 0;
    }

    size_t toGenerate = totalPoints - already;
    size_t numVectors = (toGenerate + POINTS_PER_VECTOR - 1) / POINTS_PER_VECTOR;

    std::ofstream outFile(outPath, std::ios::binary | std::ios::app);
    if (!outFile) {
        std::cerr << "Error opening file for writing!\n";
        return 1;
    }

    // Initialize RNG based on already generated points to avoid repeating sequence
    std::mt19937 rng(42 + already); 
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);

    size_t generated = 0;
    size_t progressInterval = numVectors / 100; // Report progress every 1%
    if (progressInterval == 0) progressInterval = 1;

    std::cout << "Generating " << toGenerate << " points...\n";

    for (size_t v = 0; v < numVectors; v++) {
        std::vector<float> data(VECTOR_SIZE_FLOATS, 0.0f);

        for (size_t p = 0; p < POINTS_PER_VECTOR && generated < toGenerate; p++, generated++) {
            float x = dist(rng);
            float y = dist(rng);

            data[p * POINT_FLOATS + 0] = x;
            data[p * POINT_FLOATS + 1] = y;
        }

        outFile.write(reinterpret_cast<char*>(data.data()), VECTOR_SIZE_BYTES);

        // Progress reporting for large files
        if (v % progressInterval == 0 && v > 0) {
            double progress = (static_cast<double>(v) / numVectors) * 100.0;
            std::cout << "Progress: " << std::fixed << std::setprecision(1) 
                      << progress << "%\r" << std::flush;
        }
    }

    outFile.close();

    size_t finalSize = get_file_size(outPath);
    std::cout << "\nFile updated: " << outPath << "\n"
              << "  Points: " << totalPoints << "\n"
              << "  Size:   " << format_size(finalSize)
              << " (" << finalSize << " bytes)\n";
    
    return 0;
}
