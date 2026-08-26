// gen_rects.cpp
// Generates a synthetic rectangle dataset for the STR R-Tree bulk-loader.
//
// Output records are Entry{ Rect mbr; uint64_t child; } = 24 bytes, where
// `child` is the object ID.  IDs are assigned sequentially from 0, so the
// builder can read the file straight into its sort buffers with no transform.
//
// Usage:
//   ./gen_rects <count> [out.bin] [--extent E] [--min-size S] [--max-size S]
//               [--clustered K] [--seed N]
//
// Compile: g++ -O3 -std=c++17 gen_rects.cpp -o gen_rects

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <string>
#include <cstdlib>
#include <cstring>

#include "rect_rtree_format.h"

static std::string human(size_t bytes) {
    const char* u[] = {"B", "KB", "MB", "GB", "TB"};
    int i = 0; double s = (double)bytes;
    while (s >= 1024.0 && i < 4) { s /= 1024.0; i++; }
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << s << " " << u[i];
    return o.str();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr <<
        "Usage: " << argv[0] << " <count> [out.bin] [options]\n"
        "  out.bin defaults to data/rects.bin\n\n"
        "Options:\n"
        "  --extent E      domain is [0,E] x [0,E]        (default 1000)\n"
        "  --min-size S    minimum rectangle side          (default 0.1)\n"
        "  --max-size S    maximum rectangle side          (default 10)\n"
        "  --clustered K   place rectangles around K gaussian clusters\n"
        "                  instead of uniformly (default 0 = uniform)\n"
        "  --seed N        RNG seed                        (default 42)\n";
        return 1;
    }

    size_t count = std::strtoull(argv[1], nullptr, 10);
    if (count == 0) { std::cerr << "count must be > 0\n"; return 1; }

    std::string out_path = "data/rects.bin";
    float extent = 1000.0f, min_size = 0.1f, max_size = 10.0f;
    size_t clusters = 0;
    unsigned seed = 42;

    int i = 2;
    if (i < argc && argv[i][0] != '-') out_path = argv[i++];
    for (; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if      (a == "--extent")    extent   = std::stof(next());
        else if (a == "--min-size")  min_size = std::stof(next());
        else if (a == "--max-size")  max_size = std::stof(next());
        else if (a == "--clustered") clusters = std::strtoull(next(), nullptr, 10);
        else if (a == "--seed")      seed     = (unsigned)std::strtoul(next(), nullptr, 10);
        else { std::cerr << "unknown option: " << a << "\n"; return 1; }
    }

    if (min_size <= 0 || max_size < min_size) {
        std::cerr << "invalid size range\n"; return 1;
    }

    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) { std::cerr << "cannot open " << out_path << " for writing\n"; return 1; }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> pos(0.0f, extent);
    std::uniform_real_distribution<float> size(min_size, max_size);

    // Cluster centres, if requested.  Clustered data is the hard case for STR:
    // uniform data makes every slice equally full and every MBR naturally
    // square, which flatters the algorithm.
    std::vector<float> cx, cy;
    float cluster_sigma = extent / 100.0f;
    for (size_t c = 0; c < clusters; c++) { cx.push_back(pos(rng)); cy.push_back(pos(rng)); }
    std::uniform_int_distribution<size_t> pick(0, clusters ? clusters - 1 : 0);
    std::normal_distribution<float> jitter(0.0f, cluster_sigma);

    constexpr size_t BATCH = 1u << 16;
    std::vector<Entry> batch;
    batch.reserve(BATCH);

    std::cout << "Generating " << count << " rectangles -> " << out_path << "\n"
              << "  domain    [0," << extent << "]^2\n"
              << "  sides     [" << min_size << ", " << max_size << "]\n"
              << "  layout    " << (clusters ? "clustered" : "uniform");
    if (clusters) std::cout << " (" << clusters << " clusters, sigma=" << cluster_sigma << ")";
    std::cout << "\n";

    size_t report = count / 100 ? count / 100 : 1;

    for (size_t n = 0; n < count; n++) {
        float x, y;
        if (clusters) {
            size_t c = pick(rng);
            x = cx[c] + jitter(rng);
            y = cy[c] + jitter(rng);
            if (x < 0) x = 0; if (x > extent) x = extent;
            if (y < 0) y = 0; if (y > extent) y = extent;
        } else {
            x = pos(rng); y = pos(rng);
        }

        float w = size(rng), h = size(rng);
        Entry e;
        e.mbr.min_x = x;
        e.mbr.min_y = y;
        e.mbr.max_x = x + w;
        e.mbr.max_y = y + h;
        e.child     = n;                 // object ID
        batch.push_back(e);

        if (batch.size() == BATCH) {
            out.write(reinterpret_cast<const char*>(batch.data()),
                      batch.size() * sizeof(Entry));
            batch.clear();
        }
        if (n % report == 0 && n > 0)
            std::cout << "\r  " << std::fixed << std::setprecision(1)
                      << (100.0 * n / count) << "%   " << std::flush;
    }
    if (!batch.empty())
        out.write(reinterpret_cast<const char*>(batch.data()),
                  batch.size() * sizeof(Entry));
    out.close();

    std::cout << "\rDone.                \n"
              << "  Records: " << count << " x " << sizeof(Entry) << " B\n"
              << "  Size:    " << human(count * sizeof(Entry)) << "\n";
    return 0;
}
