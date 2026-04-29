// rref_host.cpp
// Host-side code for Reduced Row Echelon Form on Tenstorrent Wormhole
// Uses the TT-Metalium Mesh API (MeshDevice / MeshBuffer / MeshWorkload)
//
// Algorithm overview:
//   Matrices are processed in batches of `num_tiles`. Within each batch every
//   matrix is assigned to its own Tensix core so all cores execute in parallel.
//   For each pivot column the host (using CPU shadow copies):
//     1. Finds each active matrix's pivot row (partial pivoting)
//     2. Swaps rows in the shadow copy and re-uploads if needed
//     3. Dispatches a single MeshWorkload whose Program spans all active cores,
//        each core scaling its pivot row and eliminating its pivot column

#include <cstdint>
#include <memory>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <random>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>

using namespace tt;
using namespace tt::tt_metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr uint32_t TILE_H     = 32;
static constexpr uint32_t TILE_W     = 32;
static constexpr uint32_t TILE_HW    = TILE_H * TILE_W;
static constexpr uint32_t FACE_H     = 16;
static constexpr uint32_t FACE_W     = 16;
static constexpr uint32_t FACE_HW    = FACE_H * FACE_W;
static constexpr uint32_t TILE_BYTES = TILE_HW * sizeof(uint16_t);
static constexpr float    PIVOT_EPS  = 1e-6f;

// ---------------------------------------------------------------------------
// Config — parsed from command-line arguments
// ---------------------------------------------------------------------------

enum class GenMode { RNG, File };

struct Config {
    GenMode     gen_mode    = GenMode::RNG; // --gen rng|file
    std::string input_file;                 // --input <path>  (required for --gen file)
    uint32_t    count       = 1;            // --count <N>     number of matrices
    uint32_t    dim         = 32;           // --dim <N>       NxN square matrix size
    std::string output_file = "rref_out.log"; // --output <path>
    uint32_t    num_tiles   = 1;            // --tiles <N>     Tensix cores to use in parallel
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n\n"
        << "Options:\n"
        << "  --gen    rng|file   Matrix source: random generation or binary file (default: rng)\n"
        << "  --input  <path>     Input file path (required when --gen file)\n"
        << "                        Format: raw float32, row-major, count*dim*dim values\n"
        << "  --count  <N>        Number of matrices to compute (default: 1)\n"
        << "  --dim    <N>        Dimension N of each NxN matrix (default: 32)\n"
        << "  --output <path>     Log file for results (default: rref_out.log)\n"
        << "  --tiles  <N>        Tensix cores to use in parallel; matrices are\n"
        << "                        distributed evenly across cores (default: 1)\n"
        << "\nExamples:\n"
        << "  " << prog << " --gen rng  --count 16 --dim 64  --output out.log --tiles 4\n"
        << "  " << prog << " --gen file --input mats.bin --count 8 --dim 32 --output out.log --tiles 2\n";
}

static Config parse_args(int argc, char* argv[]) {
    Config cfg;

    auto require_next = [&](int i, const char* flag, int argc_) -> int {
        if (i + 1 >= argc_) {
            std::cerr << "Error: " << flag << " requires a value\n";
            print_usage(argv[0]);
            std::exit(1);
        }
        return i + 1;
    };

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--gen") == 0) {
            i = require_next(i, "--gen", argc);
            if (std::strcmp(argv[i], "rng") == 0)
                cfg.gen_mode = GenMode::RNG;
            else if (std::strcmp(argv[i], "file") == 0)
                cfg.gen_mode = GenMode::File;
            else {
                std::cerr << "Error: --gen must be 'rng' or 'file'\n";
                print_usage(argv[0]);
                std::exit(1);
            }
        } else if (std::strcmp(argv[i], "--input") == 0) {
            i = require_next(i, "--input", argc);
            cfg.input_file = argv[i];
        } else if (std::strcmp(argv[i], "--count") == 0) {
            i = require_next(i, "--count", argc);
            int v = std::atoi(argv[i]);
            if (v < 1) { std::cerr << "Error: --count must be >= 1\n"; std::exit(1); }
            cfg.count = static_cast<uint32_t>(v);
        } else if (std::strcmp(argv[i], "--dim") == 0) {
            i = require_next(i, "--dim", argc);
            int v = std::atoi(argv[i]);
            if (v < 1) { std::cerr << "Error: --dim must be >= 1\n"; std::exit(1); }
            cfg.dim = static_cast<uint32_t>(v);
        } else if (std::strcmp(argv[i], "--output") == 0) {
            i = require_next(i, "--output", argc);
            cfg.output_file = argv[i];
        } else if (std::strcmp(argv[i], "--tiles") == 0) {
            i = require_next(i, "--tiles", argc);
            int v = std::atoi(argv[i]);
            if (v < 1) { std::cerr << "Error: --tiles must be >= 1\n"; std::exit(1); }
            cfg.num_tiles = static_cast<uint32_t>(v);
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Error: unknown option '" << argv[i] << "'\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    // Cross-field validation
    if (cfg.gen_mode == GenMode::File && cfg.input_file.empty()) {
        std::cerr << "Error: --gen file requires --input <path>\n";
        print_usage(argv[0]);
        std::exit(1);
    }
    if (cfg.num_tiles > cfg.count) {
        std::cerr << "Warning: --tiles " << cfg.num_tiles
                  << " exceeds --count " << cfg.count
                  << "; clamping to " << cfg.count << "\n";
        cfg.num_tiles = cfg.count;
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// Matrix generation / loading
// ---------------------------------------------------------------------------

// Generate `count` random NxN matrices with entries uniformly in [-10, 10].
static std::vector<std::vector<float>> generate_rng_matrices(
    uint32_t count, uint32_t dim)
{
    std::mt19937 rng(static_cast<uint32_t>(std::time(nullptr)));
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    uint32_t n = dim * dim;
    std::vector<std::vector<float>> mats(count, std::vector<float>(n));
    for (auto& m : mats)
        for (auto& v : m)
            v = dist(rng);
    return mats;
}

// Load `count` NxN matrices from a flat binary file of float32 values
// (row-major order, count * dim * dim floats total).
static std::vector<std::vector<float>> load_matrices_from_file(
    const std::string& path, uint32_t count, uint32_t dim)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open input file: " + path);

    uint32_t n = dim * dim;
    std::vector<std::vector<float>> mats(count, std::vector<float>(n));
    for (auto& m : mats) {
        f.read(reinterpret_cast<char*>(m.data()),
               static_cast<std::streamsize>(n * sizeof(float)));
        if (!f)
            throw std::runtime_error("Unexpected end of input file: " + path);
    }
    return mats;
}

// ---------------------------------------------------------------------------
// Result logging
// ---------------------------------------------------------------------------

static void write_results_log(
    const std::string& path,
    const Config& cfg,
    const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& results)
{
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open output file: " + path);

    f << "# RREF Results\n"
      << "# gen="    << (cfg.gen_mode == GenMode::RNG ? "rng" : "file")
      << "  count="  << cfg.count
      << "  dim="    << cfg.dim
      << "  tiles="  << cfg.num_tiles
      << "\n\n";

    for (uint32_t m = 0; m < cfg.count; m++) {
        f << "Matrix " << m << " input:\n";
        for (uint32_t r = 0; r < cfg.dim; r++) {
            f << "  ";
            for (uint32_t c = 0; c < cfg.dim; c++)
                f << std::setw(10) << std::fixed << std::setprecision(4)
                  << inputs[m][r * cfg.dim + c];
            f << "\n";
        }
        f << "Matrix " << m << " RREF:\n";
        for (uint32_t r = 0; r < cfg.dim; r++) {
            f << "  ";
            for (uint32_t c = 0; c < cfg.dim; c++)
                f << std::setw(10) << std::fixed << std::setprecision(4)
                  << results[m][r * cfg.dim + c];
            f << "\n";
        }
        f << "\n";
    }
}

// ---------------------------------------------------------------------------
// Tile layout helpers
//
// TT-Metalium uses face-major layout for 32x32 tiles:
//   Face 0: rows  0-15, cols  0-15
//   Face 1: rows  0-15, cols 16-31
//   Face 2: rows 16-31, cols  0-15
//   Face 3: rows 16-31, cols 16-31
// Within each face elements are row-major bfloat16.
// Two bfloat16 values are packed per uint32_t (lo = col even, hi = col odd).
// ---------------------------------------------------------------------------

static inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

static uint32_t tile_element_u32_idx(uint32_t r, uint32_t c) {
    uint32_t face_id      = (r / FACE_H) * 2 + (c / FACE_W);
    uint32_t elem_in_face = (r % FACE_H) * FACE_W + (c % FACE_W);
    return face_id * (FACE_HW / 2) + elem_in_face / 2;
}

static bool tile_element_is_hi(uint32_t c) { return (c % 2) == 1; }

static std::vector<uint32_t> pack_tiles(
    const std::vector<float>& mat, uint32_t rows, uint32_t cols)
{
    uint32_t nt_r = ceil_div(rows, TILE_H), nt_c = ceil_div(cols, TILE_W);
    std::vector<uint32_t> out(nt_r * nt_c * (TILE_HW / 2), 0u);

    for (uint32_t tr = 0; tr < nt_r; tr++) {
        for (uint32_t tc = 0; tc < nt_c; tc++) {
            uint32_t base = (tr * nt_c + tc) * (TILE_HW / 2);
            for (uint32_t lr = 0; lr < TILE_H; lr++) {
                for (uint32_t lc = 0; lc < TILE_W; lc++) {
                    uint32_t gr = tr * TILE_H + lr, gc = tc * TILE_W + lc;
                    float v = (gr < rows && gc < cols) ? mat[gr * cols + gc] : 0.0f;
                    uint16_t bf  = static_cast<uint16_t>(pack_two_bfloat16_into_uint32({bfloat16(v), bfloat16(0.0f)}) & 0xFFFFu);
                    uint32_t idx = tile_element_u32_idx(lr, lc);
                    if (tile_element_is_hi(lc % FACE_W))
                        out[base + idx] = (out[base + idx] & 0x0000FFFFu) | (static_cast<uint32_t>(bf) << 16);
                    else
                        out[base + idx] = (out[base + idx] & 0xFFFF0000u) | bf;
                }
            }
        }
    }
    return out;
}

static std::vector<float> unpack_tiles(
    const std::vector<uint32_t>& tiles, uint32_t rows, uint32_t cols)
{
    uint32_t nt_r = ceil_div(rows, TILE_H), nt_c = ceil_div(cols, TILE_W);
    std::vector<float> out(rows * cols, 0.0f);

    for (uint32_t tr = 0; tr < nt_r; tr++) {
        for (uint32_t tc = 0; tc < nt_c; tc++) {
            uint32_t base = (tr * nt_c + tc) * (TILE_HW / 2);
            for (uint32_t lr = 0; lr < TILE_H; lr++) {
                for (uint32_t lc = 0; lc < TILE_W; lc++) {
                    uint32_t gr = tr * TILE_H + lr, gc = tc * TILE_W + lc;
                    if (gr >= rows || gc >= cols) continue;
                    uint32_t word = tiles[base + tile_element_u32_idx(lr, lc)];
                    uint16_t bf   = tile_element_is_hi(lc % FACE_W)
                                    ? static_cast<uint16_t>(word >> 16)
                                    : static_cast<uint16_t>(word & 0xFFFFu);
                    out[gr * cols + gc] = [&]{ uint32_t u = static_cast<uint32_t>(bf) << 16; float fv; std::memcpy(&fv, &u, sizeof(fv)); return fv; }();
                }
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// RREFAccelerator
// ---------------------------------------------------------------------------

class RREFAccelerator {
public:
    // num_tiles controls how many Tensix cores work in parallel.
    // Matrices are distributed in round-robin batches of num_tiles.
    RREFAccelerator(uint32_t device_id, uint32_t num_tiles)
        : mesh_device_(distributed::MeshDevice::create_unit_mesh(device_id)),
          cq_(mesh_device_->mesh_command_queue()),
          device_range_(distributed::MeshCoordinateRange(mesh_device_->shape())),
          num_tiles_(num_tiles)
    {}

    ~RREFAccelerator() { mesh_device_->close(); }

    RREFAccelerator(const RREFAccelerator&)            = delete;
    RREFAccelerator& operator=(const RREFAccelerator&) = delete;

    // Compute RREF for every matrix in `matrices` (each dim×dim, row-major).
    // Matrices are processed in batches of num_tiles; within each batch every
    // matrix runs on its own Tensix core so they all execute concurrently.
    std::vector<std::vector<float>> compute_all(
        const std::vector<std::vector<float>>& matrices,
        uint32_t dim)
    {
        uint32_t total = static_cast<uint32_t>(matrices.size());
        std::vector<std::vector<float>> results(total);

        for (uint32_t start = 0; start < total; start += num_tiles_) {
            uint32_t end        = std::min(start + num_tiles_, total);
            uint32_t batch_size = end - start;

            std::cout << "Processing matrices " << start << "-" << (end - 1)
                      << " on " << batch_size << " core(s)\n";

            // Collect this batch
            std::vector<std::vector<float>> batch(
                matrices.begin() + start, matrices.begin() + end);

            auto batch_results = process_batch(batch, dim, dim);

            for (uint32_t i = 0; i < batch_size; i++)
                results[start + i] = std::move(batch_results[i]);
        }
        return results;
    }

private:
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    distributed::MeshCommandQueue&           cq_;
    distributed::MeshCoordinateRange         device_range_;
    uint32_t                                 num_tiles_;

    // -----------------------------------------------------------------------
    // Shadow helpers (pure CPU)
    // -----------------------------------------------------------------------

    static void swap_rows_shadow(
        std::vector<float>& m, uint32_t r0, uint32_t r1, uint32_t cols)
    {
        for (uint32_t c = 0; c < cols; c++)
            std::swap(m[r0 * cols + c], m[r1 * cols + c]);
    }

    static void apply_elimination_shadow(
        std::vector<float>& m,
        uint32_t rows, uint32_t cols,
        uint32_t pivot_row, uint32_t pivot_col, float scale)
    {
        for (uint32_t c = 0; c < cols; c++)
            m[pivot_row * cols + c] *= scale;
        for (uint32_t r = 0; r < rows; r++) {
            if (r == pivot_row) continue;
            float factor = m[r * cols + pivot_col];
            if (std::abs(factor) < PIVOT_EPS) continue;
            for (uint32_t c = 0; c < cols; c++)
                m[r * cols + c] -= factor * m[pivot_row * cols + c];
        }
    }

    // -----------------------------------------------------------------------
    // Process one batch of up to num_tiles matrices concurrently.
    // Each matrix in the batch is assigned to Tensix core {i, 0}.
    // -----------------------------------------------------------------------

    std::vector<std::vector<float>> process_batch(
        const std::vector<std::vector<float>>& batch_mats,
        uint32_t rows, uint32_t cols)
    {
        uint32_t batch_size = static_cast<uint32_t>(batch_mats.size());
        uint32_t nt_r = ceil_div(rows, TILE_H);
        uint32_t nt_c = ceil_div(cols, TILE_W);
        uint32_t buf_bytes = nt_r * nt_c * (TILE_HW / 2) * sizeof(uint32_t);

        // Allocate one DRAM MeshBuffer per matrix and upload
        distributed::ReplicatedBufferConfig  rep_cfg{.size = buf_bytes};
        distributed::DeviceLocalBufferConfig local_cfg{
            .page_size   = buf_bytes,
            .buffer_type = BufferType::DRAM,
        };

        std::vector<std::shared_ptr<distributed::MeshBuffer>> bufs(batch_size);
        for (uint32_t b = 0; b < batch_size; b++) {
            bufs[b] = distributed::MeshBuffer::create(rep_cfg, local_cfg, mesh_device_.get());
            auto packed = pack_tiles(batch_mats[b], rows, cols);
            distributed::EnqueueWriteMeshBuffer(cq_, bufs[b], packed, /*blocking=*/false);
        }

        // CPU shadow copies — used for pivot searching and row-swap re-uploads.
        // No round-trip reads are needed during the RREF iteration.
        std::vector<std::vector<float>> shadows = batch_mats;

        // Per-matrix pivot row cursor
        std::vector<uint32_t> pivot_rows(batch_size, 0u);

        // RREF main loop: advance all matrices column by column together
        for (uint32_t col = 0; col < cols; col++) {
            // Gather the subset of matrices that have an active pivot this step
            struct ActiveEntry {
                uint32_t batch_idx;
                uint32_t pivot_row;
                float    scale;
            };
            std::vector<ActiveEntry> active;
            active.reserve(batch_size);

            for (uint32_t b = 0; b < batch_size; b++) {
                if (pivot_rows[b] >= rows) continue;

                auto& shadow = shadows[b];
                uint32_t pr  = pivot_rows[b];

                // Partial pivot search
                uint32_t max_row = pr;
                float    max_val = std::abs(shadow[pr * cols + col]);
                for (uint32_t r = pr + 1; r < rows; r++) {
                    float v = std::abs(shadow[r * cols + col]);
                    if (v > max_val) { max_val = v; max_row = r; }
                }

                if (max_val < PIVOT_EPS) continue; // no pivot in this column

                // Row swap if needed
                if (max_row != pr) {
                    swap_rows_shadow(shadow, pr, max_row, cols);
                    auto repacked = pack_tiles(shadow, rows, cols);
                    distributed::EnqueueWriteMeshBuffer(cq_, bufs[b], repacked, /*blocking=*/false);
                }

                active.push_back({b, pr, 1.0f / shadow[pr * cols + col]});
            }

            if (active.empty()) continue;

            // Build argument lists for the multi-core dispatch
            std::vector<std::shared_ptr<distributed::MeshBuffer>> active_bufs;
            std::vector<uint32_t> prows, pcols;
            std::vector<float>    scales;
            active_bufs.reserve(active.size());
            prows.reserve(active.size());
            pcols.reserve(active.size());
            scales.reserve(active.size());

            for (auto& e : active) {
                active_bufs.push_back(bufs[e.batch_idx]);
                prows.push_back(e.pivot_row);
                pcols.push_back(col);
                scales.push_back(e.scale);
            }

            dispatch_elimination_batch(
                active_bufs, rows, cols, nt_r, nt_c, prows, pcols, scales);

            // Mirror the kernel's work in the shadow copies
            for (auto& e : active) {
                apply_elimination_shadow(
                    shadows[e.batch_idx], rows, cols, e.pivot_row, col, e.scale);
                pivot_rows[e.batch_idx]++;
            }
        }

        // Read back all results; only the last read is blocking so we flush
        // the queue while keeping earlier reads queued in order.
        std::vector<std::vector<float>> results(batch_size);
        for (uint32_t b = 0; b < batch_size; b++) {
            std::vector<uint32_t> raw;
            bool blocking = (b == batch_size - 1);
            distributed::EnqueueReadMeshBuffer(cq_, raw, bufs[b], blocking);
            results[b] = unpack_tiles(raw, rows, cols);
        }
        return results;
    }

    // -----------------------------------------------------------------------
    // Dispatch one elimination step across a variable-size set of cores.
    //
    // active_bufs[i]  ← DRAM buffer for the i-th active matrix
    // Core {i, 0}     ← handles matrix i
    //
    // Kernel responsibilities (same as before, now replicated across cores):
    //   rref_reader.cpp  (RISCV_0): DRAM tiles → CB::c_in0 (current row)
    //                                          → CB::c_in1 (pivot row, once)
    //   rref_compute.cpp (Tensix) : scale pivot row; FMA-eliminate other rows
    //   rref_writer.cpp  (RISCV_1): result tiles → DRAM
    //
    // NoC addresses are resolved inside the kernels via get_noc_addr().
    // -----------------------------------------------------------------------

    void dispatch_elimination_batch(
        const std::vector<std::shared_ptr<distributed::MeshBuffer>>& active_bufs,
        uint32_t rows, uint32_t cols,
        uint32_t nt_r, uint32_t nt_c,
        const std::vector<uint32_t>& pivot_rows,
        const std::vector<uint32_t>& pivot_cols,
        const std::vector<float>&    scales)
    {
        uint32_t n = static_cast<uint32_t>(active_bufs.size());
        Program  program = CreateProgram();

        // Cores are laid out along the first row: {0,0} … {n-1, 0}
        CoreRange active_range({0, 0}, {n - 1, 0});

        // Circular buffers are created once for the whole range; every core
        // gets its own private copy of the CB L1 space automatically.
        auto make_cb = [&](CB cb_id, uint32_t num_tiles_per_core) {
            CircularBufferConfig cfg(
                num_tiles_per_core * TILE_BYTES,
                {{cb_id, tt::DataFormat::Float16_b}});
            cfg.set_page_size(cb_id, TILE_BYTES);
            CreateCircularBuffer(program, active_range, cfg);
        };

        make_cb(CB::c_in0,  2 * nt_c); // double-buffered: current row
        make_cb(CB::c_in1,      nt_c); // pivot row (loaded once per step)
        make_cb(CB::c_out0, 2 * nt_c); // double-buffered: output

        // Create kernels once for the full core range
        auto reader = CreateKernel(
            program,
            "kernels/rref/rref_reader.cpp",
            active_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc       = NOC::RISCV_0_default,
            });

        auto compute = CreateKernel(
            program,
            "kernels/rref/rref_compute.cpp",
            active_range,
            ComputeConfig{
                .math_fidelity    = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .math_approx_mode = false,
            });

        auto writer = CreateKernel(
            program,
            "kernels/rref/rref_writer.cpp",
            active_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc       = NOC::RISCV_1_default,
            });

        // Set per-core runtime args — each core gets its own matrix's address
        for (uint32_t i = 0; i < n; i++) {
            CoreCoord core     = {i, 0};
            uint32_t  dram_addr = active_bufs[i]->address();
            bfloat16  bf_scale(scales[i]);

            // Reader: dram_addr, nt_r, nt_c, pivot_row, pivot_tile_col
            SetRuntimeArgs(program, reader, core, std::vector<uint32_t>{
                dram_addr,
                nt_r,
                nt_c,
                pivot_rows[i],
                pivot_cols[i] / TILE_W,
            });

            // Compute: rows, nt_c, pivot_row, pivot_local_col, scale (bf16)
            SetRuntimeArgs(program, compute, core, std::vector<uint32_t>{
                rows,
                nt_c,
                pivot_rows[i],
                pivot_cols[i] % TILE_W,
                static_cast<uint32_t>(pack_two_bfloat16_into_uint32({bf_scale, bfloat16(0.0f)}) & 0xFFFFu),
            });

            // Writer: dram_addr, nt_r, nt_c
            SetRuntimeArgs(program, writer, core, std::vector<uint32_t>{
                dram_addr,
                nt_r,
                nt_c,
            });
        }

        // Wrap in a MeshWorkload and enqueue — non-blocking for throughput
        distributed::MeshWorkload workload;
        workload.add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(cq_, workload, /*blocking=*/false);
    }
};

// ---------------------------------------------------------------------------
// Utility: pretty-print a matrix to stdout
// ---------------------------------------------------------------------------

static void print_matrix(
    const std::vector<float>& m, uint32_t dim, const char* label)
{
    std::cout << label << ":\n";
    for (uint32_t r = 0; r < dim; r++) {
        std::cout << "  [";
        for (uint32_t c = 0; c < dim; c++) {
            std::cout << std::setw(9) << std::fixed << std::setprecision(4)
                      << m[r * dim + c];
            if (c + 1 < dim) std::cout << "  ";
        }
        std::cout << " ]\n";
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    std::cout << "Config:\n"
              << "  gen    = " << (cfg.gen_mode == GenMode::RNG ? "rng" : "file") << "\n"
              << "  count  = " << cfg.count      << "\n"
              << "  dim    = " << cfg.dim << "x"  << cfg.dim << "\n"
              << "  output = " << cfg.output_file << "\n"
              << "  tiles  = " << cfg.num_tiles   << "\n\n";

    // Generate or load matrices
    std::vector<std::vector<float>> matrices;
    try {
        matrices = (cfg.gen_mode == GenMode::RNG)
            ? generate_rng_matrices(cfg.count, cfg.dim)
            : load_matrices_from_file(cfg.input_file, cfg.count, cfg.dim);
    } catch (const std::exception& e) {
        std::cerr << "Error loading matrices: " << e.what() << "\n";
        return 1;
    }

    // Echo first matrix to stdout so the user can sanity-check
    if (cfg.count >= 1)
        print_matrix(matrices[0], cfg.dim, "Matrix 0 input");

    // Run RREF
    std::vector<std::vector<float>> results;
    try {
        RREFAccelerator rref(/*device_id=*/0, cfg.num_tiles);
        results = rref.compute_all(matrices, cfg.dim);
    } catch (const std::exception& e) {
        std::cerr << "Error during RREF: " << e.what() << "\n";
        return 1;
    }

    if (cfg.count >= 1)
        print_matrix(results[0], cfg.dim, "Matrix 0 RREF result");

    // Write full results to log file
    try {
        write_results_log(cfg.output_file, cfg, matrices, results);
        std::cout << "\nResults written to " << cfg.output_file << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error writing log: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
