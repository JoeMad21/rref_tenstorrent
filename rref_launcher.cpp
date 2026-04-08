// rref_launcher.cpp
// Interactive launcher for rref_host.
//
// Prompts the user for each configuration option (with sensible defaults),
// then calls rref_host with the resulting arguments.  Popular configurations
// are stored as named presets so the user can skip the questionnaire entirely.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

// Path to the rref_host binary relative to the working directory.
// Change this if rref_host lives elsewhere on the system.
static constexpr const char* RREF_HOST_EXE = "./rref_host";

// ---------------------------------------------------------------------------
// LaunchConfig — mirrors rref_host's Config, stored as plain strings so we
// can build the argv array without including rref_host headers.
// ---------------------------------------------------------------------------

struct LaunchConfig {
    std::string gen     = "rng";         // "rng" | "file"
    std::string input;                   // path  (only used when gen == "file")
    uint32_t    count   = 1;
    uint32_t    dim     = 32;
    std::string output  = "rref_out.log";
    uint32_t    tiles   = 1;
};

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

struct Preset {
    const char* name;
    const char* description;
    LaunchConfig cfg;
};

// A file-input preset still requires the user to supply the path at runtime;
// the remaining fields carry sensible defaults.
static const Preset PRESETS[] = {
    {
        "Quick Test",
        "4x 32x32 matrices, RNG, 1 core  — sanity check after a build",
        {"rng", "", 4, 32, "quick_test.log", 1}
    },
    {
        "Benchmark Small",
        "16x 64x64 matrices, RNG, 4 cores — light throughput test",
        {"rng", "", 16, 64, "bench_small.log", 4}
    },
    {
        "Benchmark Medium",
        "32x 128x128 matrices, RNG, 8 cores — moderate saturation",
        {"rng", "", 32, 128, "bench_medium.log", 8}
    },
    {
        "Benchmark Large",
        "16x 256x256 matrices, RNG, 8 cores — heavy compute load",
        {"rng", "", 16, 256, "bench_large.log", 8}
    },
    {
        "File Input",
        "8x 64x64 matrices from binary file, 2 cores — prompts for path",
        {"file", "", 8, 64, "file_results.log", 2}
    },
};

static constexpr int NUM_PRESETS = static_cast<int>(sizeof(PRESETS) / sizeof(PRESETS[0]));

// ---------------------------------------------------------------------------
// Terminal helpers
// ---------------------------------------------------------------------------

static void print_separator(char c = '-', int width = 60) {
    std::cout << std::string(width, c) << "\n";
}

static void print_header() {
    print_separator('=');
    std::cout << "   Tenstorrent Wormhole  —  RREF Accelerator Launcher\n";
    print_separator('=');
    std::cout << "\n";
}

// Read a line from stdin; return `fallback` if the user just hits Enter.
static std::string read_line(const std::string& fallback = "") {
    std::string line;
    std::getline(std::cin, line);
    if (!std::cin) {                // EOF / pipe closed
        std::cout << "\n";
        return fallback;
    }
    // Trim leading/trailing whitespace
    auto first = line.find_first_not_of(" \t\r\n");
    auto last  = line.find_last_not_of(" \t\r\n");
    if (first == std::string::npos) return fallback;
    return line.substr(first, last - first + 1);
}

// Prompt for a string value; press Enter to accept the default.
static std::string prompt_string(
    const std::string& question,
    const std::string& default_val = "")
{
    std::cout << question;
    if (!default_val.empty())
        std::cout << " [default: " << default_val << "]";
    std::cout << ": ";
    return read_line(default_val);
}

// Prompt for a positive integer; press Enter to accept the default.
// Retries on invalid or non-positive input.
static uint32_t prompt_uint(
    const std::string& question,
    uint32_t           default_val)
{
    while (true) {
        std::cout << question << " [default: " << default_val << "]: ";
        std::string raw = read_line();
        if (raw.empty()) return default_val;
        try {
            int v = std::stoi(raw);
            if (v >= 1) return static_cast<uint32_t>(v);
        } catch (...) {}
        std::cout << "  Please enter a positive integer.\n";
    }
}

// Prompt for a yes/no answer; returns true for 'y'/'Y'.
static bool prompt_yesno(const std::string& question, bool default_yes = true) {
    std::string hint = default_yes ? "[Y/n]" : "[y/N]";
    std::cout << question << " " << hint << ": ";
    std::string raw = read_line();
    if (raw.empty()) return default_yes;
    return (raw[0] == 'y' || raw[0] == 'Y');
}

// ---------------------------------------------------------------------------
// Config building / display
// ---------------------------------------------------------------------------

static std::vector<std::string> build_args(const LaunchConfig& cfg) {
    std::vector<std::string> args;

    args.push_back("--gen");
    args.push_back(cfg.gen);

    if (cfg.gen == "file") {
        args.push_back("--input");
        args.push_back(cfg.input);
    }

    args.push_back("--count");
    args.push_back(std::to_string(cfg.count));

    args.push_back("--dim");
    args.push_back(std::to_string(cfg.dim));

    args.push_back("--output");
    args.push_back(cfg.output);

    args.push_back("--tiles");
    args.push_back(std::to_string(cfg.tiles));

    return args;
}

static void print_config(const LaunchConfig& cfg) {
    print_separator();
    std::cout << "  Configuration summary:\n\n"
              << "    Matrix source  : " << cfg.gen  << "\n";
    if (cfg.gen == "file")
        std::cout << "    Input file     : " << cfg.input << "\n";
    std::cout << "    Count          : " << cfg.count << " matri"
                                         << (cfg.count == 1 ? "x" : "ces") << "\n"
              << "    Dimension      : " << cfg.dim << "x" << cfg.dim << "\n"
              << "    Tensix cores   : " << cfg.tiles << "\n"
              << "    Output log     : " << cfg.output << "\n";
    print_separator();
}

// ---------------------------------------------------------------------------
// Custom configuration questionnaire
// ---------------------------------------------------------------------------

static LaunchConfig build_custom_config() {
    LaunchConfig cfg;
    std::cout << "\n";

    // --- Gen mode ---
    while (true) {
        std::string ans = prompt_string(
            "Should matrices be generated with RNG or loaded from a binary file? (rng/file)",
            "rng");
        if (ans == "rng" || ans == "file") { cfg.gen = ans; break; }
        std::cout << "  Please enter 'rng' or 'file'.\n";
    }

    if (cfg.gen == "file") {
        while (true) {
            cfg.input = prompt_string("Path to the binary input file (float32, row-major)");
            if (!cfg.input.empty()) break;
            std::cout << "  A file path is required for file mode.\n";
        }
    }

    // --- Count ---
    cfg.count = prompt_uint("How many matrices should be computed?", 1);

    // --- Dimension ---
    cfg.dim = prompt_uint(
        "What is the matrix dimension N (each matrix will be NxN)?", 32);
    if (cfg.dim % 32 != 0)
        std::cout << "  Note: dimensions not divisible by 32 will be padded to the "
                     "next tile boundary on-device.\n";

    // --- Output ---
    cfg.output = prompt_string("Where should results be logged?", "rref_out.log");

    // --- Tiles ---
    uint32_t max_sensible = cfg.count;
    std::cout << "How many Tensix cores should be used in parallel?\n"
              << "  Matrices are split as evenly as possible across cores.\n"
              << "  (max useful = " << max_sensible << " for " << cfg.count << " matri"
              << (cfg.count == 1 ? "x" : "ces") << ")\n";
    cfg.tiles = prompt_uint("  Cores", 1);
    if (cfg.tiles > cfg.count) {
        std::cout << "  Clamping tiles to " << cfg.count
                  << " (no point having more cores than matrices).\n";
        cfg.tiles = cfg.count;
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// Preset menu
// ---------------------------------------------------------------------------

static void print_preset_menu() {
    std::cout << "  Select an option:\n\n"
              << "  [Presets]\n";
    for (int i = 0; i < NUM_PRESETS; i++) {
        std::cout << "   " << (i + 1) << "  "
                  << PRESETS[i].name << "\n"
                  << "       " << PRESETS[i].description << "\n";
    }
    std::cout << "\n  [Other]\n"
              << "   " << (NUM_PRESETS + 1) << "  Custom configuration\n"
              << "   0  Exit\n\n"
              << "> ";
}

// Returns a fully-resolved LaunchConfig, or exits on user request.
// For the file-input preset the user is prompted for the path here.
static LaunchConfig select_config() {
    while (true) {
        print_preset_menu();
        std::string raw = read_line();
        if (raw.empty()) continue;

        int choice = -1;
        try { choice = std::stoi(raw); } catch (...) {}

        if (choice == 0) {
            std::cout << "Goodbye.\n";
            std::exit(0);
        }

        if (choice > 0 && choice <= NUM_PRESETS) {
            LaunchConfig cfg = PRESETS[choice - 1].cfg;

            // File-input preset still needs the actual path
            if (cfg.gen == "file") {
                std::cout << "\n";
                while (true) {
                    cfg.input = prompt_string("Path to the binary input file");
                    if (!cfg.input.empty()) break;
                    std::cout << "  A file path is required.\n";
                }
            }
            return cfg;
        }

        if (choice == NUM_PRESETS + 1)
            return build_custom_config();

        std::cout << "  Invalid choice — please enter a number between 0 and "
                  << (NUM_PRESETS + 1) << ".\n\n";
    }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

// Fork a child process and exec rref_host with the given argument list.
// Returns the child's exit code, or -1 on fork/exec failure.
static int run_rref(const std::vector<std::string>& args) {
    // Build a null-terminated char* argv for execvp
    std::vector<const char*> argv_ptrs;
    argv_ptrs.push_back(RREF_HOST_EXE);
    for (const auto& a : args)
        argv_ptrs.push_back(a.c_str());
    argv_ptrs.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "Error: fork() failed\n";
        return -1;
    }

    if (pid == 0) {
        // Child — replace image with rref_host
        execvp(RREF_HOST_EXE, const_cast<char* const*>(argv_ptrs.data()));
        // execvp only returns on failure
        std::cerr << "Error: could not execute '" << RREF_HOST_EXE
                  << "' — is it compiled and in the current directory?\n";
        std::exit(1);
    }

    // Parent — wait for child to finish
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    print_header();

    LaunchConfig cfg = select_config();

    print_config(cfg);
    std::cout << "\n";

    if (!prompt_yesno("Run rref_host with the above settings?", /*default_yes=*/true)) {
        std::cout << "Cancelled.\n";
        return 0;
    }

    std::cout << "\n";
    print_separator('=');

    auto args = build_args(cfg);
    int  rc   = run_rref(args);

    print_separator('=');
    if (rc == 0)
        std::cout << "rref_host finished successfully.\n";
    else
        std::cout << "rref_host exited with code " << rc << ".\n";

    return rc;
}
