#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string_view>

#include "json_frontend.h"
#include "jsonwrite.h"
#include "log.h"

USING_NEXTPNR_NAMESPACE

namespace {

enum class Stage : uint32_t
{
    Pack,
    Place,
    Route
};

struct Request
{
    const char *device;
    const char *chipdb;
    const char *input;
    const char *output;
    const char *placer;
    const char *router;
    uint64_t seed;
    Stage stop_after;
};

std::mutex nextpnr_mutex;

struct Diagnostics
{
    std::ostringstream stream;

    Diagnostics()
    {
        message_count_by_level.clear();
        had_nonfatal_error = false;
        log_streams.emplace_back(&stream, LogLevel::ERROR_MSG);
    }

    ~Diagnostics() { log_streams.clear(); }
};

int fail(char *error, size_t capacity, std::string_view message) noexcept
{
    if (capacity != 0) {
        const size_t size = std::min(capacity - 1, message.size());
        std::memcpy(error, message.data(), size);
        error[size] = '\0';
    }
    return 1;
}

void run(const Request &request)
{
    ArchArgs args;
    args.device = request.device;
    args.chipdb_override = request.chipdb;
    if (args.chipdb_override.empty())
        log_error("A chipdb path is required.\n");

    Context ctx(args);
    ctx.uarch->init(&ctx);
    ctx.late_init();
    const auto seed_id = ctx.id("seed");
    ctx.rngseed(request.seed);

    if (request.placer)
        ctx.settings[ctx.id("placer")] = std::string(request.placer);
    if (request.router)
        ctx.settings[ctx.id("router")] = std::string(request.router);
    ctx.settings[ctx.id("target_freq")] = std::to_string(12e6);
    ctx.settings[ctx.id("timing_driven")] = true;
    ctx.settings[ctx.id("auto_freq")] = false;
    if (!request.placer)
        ctx.settings[ctx.id("placer")] = Arch::defaultPlacer;
    if (!request.router)
        ctx.settings[ctx.id("router")] = Arch::defaultRouter;
    ctx.settings[ctx.id("arch.name")] = std::string(ctx.archId().c_str(&ctx));
    ctx.settings[ctx.id("arch.type")] = std::string(ctx.archArgsToId(ctx.archArgs()).c_str(&ctx));
    ctx.settings[seed_id] = Property(ctx.rngstate, 64);
    ctx.settings[ctx.id("placerHeap/alpha")] = std::to_string(0.1);
    ctx.settings[ctx.id("placerHeap/beta")] = std::to_string(0.9);
    ctx.settings[ctx.id("placerHeap/criticalityExponent")] = std::to_string(2);
    ctx.settings[ctx.id("placerHeap/timingWeight")] = std::to_string(10);
    ctx.uarch->setupArchContext();

    auto input = open_ifstream_and_log_error(request.input, "design JSON");
    if (!parse_json(input, request.input, &ctx))
        log_error("Loading design failed.\n");
    if (!ctx.pack())
        log_error("Packing design failed.\n");
    ctx.check();
    if (request.stop_after >= Stage::Place) {
        if (!ctx.place())
            log_error("Placing design failed.\n");
        ctx.check();
    }
    if (request.stop_after == Stage::Route && !ctx.route())
        log_error("Routing design failed.\n");
    if (had_nonfatal_error)
        throw log_execution_error_exception();

    std::string output_name = request.output;
    auto output = open_ofstream_and_log_error(output_name, "output JSON");
    if (!write_json_file(output, output_name, &ctx))
        log_error("Saving design failed.\n");
    output.close();
    if (!output)
        log_error("Writing output JSON failed.\n");
}

}

#ifdef _WIN32
#define EISENJOCH_EXPORT __declspec(dllexport)
#else
#define EISENJOCH_EXPORT __attribute__((visibility("default")))
#endif

extern "C" EISENJOCH_EXPORT int eisenjoch_nextpnr_run_v1(const Request *request, char *error, size_t capacity) noexcept
{
    try {
        const std::lock_guard<std::mutex> lock(nextpnr_mutex);
        Diagnostics diagnostics;
        try {
            run(*request);
        } catch (const log_execution_error_exception &) {
            return fail(error, capacity, diagnostics.stream.str());
        }
        return 0;
    } catch (const std::exception &exception) {
        return fail(error, capacity, exception.what());
    } catch (...) {
        return fail(error, capacity, "Unknown nextpnr exception");
    }
}
