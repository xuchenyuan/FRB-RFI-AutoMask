// apply_mask_onnx.cc
//
// Read a PSRCHIVE .ar file, run an ONNX RFI mask model, and write a new .ar
// with channel weights set to zero for predicted-bad channels.
//
// Expected ONNX model:
//   input  : float32 [1, 4096, 4096]   (channel, time)
//   output : float32 [1, 4096]         (bad-channel score in [0,1])
//
// The preprocessing matches the Python training/inference code:
//   I = first polarization profile data, shaped [chan, subint*bin]
//   robust normalization by median and 1.4826*MAD
//   NaN/Inf -> 0
//
// Usage:
//   apply_mask_onnx input.ar model.onnx output.ar [threshold] [threads]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include <Pulsar/Archive.h>
#include <Pulsar/Integration.h>
#include <Pulsar/Profile.h>

namespace {

constexpr int EXPECT_NCHAN = 4096;
constexpr int EXPECT_NTIME = 4096;
constexpr int VALID_FIRST = 164;
constexpr int VALID_LAST  = 3932;  // exclusive
constexpr float DEFAULT_THRESHOLD = 0.38127f;

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point& t0) {
    const auto t1 = Clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

void usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " input.ar model.onnx output.ar [threshold] [threads]\n\n"
        << "Arguments:\n"
        << "  input.ar    Input PSRCHIVE archive\n"
        << "  model.onnx  ONNX model exported with output=sigmoid score, shape [1,4096]\n"
        << "  output.ar   Output archive with AI channel weights applied\n"
        << "  threshold   Optional; mask channel if score >= threshold"
        << " (default 0.381270)\n"
        << "  threads     Optional ONNXRuntime intra-op thread count, default 1\n\n"
        << "Example:\n"
        << "  " << prog << " burst.ar cnn_bigru.onnx burst.ai.ar\n"
        << "  " << prog << " burst.ar cnn_bigru.onnx burst.ai.ar 0.381270 4\n";
}

float parse_float(const std::string& s, const std::string& name) {
    char* end = nullptr;
    const float v = std::strtof(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0' || !std::isfinite(v)) {
        throw std::runtime_error("bad " + name + ": " + s);
    }
    return v;
}

int parse_int(const std::string& s, const std::string& name) {
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || v <= 0 || v > 4096) {
        throw std::runtime_error("bad " + name + ": " + s);
    }
    return static_cast<int>(v);
}

float median_inplace(std::vector<float>& v) {
    if (v.empty()) {
        return 0.0f;
    }

    const size_t n = v.size();
    const size_t mid = n / 2;

    std::nth_element(v.begin(), v.begin() + mid, v.end());
    const float hi = v[mid];

    if (n % 2 == 1) {
        return hi;
    }

    std::nth_element(v.begin(), v.begin() + mid - 1, v.begin() + mid);
    const float lo = v[mid - 1];
    return 0.5f * (lo + hi);
}

float robust_median(const std::vector<float>& x) {
    std::vector<float> finite;
    finite.reserve(x.size());
    for (float v : x) {
        if (std::isfinite(v)) {
            finite.push_back(v);
        }
    }
    return median_inplace(finite);
}

float robust_sigma_mad(const std::vector<float>& x, float med) {
    std::vector<float> dev;
    dev.reserve(x.size());
    for (float v : x) {
        if (std::isfinite(v)) {
            dev.push_back(std::fabs(v - med));
        }
    }

    float mad = median_inplace(dev);
    float sig = 1.4826f * mad;

    if (!std::isfinite(sig) || sig <= 0.0f) {
        double sum = 0.0;
        double sum2 = 0.0;
        long long n = 0;
        for (float v : x) {
            if (std::isfinite(v)) {
                sum += v;
                sum2 += static_cast<double>(v) * v;
                ++n;
            }
        }
        if (n > 1) {
            const double mean = sum / n;
            const double var = std::max(0.0, sum2 / n - mean * mean);
            sig = static_cast<float>(std::sqrt(var));
        }
    }

    if (!std::isfinite(sig) || sig <= 0.0f) {
        sig = 1.0f;
    }

    return sig;
}

std::vector<float> extract_normalized_I(Pulsar::Archive* ar) {
    const int nsub  = static_cast<int>(ar->get_nsubint());
    const int npol  = static_cast<int>(ar->get_npol());
    const int nchan = static_cast<int>(ar->get_nchan());
    const int nbin  = static_cast<int>(ar->get_nbin());
    const int ntime = nsub * nbin;

    std::cout << "archive shape:"
              << " nsub=" << nsub
              << " npol=" << npol
              << " nchan=" << nchan
              << " nbin=" << nbin
              << " ntime=" << ntime
              << std::endl;

    if (npol < 1) {
        throw std::runtime_error("archive has npol < 1");
    }
    if (nchan != EXPECT_NCHAN) {
        throw std::runtime_error("expected nchan=4096, got " + std::to_string(nchan));
    }
    if (ntime != EXPECT_NTIME) {
        throw std::runtime_error("expected nsub*nbin=4096, got " + std::to_string(ntime));
    }

    std::vector<float> I(static_cast<size_t>(nchan) * ntime, 0.0f);

    // Match Python: data[:, 0, :, :] -> transpose to [chan, subint, bin]
    // -> reshape [chan, nsub*nbin].  We do not change archive state, and use
    // polarization index 0 exactly as the training conversion did.
    for (int isub = 0; isub < nsub; ++isub) {
        Pulsar::Integration* integ = ar->get_Integration(isub);
        for (int ichan = 0; ichan < nchan; ++ichan) {
            const Pulsar::Profile* prof = integ->get_Profile(0, ichan);
            const float* amp = prof->get_amps();
            const size_t base = static_cast<size_t>(ichan) * ntime + static_cast<size_t>(isub) * nbin;
            for (int ibin = 0; ibin < nbin; ++ibin) {
                I[base + ibin] = amp[ibin];
            }
        }
    }

    const float med = robust_median(I);
    const float sig = robust_sigma_mad(I, med);

    for (float& v : I) {
        if (std::isfinite(v)) {
            v = (v - med) / sig;
            if (!std::isfinite(v)) {
                v = 0.0f;
            }
        } else {
            v = 0.0f;
        }
    }

    std::cout << "normalization: median=" << med << " sigma=" << sig << std::endl;
    return I;
}

std::vector<float> run_onnx_model(
    const std::string& onnx_file,
    const std::vector<float>& input,
    int threads
) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cnn_bigru_rfi_mask");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, onnx_file.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    std::string input_name;
    std::string output_name;

#if ORT_API_VERSION >= 13
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    input_name = input_name_alloc.get();
    output_name = output_name_alloc.get();
#else
    char* in_name = session.GetInputName(0, allocator);
    char* out_name = session.GetOutputName(0, allocator);
    input_name = in_name;
    output_name = out_name;
    allocator.Free(in_name);
    allocator.Free(out_name);
#endif

    std::cout << "onnx input name  = " << input_name << std::endl;
    std::cout << "onnx output name = " << output_name << std::endl;

    std::array<int64_t, 3> input_shape{1, EXPECT_NCHAN, EXPECT_NTIME};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ORT does not modify the input data, but the API takes a non-const pointer.
    float* input_ptr = const_cast<float*>(input.data());
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_ptr,
        input.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        throw std::runtime_error("ONNX model did not return a tensor");
    }

    auto info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();
    size_t n_out = info.GetElementCount();

    std::cout << "onnx output shape = [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i + 1 == shape.size() ? "" : ",");
    }
    std::cout << "]" << std::endl;

    if (n_out != EXPECT_NCHAN) {
        throw std::runtime_error("expected ONNX output element count 4096, got " + std::to_string(n_out));
    }

    const float* out = output_tensors[0].GetTensorData<float>();
    return std::vector<float>(out, out + n_out);
}

void apply_mask_and_write(
    Pulsar::Archive* ar,
    const std::vector<float>& score,
    float threshold,
    const std::string& output_file
) {
    const int nsub  = static_cast<int>(ar->get_nsubint());
    const int nchan = static_cast<int>(ar->get_nchan());

    if (nchan != EXPECT_NCHAN || static_cast<int>(score.size()) != EXPECT_NCHAN) {
        throw std::runtime_error("bad nchan/score size in apply_mask_and_write");
    }

    int pred_bad_valid = 0;
    int keep_valid = 0;

    for (int ichan = VALID_FIRST; ichan < VALID_LAST; ++ichan) {
        if (score[ichan] >= threshold) {
            ++pred_bad_valid;
        } else {
            ++keep_valid;
        }
    }

    std::cout << "threshold = " << threshold << std::endl;
    std::cout << "valid channels = " << (VALID_LAST - VALID_FIRST) << std::endl;
    std::cout << "pred_bad_valid = " << pred_bad_valid << std::endl;
    std::cout << "keep_valid     = " << keep_valid << std::endl;
    std::cout << "pred_bad_frac_valid = "
              << static_cast<double>(pred_bad_valid) / (VALID_LAST - VALID_FIRST)
              << std::endl;

    for (int isub = 0; isub < nsub; ++isub) {
        Pulsar::Integration* integ = ar->get_Integration(isub);
        for (int ichan = 0; ichan < nchan; ++ichan) {
            bool keep = false;
            if (ichan >= VALID_FIRST && ichan < VALID_LAST) {
                keep = (score[ichan] < threshold);
            }

            // Preserve any pre-existing zero weights.  For normal dspsr output,
            // old weights are usually 1.  Edge channels are always zeroed.
            const float old_weight = integ->get_weight(ichan);
            const float new_weight = (keep && old_weight > 0.0f) ? old_weight : 0.0f;
            integ->set_weight(ichan, new_weight);
        }
    }

    ar->unload(output_file);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        usage(argv[0]);
        return 2;
    }

    try {
        const std::string input_ar = argv[1];
        const std::string onnx_file = argv[2];
        const std::string output_ar = argv[3];
        const float threshold = (argc >= 5)
            ? parse_float(argv[4], "threshold")
            : DEFAULT_THRESHOLD;
        const int threads = (argc >= 6) ? parse_int(argv[5], "threads") : 1;

        if (threshold < 0.0f || threshold > 1.0f) {
            throw std::runtime_error("threshold should be in [0,1] for score-output ONNX model");
        }

        std::cout << "input_ar  = " << input_ar << std::endl;
        std::cout << "onnx_file = " << onnx_file << std::endl;
        std::cout << "output_ar = " << output_ar << std::endl;
        std::cout << "threshold = " << threshold << std::endl;
        std::cout << "threads   = " << threads << std::endl;

        const auto t_total = Clock::now();

        const auto t_read = Clock::now();
        Reference::To<Pulsar::Archive> ar = Pulsar::Archive::load(input_ar);
        std::cout << "read_archive_time = " << seconds_since(t_read) << " s" << std::endl;

        const auto t_extract = Clock::now();
        std::vector<float> input = extract_normalized_I(ar.ptr());
        std::cout << "extract_preprocess_time = " << seconds_since(t_extract) << " s" << std::endl;

        const auto t_infer = Clock::now();
        std::vector<float> score = run_onnx_model(onnx_file, input, threads);
        std::cout << "onnx_inference_time = " << seconds_since(t_infer) << " s" << std::endl;

        const auto t_write = Clock::now();
        apply_mask_and_write(ar.ptr(), score, threshold, output_ar);
        std::cout << "write_archive_time = " << seconds_since(t_write) << " s" << std::endl;

        std::cout << "total_time = " << seconds_since(t_total) << " s" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
