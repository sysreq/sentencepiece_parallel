// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#include "unigram_model_trainer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "filesystem.h"
#include "normalizer.h"
#include "pretokenizer_for_training.h"
#include "sentencepiece_trainer.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_replace.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/esaxx/esa.hxx"  // Suffix array library.
#include "trainer_interface.h"
#include "unicode_script.h"
#include "util.h"

namespace sentencepiece {
namespace unigram {
namespace {

constexpr char32 kSentenceBoundary = 0x0000;

double Digamma(double x) {
  double result = 0.0;
  for (; x < 7; ++x) result -= 1 / x;
  x -= 1.0 / 2.0;
  const double xx = 1.0 / x;
  const double xx2 = xx * xx;
  const double xx4 = xx2 * xx2;
  result += std::log(x) + (1.0 / 24.0) * xx2 - (7.0 / 960.0) * xx4 +
            (31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4;
  return result;
}

template <typename IT>
void ToLogProb(IT begin, IT end) {
  float sum = 0.0;
  for (auto it = begin; it != end; ++it) {
    sum += it->second;
  }
  float logsum = std::log(static_cast<double>(sum));
  for (auto it = begin; it != end; ++it) {
    it->second = std::log(static_cast<double>(it->second)) - logsum;
  }
}

template <class T>
class BoundedPriorityQueue {
 public:
  explicit BoundedPriorityQueue(size_t size) : size_(size) {}
  ~BoundedPriorityQueue() = default;

  void push(T elem, int64_t score) {
    if (queue_.size() > 4 * size_) resize();
    if (sorted && queue_.size() >= size_ && queue_[size_ - 1].second > score)
      return;
    queue_.emplace_back(elem, score);
  }

  const std::vector<std::pair<T, int64_t>> &get() {
    resize();
    return queue_;
  }

 private:
  void resize() {
    std::sort(queue_.begin(), queue_.end(), [](const auto &p1, const auto &p2) {
      return (p1.second > p2.second ||
              (p1.second == p2.second && p1.first < p2.first));
    });
    sorted = true;
    if (queue_.size() > size_) queue_.resize(size_);
  }

  bool sorted = false;
  size_t size_ = 0;
  std::vector<std::pair<T, int64_t>> queue_;
};

}  // namespace

TrainerModel::TrainerModel(const TrainerSpec &trainer_spec,
                           const NormalizerSpec &normalizer_spec)
    : trainer_spec_(trainer_spec), normalizer_spec_(normalizer_spec) {}

TrainerModel::~TrainerModel() {}

const TrainerModel::SentencePieces &TrainerModel::GetSentencePieces() const {
  return sentencepieces_;
}

void TrainerModel::SetSentencePieces(SentencePieces &&sentencepieces) {
  sentencepieces_ = std::move(sentencepieces);
  CHECK(!sentencepieces_.empty());

  min_score_ = FLT_MAX;
  model_proto_data_.Clear();
  model_proto_ = &model_proto_data_;
  std::vector<std::pair<absl::string_view, int>> pieces;

  for (size_t i = 0; i < sentencepieces_.size(); ++i) {
    const absl::string_view w = sentencepieces_[i].first;  // piece
    const float score = sentencepieces_[i].second;         // score.
    CHECK(!std::isnan(score));
    pieces.emplace_back(w, i);
    min_score_ = std::min(min_score_, score);
    auto *piece = model_proto_data_.add_pieces();
    piece->set_piece(w.data(), w.size());
    piece->set_score(score);
  }

  BuildTrie(&pieces);
  CHECK(status().ok());
}

TrainerModel::SentencePieces Trainer::MakeSeedSentencePieces() {
  return trainer_spec_.train_extremely_large_corpus()
             ? MakeSeedSentencePiecesInternal<int64_t>()
             : MakeSeedSentencePiecesInternal<int32_t>();
}

// Returns seed sentencepieces for EM training.
template <typename node_int_type>
TrainerModel::SentencePieces Trainer::MakeSeedSentencePiecesInternal() {
  CHECK(!sentences_.empty());
  CHECK(!required_chars_.empty());

  // Pretokenizer applied only in training time.
  // Pretokenizer is used as a constraint of piece extractions.
  const auto *pretokenizer = SentencePieceTrainer::GetPretokenizerForTraining();

  auto pretokenize_or_rewrite = [&](std::pair<std::string, int64_t> *w) {
    if (pretokenizer) {
      std::vector<char32> chars;
      for (const auto &w : pretokenizer->PreTokenize(w->first)) {
        for (const auto &c : string_util::UTF8ToUnicodeText(w)) {
          chars.push_back(c);
        }
        chars.push_back(kSentenceBoundary);
      }
      return chars;
    } else if (!trainer_spec_.pretokenization_delimiter().empty()) {
      // When delimiter is specified, tokenize the input with the delimiter.
      // For EM training, we assume that the delimiter doesn't exist and
      // rewrite the original sentence.
      std::vector<char32> chars;
      absl::string_view delimiter = trainer_spec_.pretokenization_delimiter();
      for (const auto &w : absl::StrSplit(w->first, delimiter)) {
        for (const auto &c : string_util::UTF8ToUnicodeText(w)) {
          chars.push_back(c);
        }
        chars.push_back(kSentenceBoundary);
      }
      // Removes the delimiter.
      w->first = absl::StrReplaceAll(w->first, {{delimiter, ""}});
      return chars;
    }
    return string_util::UTF8ToUnicodeText(w->first);
  };

  // Merges all sentences into one array with 0x0000 delimiter.
  std::vector<char32> array;
  // Use char32 keys to avoid billions of UnicodeCharToUTF8 string allocations.
  absl::flat_hash_map<char32, int64_t> all_chars;
  // Sentence boundary positions, computed during array construction to avoid
  // a separate O(n) scan over the multi-billion-element array.
  std::vector<node_int_type> boundary_positions;

  const bool is_tsv = trainer_spec_.input_format() == "tsv";
  const bool has_pretokenizer = (pretokenizer != nullptr) ||
      !trainer_spec_.pretokenization_delimiter().empty();

  if (!has_pretokenizer && trainer_spec_.num_threads() > 1) {
    // === Parallel array construction (common case, no pretokenizer). ===
    const int n_threads = trainer_spec_.num_threads();

    // Pass 1: Count UTF-8 characters per sentence via lead-byte counting.
    // A byte is a UTF-8 lead byte iff (byte & 0xC0) != 0x80. This gives
    // an exact char32 count for well-formed UTF-8 (guaranteed post-normalize).
    LOG(INFO) << "Counting characters for array construction...";
    std::vector<size_t> char_counts(sentences_.size());
    {
      auto pool = std::make_unique<ThreadPool>(n_threads);
      pool->StartWorkers();
      for (int t = 0; t < n_threads; ++t) {
        pool->Schedule([&, t]() {
          for (size_t i = t; i < sentences_.size();
               i += static_cast<size_t>(n_threads)) {
            size_t count = 0;
            for (unsigned char c : sentences_[i].first) {
              if ((c & 0xC0) != 0x80) ++count;
            }
            char_counts[i] = count;
          }
        });
      }
    }

    // Prefix sum to compute each sentence's write offset.
    // Each sentence contributes char_count + 1 (boundary) elements,
    // doubled in TSV mode.
    const size_t stride = is_tsv ? 2 : 1;
    std::vector<size_t> offsets(sentences_.size() + 1);
    offsets[0] = 0;
    for (size_t i = 0; i < sentences_.size(); ++i) {
      offsets[i + 1] = offsets[i] + (char_counts[i] + 1) * stride;
    }
    array.resize(offsets.back());

    // Pre-compute boundary positions from offsets (eliminates O(n) scan).
    boundary_positions.reserve(sentences_.size() * stride);
    for (size_t i = 0; i < sentences_.size(); ++i) {
      boundary_positions.push_back(
          static_cast<node_int_type>(offsets[i] + char_counts[i]));
      if (is_tsv) {
        boundary_positions.push_back(
            static_cast<node_int_type>(offsets[i] + 2 * char_counts[i] + 1));
      }
    }

    // Pass 2: Parallel UTF-8 decode + write to array + count char frequencies.
    LOG(INFO) << "Building array (" << offsets.back() << " elements, "
              << n_threads << " threads)...";
    std::vector<absl::flat_hash_map<char32, int64_t>> local_all_chars(n_threads);
    {
      auto pool = std::make_unique<ThreadPool>(n_threads);
      pool->StartWorkers();
      for (int t = 0; t < n_threads; ++t) {
        pool->Schedule([&, t]() {
          for (size_t i = t; i < sentences_.size();
               i += static_cast<size_t>(n_threads)) {
            const auto &w = sentences_[i];
            const auto ut = string_util::UTF8ToUnicodeText(w.first);
            CHECK_EQ(ut.size(), char_counts[i])
                << "Character count mismatch for sentence " << i;
            size_t pos = offsets[i];
            for (const auto &c : ut) {
              array[pos++] = c;
              if (c != kUNKChar && c != kSentenceBoundary) {
                local_all_chars[t][c] += w.second;
              }
            }
            array[pos++] = kSentenceBoundary;
            if (is_tsv) {
              for (const auto &c : ut) array[pos++] = c;
              array[pos++] = kSentenceBoundary;
            }
          }
        });
      }
    }

    // Merge thread-local char frequency maps.
    for (int t = 0; t < n_threads; ++t) {
      for (const auto &kv : local_all_chars[t]) {
        all_chars[kv.first] += kv.second;
      }
      local_all_chars[t].clear();
    }
  } else {
    // === Sequential fallback (pretokenizer active or single-threaded). ===
    {
      size_t estimated_size = 0;
      if (has_pretokenizer) {
        // Pretokenizer may add internal boundaries; use conservative estimate.
        for (const auto &w : sentences_) {
          estimated_size += w.first.size() * 2 + 1;
        }
      } else {
        // Count UTF-8 lead bytes for exact char count.
        for (const auto &w : sentences_) {
          for (unsigned char c : w.first) {
            if ((c & 0xC0) != 0x80) ++estimated_size;
          }
          estimated_size += 1;
        }
      }
      if (is_tsv) estimated_size *= 2;
      array.reserve(estimated_size);
    }

    boundary_positions.reserve(sentences_.size() * (is_tsv ? 2 : 1));
    for (auto &w : sentences_) {
      const auto ut = pretokenize_or_rewrite(&w);
      for (const auto &c : ut) {
        array.push_back(c);
        if (c != kUNKChar && c != kSentenceBoundary) {
          all_chars[c] += w.second;
        }
      }
      boundary_positions.push_back(static_cast<node_int_type>(array.size()));
      array.push_back(kSentenceBoundary);
      if (is_tsv) {
        for (const auto &c : ut) array.push_back(c);
        boundary_positions.push_back(static_cast<node_int_type>(array.size()));
        array.push_back(kSentenceBoundary);
      }
    }
    array.shrink_to_fit();  // Release excess capacity before ESA allocation.
  }

  // all_chars must be included in the seed sentencepieces.
  // Convert char32 keys to UTF-8 strings for seed pieces.
  TrainerModel::SentencePieces seed_sentencepieces;
  for (const auto &it : Sorted(all_chars)) {
    seed_sentencepieces.emplace_back(
        string_util::UnicodeCharToUTF8(it.first), it.second);
  }

  if (!trainer_spec_.seed_sentencepieces_file().empty()) {
    auto seed_sentencepieces_file = sentencepiece::filesystem::NewReadableFile(
        trainer_spec_.seed_sentencepieces_file());
    std::string line;
    int64_t freq = 1;
    int skipped_sentencepieces = 0;
    while (seed_sentencepieces_file->ReadLine(&line)) {
      const std::vector<std::string> fields = absl::StrSplit(line, '\t');
      CHECK_GE(fields.size(), 2);
      const auto &seed_sentencepiece = fields[0];
      CHECK(absl::SimpleAtoi(fields[1], &freq))
          << "Could not parse the frequency; line: " << line;
      const UnicodeText uw = string_util::UTF8ToUnicodeText(seed_sentencepiece);
      if (!IsValidSentencePiece(uw)) {
        ++skipped_sentencepieces;
        continue;
      }
      // Initialise score of a piece by character coverage.
      seed_sentencepieces.emplace_back(seed_sentencepiece, freq * uw.size());
      if (seed_sentencepieces.size() % 1000000 == 0) {
        LOG(INFO) << "loaded " << seed_sentencepieces.size()
                  << " seed sentencepieces";
      }
    }

    LOG(INFO) << "skipped " << skipped_sentencepieces << " seed sentencepieces";

    // Take highest scoring pieces as initial vocab.
    seed_sentencepieces = Sorted(seed_sentencepieces);
    seed_sentencepieces.resize(std::min<size_t>(
        trainer_spec_.seed_sentencepiece_size(), seed_sentencepieces.size()));

    LOG(INFO) << "Initialized " << seed_sentencepieces.size()
              << " seed sentencepieces from file.";
  } else {
    CHECK_LE(array.size(),
             static_cast<size_t>(std::numeric_limits<node_int_type>::max()))
        << "Input corpus too large, try with train_extremely_large_corpus=true";
    const node_int_type n = array.size();
    constexpr node_int_type kAlphabetSize = 0x110000;  // All UCS4 range.
    const size_t queue_size =
        static_cast<size_t>(trainer_spec_.seed_sentencepiece_size());
    const int num_threads = trainer_spec_.num_threads();

    // Minimum array size to justify the overhead of parallel splitting.
    constexpr node_int_type kParallelMinArraySize = 1000000;

    if (num_threads > 1 && n > kParallelMinArraySize) {
      // === Parallel path: split corpus at sentence boundaries, build
      // independent suffix arrays per chunk, merge top-K results. ===
      //
      // Valid seed pieces never span a sentence boundary (they are filtered
      // out), so splitting at boundaries preserves all practically relevant
      // candidates. Substrings with global freq >= 2 but per-chunk freq <= 1
      // everywhere are not surfaced; these have negligible scores and do not
      // affect the top-K.

      // boundary_positions was already computed during array construction.

      // Compute chunk ranges, splitting boundaries into ~equal groups.
      struct ChunkRange {
        node_int_type start;
        node_int_type end;
      };
      // Cap chunk size at 256M elements so each chunk fits in int32_t
      // (for half-sized SA/L/R/D arrays) and per-batch memory stays bounded.
      // With int32_t: 4 arrays * 256M * 4 bytes = 4 GB per chunk.
      constexpr int64_t kMaxChunkElements = 256LL * 1024 * 1024;
      const int desired_K = std::max(num_threads,
          static_cast<int>((static_cast<int64_t>(n) + kMaxChunkElements - 1) /
                           kMaxChunkElements));
      const int K = std::min(desired_K,
          std::max(1, static_cast<int>(boundary_positions.size())));
      std::vector<ChunkRange> chunks(K);
      const int64_t num_boundaries = boundary_positions.size();
      for (int k = 0; k < K; ++k) {
        if (k == 0) {
          chunks[k].start = 0;
        } else {
          const int64_t boundary_idx =
              static_cast<int64_t>(k) * num_boundaries / K - 1;
          chunks[k].start = boundary_positions[boundary_idx] + 1;
        }
        if (k == K - 1) {
          chunks[k].end = n;
        } else {
          const int64_t boundary_idx =
              static_cast<int64_t>(k + 1) * num_boundaries / K - 1;
          chunks[k].end = boundary_positions[boundary_idx] + 1;
        }
      }
      // Remove empty chunks.
      chunks.erase(
          std::remove_if(chunks.begin(), chunks.end(),
                         [](const ChunkRange &c) { return c.start >= c.end; }),
          chunks.end());
      const int actual_K = chunks.size();

      LOG(INFO) << "Making suffix array (parallel, " << actual_K
                << " chunks)...";

      // Per-chunk priority queues collecting top-K candidates as strings.
      std::vector<BoundedPriorityQueue<std::string>> local_queues;
      local_queues.reserve(actual_K);
      for (int k = 0; k < actual_K; ++k) {
        local_queues.emplace_back(queue_size);
      }

      // Process chunks in batches to cap peak memory. Each chunk's SA/L/R/D
      // arrays use int32_t (4 bytes each, 16 bytes/element total).
      // Budget: limit concurrent ESA memory to ~48 GB.
      constexpr int64_t kMaxBatchMemory = 48LL * 1024 * 1024 * 1024;
      const int64_t avg_chunk_size = (static_cast<int64_t>(n) + actual_K - 1) / actual_K;
      const int64_t per_chunk_bytes = 4LL * avg_chunk_size * sizeof(int32_t);
      const int mem_limited = std::max(1,
          static_cast<int>(kMaxBatchMemory / std::max(per_chunk_bytes, int64_t{1})));
      const int batch_size = std::min(actual_K, std::min(num_threads, mem_limited));
      for (int batch_start = 0; batch_start < actual_K;
           batch_start += batch_size) {
        const int batch_end = std::min(batch_start + batch_size, actual_K);
        const int batch_count = batch_end - batch_start;
        {
          ThreadPool pool(batch_count);
          pool.StartWorkers();
          for (int b = 0; b < batch_count; ++b) {
            const int k = batch_start + b;
            pool.Schedule([&, k]() {
              const node_int_type chunk_start = chunks[k].start;
              const node_int_type chunk_sz = chunks[k].end - chunk_start;

              // Use int32_t for chunk-local ESA arrays. Each chunk is
              // bounded by kMaxChunkElements (256M) which fits in int32_t,
              // halving memory vs int64_t (4 vs 8 bytes per element).
              CHECK_LE(chunk_sz, static_cast<node_int_type>(
                  std::numeric_limits<int32_t>::max()))
                  << "Chunk too large for int32_t indexing";
              const int32_t chunk_size_i32 = static_cast<int32_t>(chunk_sz);

              std::vector<int32_t> chunk_SA(chunk_size_i32);
              std::vector<int32_t> chunk_L(chunk_size_i32);
              std::vector<int32_t> chunk_R(chunk_size_i32);
              std::vector<int32_t> chunk_D(chunk_size_i32);
              int32_t chunk_node_num = 0;

              CHECK_EQ(0, esaxx(array.begin() + chunk_start, chunk_SA.begin(),
                                chunk_L.begin(), chunk_R.begin(),
                                chunk_D.begin(), chunk_size_i32,
                                static_cast<int32_t>(kAlphabetSize),
                                chunk_node_num));

              for (int32_t i = 0; i < chunk_node_num; ++i) {
                const int32_t offset = chunk_SA[chunk_L[i]];
                const int32_t len = chunk_D[i];
                if (len <= 1 || offset >= chunk_size_i32 ||
                    offset + len >= chunk_size_i32) {
                  continue;
                }
                // Widen back to node_int_type for global array indexing.
                const char32 *begin = &array[chunk_start +
                    static_cast<node_int_type>(offset)];
                const char32 *end = &array[chunk_start +
                    static_cast<node_int_type>(offset + len)];
                if (std::find(begin, end, kSentenceBoundary) != end) {
                  continue;
                }
                const UnicodeText uw(begin, end);
                if (!IsValidSentencePiece(uw)) {
                  continue;
                }

                const int32_t freq = chunk_R[i] - chunk_L[i];
                const int64_t score =
                    static_cast<int64_t>(freq) * static_cast<int64_t>(len);
                const std::string w = string_util::UnicodeTextToUTF8(uw);
                local_queues[k].push(w, score);
              }
            });
          }
        }  // ThreadPool destructor joins batch threads.
      }

      LOG(INFO) << "Merging results from " << actual_K << " chunks...";

      // Merge per-chunk queues into a single final queue.
      BoundedPriorityQueue<std::string> queue(queue_size);
      for (int k = 0; k < actual_K; ++k) {
        for (const auto &p : local_queues[k].get()) {
          queue.push(p.first, p.second);
        }
      }

      // Dedup: the same substring may appear in multiple chunks' queues.
      // The queue is sorted by score descending, so the first occurrence wins.
      absl::flat_hash_set<std::string> seen;
      for (const auto &p : queue.get()) {
        const std::string &w = p.first;
        if (!seen.insert(w).second) continue;
        seed_sentencepieces.emplace_back(w, static_cast<float>(p.second));
      }
    } else {
      // === Sequential fallback with parallel post-processing. ===
      std::vector<node_int_type> SA(n);  // suffix array
      std::vector<node_int_type> L(n);   // left boundaries of internal node
      std::vector<node_int_type> R(n);   // right boundaries of internal node
      std::vector<node_int_type> D(n);   // depths of internal node

      node_int_type node_num = 0;
      LOG(INFO) << "Making suffix array...";
      CHECK_EQ(0, esaxx(array.begin(), SA.begin(), L.begin(), R.begin(),
                        D.begin(), n, kAlphabetSize, node_num));

      LOG(INFO) << "Extracting frequent sub strings... node_num=" << node_num;

      // Parallelize node extraction using thread-local queues.
      std::vector<BoundedPriorityQueue<node_int_type>> local_queues;
      local_queues.reserve(num_threads);
      for (int t = 0; t < num_threads; ++t) {
        local_queues.emplace_back(queue_size);
      }

      {
        ThreadPool pool(num_threads);
        pool.StartWorkers();
        for (int t = 0; t < num_threads; ++t) {
          pool.Schedule([&, t]() {
            for (node_int_type i = t; i < node_num;
                 i += static_cast<node_int_type>(num_threads)) {
              const node_int_type offset = SA[L[i]];
              const node_int_type len = D[i];
              if (len <= 1 || offset >= array.size() ||
                  offset + len >= array.size()) {
                continue;
              }
              const char32 *begin = &array[offset];
              const char32 *end = &array[offset + len];
              if (std::find(begin, end, kSentenceBoundary) != end) {
                continue;
              }
              const UnicodeText uw(begin, end);
              if (!IsValidSentencePiece(uw)) {
                continue;
              }

              const node_int_type freq = R[i] - L[i];
              const node_int_type score = freq * len;
              local_queues[t].push(i, score);
            }
          });
        }
      }  // ThreadPool destructor joins all threads.

      // Merge thread-local queues.
      BoundedPriorityQueue<node_int_type> queue(queue_size);
      for (int t = 0; t < num_threads; ++t) {
        for (const auto &p : local_queues[t].get()) {
          queue.push(p.first, p.second);
        }
      }

      for (const auto &p : queue.get()) {
        const node_int_type offset = SA[L[p.first]];
        const node_int_type len = D[p.first];
        CHECK_GT(len, 0);
        const char32 *begin = &array[offset];
        const char32 *end = &array[offset + len];
        const UnicodeText uw(begin, end);
        const std::string w = string_util::UnicodeTextToUTF8(uw);
        CHECK(IsValidSentencePiece(uw));  // just in case.
        seed_sentencepieces.emplace_back(w, p.second);
      }
    }
  }

  ToLogProb(seed_sentencepieces.begin(), seed_sentencepieces.end());

  LOG(INFO) << "Initialized " << seed_sentencepieces.size()
            << " seed sentencepieces";

  return seed_sentencepieces;
}

std::vector<float> Trainer::RunEStep(const TrainerModel &model, float *obj,
                                     int64_t *num_tokens) const {
  std::vector<std::vector<float>> expected(trainer_spec_.num_threads());
  std::vector<float> objs(trainer_spec_.num_threads(), 0.0);
  std::vector<int64_t> ntokens(trainer_spec_.num_threads(), 0.0);

  auto pool = std::make_unique<ThreadPool>(trainer_spec_.num_threads());
  pool->StartWorkers();

  // Executes E step in parallel
  for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
    pool->Schedule([&, n]() {
      Lattice lattice;
      expected[n].resize(model.GetPieceSize(), 0.0);
      for (size_t i = n; i < sentences_.size();
           i += trainer_spec_.num_threads()) {
        const std::string &w = sentences_[i].first;
        const int64_t freq = sentences_[i].second;
        lattice.SetSentence(w);
        model.PopulateNodes(&lattice);
        const float Z = lattice.PopulateMarginal(freq, &expected[n]);
        ntokens[n] += lattice.Viterbi().first.size() * freq;
        CHECK(!std::isnan(Z))
            << "likelihood is NAN. Input sentence may be too long";
        objs[n] -= Z / all_sentence_freq_;
      }
    });
  }
  pool.reset(nullptr);

  // Merges expectations
  for (int n = 1; n < trainer_spec_.num_threads(); ++n) {
    objs[0] += objs[n];
    ntokens[0] += ntokens[n];
    for (size_t k = 0; k < expected[0].size(); ++k) {
      expected[0][k] += expected[n][k];
    }
  }

  *obj = objs[0];
  *num_tokens = ntokens[0];
  CHECK(!std::isnan(*obj));

  return expected[0];
}

TrainerModel::SentencePieces Trainer::RunMStep(
    const TrainerModel &model, const std::vector<float> &expected) const {
  const auto &sentencepieces = model.GetSentencePieces();
  CHECK_EQ(sentencepieces.size(), expected.size());
  TrainerModel::SentencePieces new_sentencepieces;

  float sum = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const float freq = expected[i];

    // Filter infrequent sentencepieces here.
    constexpr float kExpectedFrequencyThreshold = 0.5;
    if (freq < kExpectedFrequencyThreshold) {
      continue;
    }

    new_sentencepieces.emplace_back(sentencepieces[i].first, freq);
    sum += freq;
  }

  // Here we do not use the original EM, but use the
  // Bayesianified/DPified EM algorithm.
  // https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
  // This modification will act as a sparse prior.
  const float logsum = Digamma(sum);
  for (auto &w : new_sentencepieces) {
    w.second = Digamma(w.second) - logsum;
  }

  return new_sentencepieces;
}

TrainerModel::SentencePieces Trainer::PruneSentencePieces(
    const TrainerModel &model) const {
  const auto &sentencepieces = model.GetSentencePieces();

  // Use char instead of bool to avoid std::vector<bool> bit-packing,
  // which is unsafe for concurrent writes to adjacent indices.
  std::vector<char> always_keep(sentencepieces.size(), 1);
  std::vector<std::vector<int>> alternatives(sentencepieces.size());

  // First, segments the current sentencepieces to know
  // how each sentencepiece is resegmented if this sentencepiece is removed
  // from the vocabulary.
  // To do so, we take the second best segmentation of sentencepiece[i].
  // alternatives[i] stores the sequence of second best sentencepieces.
  {
    const int num_threads = trainer_spec_.num_threads();
    auto pool = std::make_unique<ThreadPool>(num_threads);
    pool->StartWorkers();
    for (int n = 0; n < num_threads; ++n) {
      pool->Schedule([&, n]() {
        Lattice lattice;
        for (size_t i = n; i < sentencepieces.size();
             i += static_cast<size_t>(num_threads)) {
          const auto &w = sentencepieces[i];
          lattice.SetSentence(w.first);
          model.PopulateNodes(&lattice);
          const auto nbests = lattice.NBest(2, false, 0.0);
          if (nbests.size() == 1) {
            always_keep[i] = 1;
            continue;
          } else if (nbests[0].first.size() >= 2) {
            always_keep[i] = 0;
          } else if (nbests[0].first.size() == 1) {
            always_keep[i] = 1;
            for (const auto *node : nbests[1].first) {
              alternatives[i].push_back(node->id);
            }
          }
        }
      });
    }
  }

  // Second, segment all sentences to compute token frequencies
  // with a unigram language model using the Viterbi path.
  std::vector<float> freq(sentencepieces.size(), 0.0);
  {
    std::vector<std::vector<float>> freqs(trainer_spec_.num_threads());

    auto pool = std::make_unique<ThreadPool>(trainer_spec_.num_threads());
    pool->StartWorkers();
    for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
      freqs[n].resize(sentencepieces.size(), 0.0);

      pool->Schedule([&, n]() {
        Lattice lattice;
        for (size_t i = n; i < sentences_.size();
             i += trainer_spec_.num_threads()) {
          const auto &w = sentences_[i];
          lattice.SetSentence(w.first);
          model.PopulateNodes(&lattice);
          for (const auto *node : lattice.Viterbi().first) {
            if (node->id >= 0) {
              freqs[n][node->id] += w.second;
            }
          }
        }
      });
    }
    pool.reset(nullptr);

    for (int n = 0; n < trainer_spec_.num_threads(); ++n) {
      for (size_t i = 0; i < sentencepieces.size(); ++i) {
        freq[i] += freqs[n][i];
      }
    }
  }

  const float sum = std::accumulate(freq.begin(), freq.end(), 0.0);
  const float logsum = std::log(static_cast<double>(sum));
  std::vector<std::pair<int, float>> candidates;
  TrainerModel::SentencePieces new_sentencepieces;

  // Finally, computes how likely the LM likelihood is reduced if
  // the sentencepiece[i] is removed from the vocabulary.
  // Since the exact computation of loss is difficult, we compute the
  // loss approximately by assuming that all sentencepiece[i] in the sentences
  // are replaced with alternatives[i] when sentencepiece[i] is removed.
  for (size_t i = 0; i < sentencepieces.size(); ++i) {
    if (freq[i] == 0 || !always_keep[i]) {
      // not found in Viterbi path. Can remove this entry safely.
      continue;
    } else if (alternatives[i].empty()) {
      // no alternatives. Keeps this entry.
      new_sentencepieces.push_back(sentencepieces[i]);
    } else {
      // The logprob with the sentencepiece[i].
      const float logprob_sp = std::log(static_cast<double>(freq[i])) - logsum;

      // After removing the sentencepiece[i], its frequency freq[i] is
      // re-assigned to alternatives.
      // new_sum = current_sum - freq[i] + freq[i] * alternatives[i].size()
      //         = current_sum + freq[i] * (alternatives[i] - 1)
      const float logsum_alt = std::log(
          static_cast<double>(sum + freq[i] * (alternatives[i].size() - 1)));

      // The frequencies of altenatives are increased by freq[i].
      float logprob_alt = 0.0;
      for (const int n : alternatives[i]) {
        logprob_alt +=
            (std::log(static_cast<double>(freq[n] + freq[i])) - logsum_alt);
      }

      // loss: the diff of likelihood after removing the sentencepieces[i].
      float F = freq[i] / sum;  // normalized token frequency
      const float loss = F * (logprob_sp - logprob_alt);
      candidates.emplace_back(i, loss);
    }
  }

  const int pruned_size =
      std::max<int>(desired_vocab_size_,
                    trainer_spec_.shrinking_factor() * sentencepieces.size());

  // Keeps trainer_spec_.shrinking_factor * sentencepieces.size() pieces.
  // shrinking_factor is 0.75 by default.
  for (const auto &w : Sorted(candidates)) {
    if (new_sentencepieces.size() == static_cast<size_t>(pruned_size)) {
      break;
    }
    new_sentencepieces.emplace_back(sentencepieces[w.first]);
  }

  return new_sentencepieces;
}

TrainerModel::SentencePieces Trainer::FinalizeSentencePieces(
    const TrainerModel &model) const {
  const auto &sentencepieces = model.GetSentencePieces();
  absl::flat_hash_map<std::string, float> final_sentencepieces;
  absl::flat_hash_map<std::string, float> sp(sentencepieces.begin(),
                                             sentencepieces.end());

  // required_chars_ must be included in the final sentencepieces.
  float min_score_penalty = 0.0;
  constexpr float kMinScorePenaltyDelta = 0.0001;
  for (const auto &w : Sorted(required_chars_)) {
    const std::string s = string_util::UnicodeCharToUTF8(w.first);
    if (port::ContainsKey(sp, s)) {
      final_sentencepieces[s] = sp[s];
    } else {
      // Add penalty to avoid required pieces from having the same score.
      // Since the required_chars_ is sorted, frequent pieces have
      // less penalties.
      final_sentencepieces[s] = model.min_score() + min_score_penalty;
      min_score_penalty += kMinScorePenaltyDelta;
    }
  }

  const int vocab_size_size = trainer_spec_.vocab_size() - meta_pieces_.size();
  CHECK_GT(vocab_size_size, 0);

  // Then keeps sentencepieces with higher scores.
  for (const auto &w : Sorted(sentencepieces)) {
    if (port::ContainsKey(final_sentencepieces, w.first)) {
      continue;
    }
    if (static_cast<size_t>(vocab_size_size) == final_sentencepieces.size()) {
      break;
    }
    final_sentencepieces[w.first] = w.second;
  }

  return Sorted(final_sentencepieces);
}

util::Status Trainer::Train() {
  RETURN_IF_ERROR(status());

  CHECK_EQ_OR_RETURN(TrainerSpec::UNIGRAM, trainer_spec_.model_type());
  CHECK_OR_RETURN(normalizer_spec_.escape_whitespaces());

  TrainerModel model(trainer_spec_, normalizer_spec_);

  RETURN_IF_ERROR(model.status());
  RETURN_IF_ERROR(LoadSentences());

  auto seed_sentencepieces = MakeSeedSentencePieces();
  model.SetSentencePieces(std::move(seed_sentencepieces));

  if (trainer_spec_.split_by_whitespace()) {
    SplitSentencesByWhitespace();
  }

  LOG(INFO) << "Using " << sentences_.size() << " sentences for EM training";

  all_sentence_freq_ = 0;
  for (const auto &w : sentences_) {
    all_sentence_freq_ += w.second;
  }

  desired_vocab_size_ = static_cast<size_t>(trainer_spec_.vocab_size() * 1.1);

  while (true) {
    // Sub-EM iteration.
    for (int iter = 0; iter < trainer_spec_.num_sub_iterations(); ++iter) {
      // Executes E step
      float objective = 0.0;
      int64_t num_tokens = 0;
      const auto expected = RunEStep(model, &objective, &num_tokens);

      // Executes M step.
      auto new_sentencepieces = RunMStep(model, expected);
      model.SetSentencePieces(std::move(new_sentencepieces));

      LOG(INFO) << "EM sub_iter=" << iter << " size=" << model.GetPieceSize()
                << " obj=" << objective << " num_tokens=" << num_tokens
                << " num_tokens/piece="
                << 1.0 * num_tokens / model.GetPieceSize();
    }  // end of Sub EM iteration

    // Stops the iteration when the size of sentences reaches to the
    // desired symbol size.
    if (model.GetPieceSize() <= desired_vocab_size_) {
      break;
    }

    // Prunes pieces.
    auto new_sentencepieces = PruneSentencePieces(model);
    model.SetSentencePieces(std::move(new_sentencepieces));
  }  // end of EM iteration

  // Finally, adjusts the size of sentencepices to be |vocab_size|.
  final_pieces_ = FinalizeSentencePieces(model);

  return Save();
}
}  // namespace unigram
}  // namespace sentencepiece
