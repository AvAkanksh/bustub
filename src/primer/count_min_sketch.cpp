//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// count_min_sketch.cpp
//
// Identification: src/primer/count_min_sketch.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "primer/count_min_sketch.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace bustub {

/**
 * Constructor for the count-min sketch.
 *
 * @param width The width of the sketch matrix.
 * @param depth The depth of the sketch matrix.
 * @throws std::invalid_argument if width or depth are zero.
 */
template <typename KeyType>
CountMinSketch<KeyType>::CountMinSketch(uint32_t width, uint32_t depth) : width_(width), depth_(depth) {
  if (width_ == 0 || depth_ == 0) {
    throw std::invalid_argument("CountMinSketch width and depth must be non-zero.");
  }
  counts_ = std::vector<std::atomic<uint32_t>>(static_cast<size_t>(width_) * depth_);

  /** @fall2025 PLEASE DO NOT MODIFY THE FOLLOWING */
  // Initialize seeded hash functions
  hash_functions_.reserve(depth_);
  for (size_t i = 0; i < depth_; i++) {
    hash_functions_.push_back(this->HashFunction(i));
  }
}

template <typename KeyType>
CountMinSketch<KeyType>::CountMinSketch(CountMinSketch &&other) noexcept
    : width_(other.width_), depth_(other.depth_), counts_(std::move(other.counts_)) {
  // The hash functions capture `this`, so they must be rebuilt for the new object instead of moved.
  hash_functions_.reserve(depth_);
  for (size_t i = 0; i < depth_; i++) {
    hash_functions_.push_back(this->HashFunction(i));
  }
  // Leave the moved-from sketch empty but safe to destroy or query.
  other.width_ = 0;
  other.depth_ = 0;
  other.hash_functions_.clear();
}

template <typename KeyType>
auto CountMinSketch<KeyType>::operator=(CountMinSketch &&other) noexcept -> CountMinSketch & {
  if (this == &other) {
    return *this;
  }
  width_ = other.width_;
  depth_ = other.depth_;
  counts_ = std::move(other.counts_);
  // The hash functions capture `this`, so they must be rebuilt for this object instead of moved.
  hash_functions_.clear();
  hash_functions_.reserve(depth_);
  for (size_t i = 0; i < depth_; i++) {
    hash_functions_.push_back(this->HashFunction(i));
  }
  other.width_ = 0;
  other.depth_ = 0;
  other.hash_functions_.clear();
  return *this;
}

template <typename KeyType>
void CountMinSketch<KeyType>::Insert(const KeyType &item) {
  // Each counter is incremented atomically, so no latch is needed. Relaxed ordering is enough because
  // counters are independent and readers only need the eventual totals.
  for (uint32_t row = 0; row < depth_; row++) {
    counts_[Index(row, hash_functions_[row](item))].fetch_add(1, std::memory_order_relaxed);
  }
}

template <typename KeyType>
void CountMinSketch<KeyType>::Merge(const CountMinSketch<KeyType> &other) {
  if (width_ != other.width_ || depth_ != other.depth_) {
    throw std::invalid_argument("Incompatible CountMinSketch dimensions for merge.");
  }
  // Both sketches use the same seeded hash functions, so summing cell-wise is equivalent to
  // having inserted both streams into one sketch.
  for (size_t i = 0; i < counts_.size(); i++) {
    counts_[i].fetch_add(other.counts_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
  }
}

template <typename KeyType>
auto CountMinSketch<KeyType>::Count(const KeyType &item) const -> uint32_t {
  if (depth_ == 0) {
    return 0;
  }
  // Collisions can only inflate a counter, so the smallest one is the tightest estimate.
  uint32_t min_count = std::numeric_limits<uint32_t>::max();
  for (uint32_t row = 0; row < depth_; row++) {
    min_count = std::min(min_count, counts_[Index(row, hash_functions_[row](item))].load(std::memory_order_relaxed));
  }
  return min_count;
}

template <typename KeyType>
void CountMinSketch<KeyType>::Clear() {
  for (auto &count : counts_) {
    count.store(0, std::memory_order_relaxed);
  }
}

template <typename KeyType>
auto CountMinSketch<KeyType>::TopK(uint16_t k, const std::vector<KeyType> &candidates)
    -> std::vector<std::pair<KeyType, uint32_t>> {
  std::vector<std::pair<KeyType, uint32_t>> result;
  result.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    result.emplace_back(candidate, Count(candidate));
  }
  // Stable sort keeps candidates with equal estimates in their original order.
  std::stable_sort(result.begin(), result.end(),
                   [](const auto &a, const auto &b) -> bool { return a.second > b.second; });
  if (result.size() > k) {
    result.erase(result.begin() + k, result.end());
  }
  return result;
}

// Explicit instantiations for all types used in tests
template class CountMinSketch<std::string>;
template class CountMinSketch<int64_t>;  // For int64_t tests
template class CountMinSketch<int>;      // This covers both int and int32_t
}  // namespace bustub
