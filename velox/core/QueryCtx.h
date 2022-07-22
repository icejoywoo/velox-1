/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <folly/Executor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include "velox/common/memory/MappedMemory.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/Context.h"
#include "velox/core/QueryConfig.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/VectorPool.h"

namespace facebook::velox::core {

class QueryCtx : public Context {
 public:
  // Returns QueryCtx with new executor created if not supplied. For testing
  // purpose.
  static std::shared_ptr<QueryCtx> createForTest(
      std::shared_ptr<Config> config = std::make_shared<MemConfig>(),
      std::shared_ptr<folly::Executor> executor =
          std::make_shared<folly::CPUThreadPoolExecutor>(
              std::thread::hardware_concurrency())) {
    return std::make_shared<QueryCtx>(std::move(executor), std::move(config));
  }

  // QueryCtx is used in different places. When used with `Task`, it's required
  // that callers supply executors. In contrast, when used in expression
  // evaluation through `ExecCtx`, executor is not needed. Hence, we don't
  // require executor to always be passed in here, but instead, ensure that
  // executor exists when actually being used.
  //
  // This constructor keeps the ownership of `executor`.
  QueryCtx(
      std::shared_ptr<folly::Executor> executor = nullptr,
      std::shared_ptr<Config> config = std::make_shared<MemConfig>(),
      std::unordered_map<std::string, std::shared_ptr<Config>>
          connectorConfigs = {},
      memory::MappedMemory* FOLLY_NONNULL mappedMemory =
          memory::MappedMemory::getInstance(),
      std::unique_ptr<memory::MemoryPool> pool = nullptr,
      std::shared_ptr<folly::Executor> spillExecutor = nullptr)
      : Context{ContextScope::QUERY},
        pool_(std::move(pool)),
        mappedMemory_(mappedMemory),
        connectorConfigs_(connectorConfigs),
        executor_{std::move(executor)},
        config_{this},
        spillExecutor_(std::move(spillExecutor)) {
    setConfigOverrides(config);
    if (!pool_) {
      initPool();
    }
  }

  // Constructor to block the destruction of executor while this
  // object is alive.
  //
  // This constructor does not keep the ownership of executor.
  explicit QueryCtx(
      folly::Executor::KeepAlive<> executorKeepalive,
      std::shared_ptr<Config> config = std::make_shared<MemConfig>(),
      std::unordered_map<std::string, std::shared_ptr<Config>>
          connectorConfigs = {},
      memory::MappedMemory* FOLLY_NONNULL mappedMemory =
          memory::MappedMemory::getInstance(),
      std::unique_ptr<memory::MemoryPool> pool = nullptr)
      : Context{ContextScope::QUERY},
        pool_(std::move(pool)),
        mappedMemory_(mappedMemory),
        connectorConfigs_(connectorConfigs),
        executorKeepalive_(std::move(executorKeepalive)),
        config_{this} {
    setConfigOverrides(config);
    if (!pool_) {
      initPool();
    }
  }

  memory::MemoryPool* FOLLY_NONNULL pool() const {
    return pool_.get();
  }

  memory::MappedMemory* FOLLY_NONNULL mappedMemory() const {
    return mappedMemory_;
  }

  folly::Executor* FOLLY_NONNULL executor() const {
    if (executor_) {
      return executor_.get();
    }
    auto executor = executorKeepalive_.get();
    VELOX_CHECK(executor, "Executor was not supplied.");
    return executor;
  }

  const QueryConfig& config() const {
    return config_;
  }

  Config* FOLLY_NONNULL
  getConnectorConfig(const std::string& connectorId) const {
    auto it = connectorConfigs_.find(connectorId);
    if (it == connectorConfigs_.end()) {
      return getEmptyConfig();
    }
    return it->second.get();
  }

  // Overrides the previous configuration. Note that this function is NOT
  // thread-safe and should probably only be used in tests.
  void setConfigOverridesUnsafe(
      std::unordered_map<std::string, std::string>&& configOverrides) {
    setConfigOverrides(
        std::make_shared<const MemConfig>(std::move(configOverrides)));
  }

  folly::Executor* FOLLY_NULLABLE spillExecutor() const {
    return spillExecutor_.get();
  }

 private:
  static Config* FOLLY_NONNULL getEmptyConfig() {
    static const std::unique_ptr<Config> kEmptyConfig =
        std::make_unique<MemConfig>();
    return kEmptyConfig.get();
  }

  void initPool() {
    pool_ = memory::getProcessDefaultMemoryManager().getRoot().addScopedChild(
        kQueryRootMemoryPool);
    static const auto kUnlimited = std::numeric_limits<int64_t>::max();
    pool_->setMemoryUsageTracker(
        memory::MemoryUsageTracker::create(kUnlimited, kUnlimited, kUnlimited));
  }

  static constexpr const char* FOLLY_NONNULL kQueryRootMemoryPool =
      "query_root";

  std::unique_ptr<memory::MemoryPool> pool_;
  memory::MappedMemory* FOLLY_NONNULL mappedMemory_;
  std::unordered_map<std::string, std::shared_ptr<Config>> connectorConfigs_;
  std::shared_ptr<folly::Executor> executor_;
  folly::Executor::KeepAlive<> executorKeepalive_;
  QueryConfig config_;
  std::shared_ptr<folly::Executor> spillExecutor_;
};

// Represents the state of one thread of query execution.
class ExecCtx : public Context {
 public:
  ExecCtx(
      memory::MemoryPool* FOLLY_NONNULL pool,
      QueryCtx* FOLLY_NULLABLE queryCtx)
      : Context{ContextScope::QUERY},
        pool_(pool),
        queryCtx_(queryCtx),
        vectorPool_{pool} {}

  velox::memory::MemoryPool* FOLLY_NONNULL pool() const {
    return pool_;
  }

  QueryCtx* FOLLY_NONNULL queryCtx() const {
    return queryCtx_;
  }

  /// Returns an uninitialized  SelectivityVector from a pool. Allocates new one
  /// if none is available. Make sure to call 'releaseSelectivityVector' when
  /// done using the vector to allow for reuse.
  ///
  /// Prefer using LocalSelectivityVector which takes care of returning the
  /// vector to the pool on destruction.
  std::unique_ptr<SelectivityVector> getSelectivityVector(int32_t size) {
    if (selectivityVectorPool_.empty()) {
      return std::make_unique<SelectivityVector>(size);
    }
    auto vector = std::move(selectivityVectorPool_.back());
    selectivityVectorPool_.pop_back();
    vector->resize(size);
    return vector;
  }

  // Returns an arbitrary SelectivityVector with undefined
  // content. The caller is responsible for setting the size and
  // assigning the contents.
  std::unique_ptr<SelectivityVector> getSelectivityVector() {
    if (selectivityVectorPool_.empty()) {
      return std::make_unique<SelectivityVector>();
    }
    auto vector = std::move(selectivityVectorPool_.back());
    selectivityVectorPool_.pop_back();
    return vector;
  }

  void releaseSelectivityVector(std::unique_ptr<SelectivityVector>&& vector) {
    selectivityVectorPool_.push_back(std::move(vector));
  }

  std::unique_ptr<DecodedVector> getDecodedVector() {
    if (decodedVectorPool_.empty()) {
      return std::make_unique<DecodedVector>();
    }
    auto vector = std::move(decodedVectorPool_.back());
    decodedVectorPool_.pop_back();
    return vector;
  }

  void releaseDecodedVector(std::unique_ptr<DecodedVector>&& vector) {
    decodedVectorPool_.push_back(std::move(vector));
  }

  VectorPool& vectorPool() {
    return vectorPool_;
  }

  /// Gets a possibly recycled vector of 'type and 'size'. Allocates from
  /// 'pool_' if no pre-allocated vector.
  VectorPtr getVector(const TypePtr& type, vector_size_t size) {
    return vectorPool_.get(type, size);
  }

  /// Moves 'vector' to the pool if it is reusable, else leaves it in
  /// place. Returns true if the vector was moved into the pool.
  bool releaseVector(VectorPtr& vector) {
    return vectorPool_.release(vector);
  }

  /// Moves elements of 'vectors' to the pool if reusable, else leaves them
  /// in place. Returns number of vectors that were moved into the pool.
  size_t releaseVectors(std::vector<VectorPtr>& vectors) {
    return vectorPool_.release(vectors);
  }

 private:
  // Pool for all Buffers for this thread
  memory::MemoryPool* FOLLY_NONNULL pool_;
  QueryCtx* FOLLY_NULLABLE queryCtx_;
  // A pool of preallocated DecodedVectors for use by expressions and operators.
  std::vector<std::unique_ptr<DecodedVector>> decodedVectorPool_;
  // A pool of preallocated SelectivityVectors for use by expressions
  // and operators.
  std::vector<std::unique_ptr<SelectivityVector>> selectivityVectorPool_;
  VectorPool vectorPool_;
};

} // namespace facebook::velox::core
