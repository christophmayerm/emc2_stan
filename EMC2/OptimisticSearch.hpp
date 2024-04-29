#ifndef EB_OPTIMISTIC_SEARCH_HPP_
#define EB_OPTIMISTIC_SEARCH_HPP_

#include <cassert>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>

#include "AdamGradientAscent.hpp"
#include "Hypercube.hpp"
#include "RandomValue.hpp"

namespace EB {

template <typename T, int SIZE>
struct Maximum {
  Eigen::Vector<T, SIZE> theta;
  T func_value;
  Eigen::Vector<T, SIZE> gradient;

  constexpr Maximum(Eigen::Vector<T, SIZE> theta,
                    T func_value,
                    Eigen::Vector<T, SIZE> gradient) noexcept
      : theta(std::move(theta)),
        func_value(func_value),
        gradient(std::move(gradient)) {}
};

// - Get size of a hypercube for a given maximum ---------—---------—---------—---------—-----------
template <typename T, int SIZE>
  requires(SIZE > 0)
constexpr void get_hypercube_size(const Maximum<T, SIZE>& maximum,
                                  const Hypercube<T, SIZE>& boundaries,
                                  std::array<T, static_cast<size_t>(SIZE)>& sizes_out,
                                  T max_factor = 0.05,
                                  T min_factor = 0.01) noexcept {
  assert(hypercube_contains_point(boundaries, maximum.theta) &&
         "Boundaries must contain the maximum.");

  for (size_t i = 0; i < static_cast<size_t>(SIZE); ++i) {
    const auto length = boundaries[i].max - boundaries[i].min;
    const auto factor = std::max(min_factor, std::min(maximum.gradient.norm(), max_factor));
    sizes_out[i]      = factor * length;
  }
}

// - Get size of a hypercube around a maximum with a given size ---------—---------—---------—------
template <typename T, int SIZE>
  requires(SIZE > 0)
constexpr void get_hypercube(const Maximum<T, SIZE>& maximum,
                             std::array<T, static_cast<size_t>(SIZE)>& hc_sizes,
                             const Hypercube<T, SIZE>& boundaries,
                             Hypercube<T, SIZE>& hc_out) noexcept {
  assert(hypercube_contains_point(boundaries, maximum.theta) &&
         "Boundaries must contain the maximum.");

  for (size_t i = 0; i < static_cast<size_t>(SIZE); ++i) {
    hc_out[i].min = std::max(maximum.theta[i] - hc_sizes[i] / 2.0, boundaries[i].min);
    hc_out[i].max = std::min(maximum.theta[i] + hc_sizes[i] / 2.0, boundaries[i].max);
  }
}

// - Get list of all hypercubes for a given list of maxima ---------—---------—---------—---------—-
template <typename T, int SIZE>
  requires(SIZE > 0)
[[nodiscard]] constexpr auto get_hypercubes(const std::vector<Maximum<T, SIZE>>& maxima,
                                            const Hypercube<T, SIZE>& boundaries,
                                            T max_factor = 0.05,
                                            T min_factor = 0.01) noexcept
    -> std::vector<Hypercube<T, SIZE>> {
  std::vector<Hypercube<T, SIZE>> hcs{};
  hcs.reserve(maxima.size());

  for (const auto& maximum : maxima) {
    std::array<T, static_cast<size_t>(SIZE)> hc_size;
    get_hypercube_size(maximum, boundaries, hc_size, max_factor, min_factor);

    Hypercube<T, SIZE> hc;
    get_hypercube(maximum, hc_size, boundaries, hc);

    if (std::none_of(std::cbegin(hcs), std::cend(hcs), [&hc](const auto& hc_outer) {
          return hypercube_contains_hypercube(hc_outer, hc);
        })) {
      std::erase_if(
          hcs, [&hc](const auto& hc_inner) { return hypercube_contains_hypercube(hc, hc_inner); });
      hcs.push_back(std::move(hc));
    }
  }

  return hcs;
}

// - Count number of maxima contained in the corresponding hypercube ---------—---------—---------—-
template <typename T, int SIZE>
  requires(SIZE > 0)
[[nodiscard]] auto count_contained_maxima(const std::vector<Maximum<T, SIZE>>& maxima,
                                          const std::vector<Hypercube<T, SIZE>>& hcs) noexcept
    -> std::vector<size_t> {
  std::vector<size_t> num_contained_maxima{};
  num_contained_maxima.resize(hcs.size());

  for (size_t i = 0ul; i < hcs.size(); ++i) {
    num_contained_maxima[i] = std::transform_reduce(
        std::cbegin(maxima),
        std::cend(maxima),
        0ul,
        std::plus{},
        [&hcs, i](const Maximum<T, SIZE>& maximum) {
          return static_cast<size_t>(hypercube_contains_point(hcs[i], maximum.theta));
        });
  }
  return num_contained_maxima;
}

// - Optimistic search -----------------------------------------------------------------------------
template <typename Float, typename StanModel, int SIZE>
[[nodiscard]] auto optimistic_search(StanModel& model,
                                     size_t num_iterations,
                                     Float learning_rate,
                                     const Hypercube<Float, SIZE>& boundaries,
                                     Float tol,
                                     size_t max_iter) noexcept
    -> std::vector<Maximum<Float, SIZE>> {
  static_assert(SIZE > 0);

  std::vector<Maximum<Float, SIZE>> maxima{};

  tbb::spin_mutex emplace_mutex{};
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0UL, num_iterations, 1UL),
      [&](const tbb::blocked_range<size_t>& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
          Eigen::Vector<Float, Eigen::Dynamic> x(SIZE);
          for (Eigen::Index i = 0; i < SIZE; ++i) {
            x(i) = EB::Random::uniform_real(boundaries[i].min, boundaries[i].max);
          }

          Float func_value;
          Eigen::Vector<Float, Eigen::Dynamic> gradient;
          const auto success = adam_gradient_ascent(
              model, x, learning_rate, func_value, gradient, boundaries, tol, max_iter);

          if (success) {
            tbb::spin_mutex::scoped_lock lock{emplace_mutex};
            maxima.emplace_back(x, func_value, gradient);
          }
        }
      },
      tbb::simple_partitioner{});

  return maxima;
}

}  // namespace EB

#endif  // EB_OPTIMISTIC_SEARCH_HPP_
