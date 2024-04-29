#ifndef EB_RANDOM_VALUE_HPP_
#define EB_RANDOM_VALUE_HPP_

#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace EB::Random {

namespace internal {

// Option to define a seed for the random number generator
#ifdef EB_RAND_SEED
static_assert(std::is_integral_v<decltype(EB_RAND_SEED)>,
              "`EB_RAND_SEED` is not a valid seed, must be an integral type.");
#pragma message "You are using the fixed seed '" EB_STRINGIFY(                                     \
    EB_RAND_SEED) "' for the random number generator."
#else
#define EB_RAND_SEED std::random_device{}()
#endif  // EB_RAND_SEED

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::mt19937 generator(EB_RAND_SEED);

/// Normal distribution with mean = 0 and std_dev = 1
std::normal_distribution<double> normal_real_dist(0.0, 1.0);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

}  // namespace internal

/// Returns random real number between two values -> uniform distribution
template <typename Float>
  requires std::is_floating_point_v<Float>
[[nodiscard]] auto uniform_real(Float lower, Float upper) noexcept -> Float {
  std::uniform_real_distribution<Float> dist(lower, upper);
  return dist(internal::generator);
}

/// Returns random int between two values -> uniform distribution
template <typename Int>
  requires std::is_integral_v<Int>
[[nodiscard]] auto uniform_int(Int lower, Int upper) noexcept -> Int {
  std::uniform_int_distribution<Int> distribution(lower, upper);
  return distribution(internal::generator);
}

/// Returns random real value -> normal distribution
/// Gets mean and standard deviation
template <typename Float>
  requires std::is_floating_point_v<Float>
[[nodiscard]] auto normal_real(Float mean, Float std_dev) noexcept -> Float {
  return std_dev * static_cast<Float>(internal::normal_real_dist(internal::generator)) + mean;
}

}  // namespace EB::Random

#endif  // EB_RANDOM_VALUE_HPP_
