#ifndef EB_HYPERCUBE_HPP_
#define EB_HYPERCUBE_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <iosfwd>

#include <Eigen/Dense>

namespace EB {

template <typename T>
struct Interval {
  T min;
  T max;

  [[nodiscard]] constexpr auto operator==(const Interval<T>& other) const noexcept -> bool {
    return (min == other.min) && (max == other.max);
  }
};

template <typename T, int SIZE>
  requires(SIZE > 0)
struct Hypercube {
  std::array<Interval<T>, static_cast<size_t>(SIZE)> bounds;

  [[nodiscard]] constexpr auto begin() noexcept { return std::begin(bounds); }
  [[nodiscard]] constexpr auto begin() const noexcept { return std::cbegin(bounds); }
  [[nodiscard]] constexpr auto cbegin() const noexcept { return std::cbegin(bounds); }
  [[nodiscard]] constexpr auto end() noexcept { return std::end(bounds); }
  [[nodiscard]] constexpr auto end() const noexcept { return std::cend(bounds); }
  [[nodiscard]] constexpr auto cend() const noexcept { return std::cend(bounds); }

  [[nodiscard]] constexpr auto operator[](size_t idx) noexcept -> Interval<T>& {
    assert(idx < static_cast<size_t>(SIZE));
    return bounds[idx];
  }

  [[nodiscard]] constexpr auto operator[](size_t idx) const noexcept -> const Interval<T>& {
    assert(idx < static_cast<size_t>(SIZE));
    return bounds[idx];
  }

  [[nodiscard]] constexpr auto operator==(const Hypercube<T, SIZE>& other) const noexcept -> bool {
    return bounds == other.bounds;
  }
};

template <typename T, int SIZE>
auto operator<<(std::ostream& out, const Hypercube<T, SIZE>& hc) -> std::ostream& {
  out << "{ ";
  for (auto [min, max] : hc) {
    out << '[' << min << ", " << max << "], ";
  }
  out << "}";
  return out;
}

// - Calculate volume of a single hypercube --------------------------------------------------------
template <typename T, int SIZE>
[[nodiscard]] constexpr auto hypercube_volume(const Hypercube<T, SIZE>& hc) noexcept -> T {
  T volume = 1.0;
  for (const auto& [min, max] : hc.bounds) {
    volume *= max - min;
  }
  return volume;
}

// - Test if a hypercube contains a given point ----------------------------------------------------
template <typename Float, int SIZE>
[[nodiscard]] constexpr auto
hypercube_contains_point(const Hypercube<Float, SIZE>& hc,
                         const Eigen::Vector<Float, SIZE>& point) noexcept -> bool {
  static_assert(SIZE > 0, "Dynamic Eigen vectors not supported.");

  for (size_t i = 0; i < static_cast<size_t>(SIZE); ++i) {
    if (hc[i].min > point(i) || hc[i].max < point(i)) {
      return false;
    }
  }
  return true;
}

// - Test if a hypercube contains a given other hypercube ------------------------------------------
template <typename Float, int SIZE>
[[nodiscard]] constexpr auto
hypercube_contains_hypercube(const Hypercube<Float, SIZE>& hc_outer,
                             const Hypercube<Float, SIZE>& hc_inner) noexcept -> bool {
  for (size_t i = 0; i < static_cast<size_t>(SIZE); ++i) {
    if (hc_outer[i].min > hc_inner[i].min || hc_outer[i].max < hc_inner[i].max) {
      return false;
    }
  }
  return true;
}

// - Check if two hypercube overlap ----------------------------------------------------------------
template <typename Float, int SIZE>
[[nodiscard]] constexpr auto hypercubes_overlap(const Hypercube<Float, SIZE>& hc1,
                                                const Hypercube<Float, SIZE>& hc2) noexcept
    -> bool {
  for (size_t i = 0; i < static_cast<size_t>(SIZE); ++i) {
    if (hc1[i].min > hc2[i].max || hc1[i].max < hc2[i].min)
      return false;
  }
  return true;
}

// - Merge two hypercubes; assumes overlap ---------------------------------------------------------
namespace internal {
template <typename T, int SIZE>
  requires(SIZE > 0)
constexpr void merge_two_hypercubes(Hypercube<T, SIZE>& hc_in_out,
                                    const Hypercube<T, SIZE>& hc_other) noexcept {
  for (size_t i = 0; i < static_cast<size_t>(SIZE); ++i) {
    hc_in_out[i].min = std::min(hc_in_out[i].min, hc_other[i].min);
    hc_in_out[i].max = std::max(hc_in_out[i].max, hc_other[i].max);
  }
}
}  // namespace internal

// - Merge hypercubes if they overlap --------------------------------------------------------------
template <typename T, int SIZE>
  requires(SIZE > 0)
[[nodiscard]] constexpr auto
merge_hypercubes(const std::vector<Hypercube<T, SIZE>>& single_hcs) noexcept
    -> std::vector<Hypercube<T, SIZE>> {
  std::vector<Hypercube<T, SIZE>> merged_hcs{};
  merged_hcs.reserve(single_hcs.size());

  for (auto hc : single_hcs) {
    auto find_overlap = [&hc](const auto& other_hc) { return hypercubes_overlap(hc, other_hc); };
    for (auto overlap = std::find_if(std::begin(merged_hcs), std::end(merged_hcs), find_overlap);
         overlap != std::end(merged_hcs);
         overlap = std::find_if(std::begin(merged_hcs), std::end(merged_hcs), find_overlap)) {
      internal::merge_two_hypercubes(hc, *overlap);
      merged_hcs.erase(overlap);
    }
    merged_hcs.push_back(std::move(hc));
  }

  return merged_hcs;
}

}  // namespace EB

#endif  // EB_HYPERCUBE_HPP_
