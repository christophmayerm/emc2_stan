#ifndef EB_ADAM_GRADIENT_ASCENT_HPP_
#define EB_ADAM_GRADIENT_ASCENT_HPP_

#include <vector>

#include <Eigen/Dense>

#include <stan/model/gradient.hpp>

#include "Hypercube.hpp"

namespace EB {

template <typename StanModel, typename Float, int SIZE>
[[nodiscard]] constexpr auto
adam_gradient_ascent(StanModel& model,
                     Eigen::VectorX<Float>& x,
                     Float learning_rate,
                     Float& function_value,
                     Eigen::VectorX<Float>& gradient,
                     const Hypercube<Float, SIZE>& boundaries,
                     Float tol       = std::sqrt(std::numeric_limits<Float>::epsilon()),
                     size_t max_iter = 500ul,
                     Float beta1     = 0.9,
                     Float beta2     = 0.999,
                     Float eps       = 1e-8) noexcept -> bool {
  assert(x.rows() == SIZE);
  constexpr auto n = SIZE;

  const auto update_x = [eps, n](Eigen::VectorX<Float>& local_x_new,
                                 const Eigen::VectorX<Float>& local_x,
                                 Float lr,
                                 const Eigen::VectorX<Float>& bc_first_mom,
                                 const Eigen::VectorX<Float>& bc_second_mom) {
    for (Eigen::Index i = 0; i < n; ++i) {
      local_x_new(i) = local_x(i) + lr * bc_first_mom(i) / (std::sqrt(bc_second_mom(i)) + eps);
    }
  };

  size_t iter                  = 0ul;
  constexpr Float diverged_tol = 1e8;

  Eigen::VectorX<Float> first_momentum  = Eigen::VectorX<Float>::Zero(n);
  Eigen::VectorX<Float> second_momentum = Eigen::VectorX<Float>::Zero(n);
  Eigen::VectorX<Float> bias_corr_first_momentum(n);
  Eigen::VectorX<Float> bias_corr_second_momentum(n);
  Eigen::VectorX<Float> x_new(n);

  Float alpha                         = 1.0;
  constexpr Float alpha_adjust_factor = 2.0;
  constexpr Float alpha_min = static_cast<Float>(1) / static_cast<Float>(1 << 8);  // 2^(-8)
  bool adjust_alpha         = true;

  const auto abort_early = [max_iter, diverged_tol](size_t local_iter,
                                                    const Float& local_function_value,
                                                    const Eigen::VectorX<Float>& local_gradient) {
    return local_iter > max_iter || std::isnan(local_function_value) ||
           std::isinf(local_function_value) || local_gradient.norm() > diverged_tol;
  };

  do {
    try {
      stan::model::gradient(model, x, function_value, gradient);
    } catch (const std::exception& e) {
      std::cerr << "Could not calculate gradient: " << e.what() << '\n';
      std::exit(1);
    }

    first_momentum  = beta1 * first_momentum + (1.0 - beta1) * gradient;
    second_momentum = beta2 * second_momentum + (1.0 - beta2) * gradient.cwiseProduct(gradient);

    ++iter;
    bias_corr_first_momentum  = first_momentum / (1.0 - std::pow(beta1, iter));
    bias_corr_second_momentum = second_momentum / (1.0 - std::pow(beta2, iter));

    alpha        = 1.0;
    adjust_alpha = true;
    while (adjust_alpha) {
      update_x(
          x_new, x, alpha * learning_rate, bias_corr_first_momentum, bias_corr_second_momentum);
      for (Eigen::Index i = 0; i < n; ++i) {
        if (x_new(i) < boundaries[i].min || x_new(i) > boundaries[i].max) {
          alpha /= alpha_adjust_factor;
          if (alpha < alpha_min) {
            return false;
          }
          adjust_alpha = true;
          break;
        } else {
          adjust_alpha = false;
        }
      }
    }
    x = x_new;
  } while (gradient.norm() > tol && !abort_early(iter, function_value, gradient));

  return !abort_early(iter, function_value, gradient);
}

template <typename StanModel, typename Float, int SIZE>
[[nodiscard]] constexpr auto
adam_gradient_ascent(StanModel& model,
                     Eigen::VectorX<Float>& x,
                     Float learning_rate,
                     const Hypercube<Float, SIZE>& boundaries,
                     Float tol       = std::sqrt(std::numeric_limits<Float>::epsilon()),
                     size_t max_iter = 500ul,
                     Float beta1     = 0.9,
                     Float beta2     = 0.999,
                     Float eps       = 1e-8) noexcept -> bool {
  Float tmp_function_value;
  Eigen::VectorX<Float> tmp_gradient;

  return adam_gradient_ascent(model,
                              x,
                              learning_rate,
                              tmp_function_value,
                              tmp_gradient,
                              boundaries,
                              tol,
                              max_iter,
                              beta1,
                              beta2,
                              eps);
}

}  // namespace EB

#endif  // EB_ADAM_GRADIENT_ASCENT_HPP_
