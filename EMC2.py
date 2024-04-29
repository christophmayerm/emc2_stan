import cmdstanpy
import pybind11

import sysconfig
import platform
import shlex
import subprocess
import sys
from time import perf_counter
import os

from typing import Any, List


def cmd_run_echoed(cmd: List[str], echo: bool = True, **kwargs: Any) -> subprocess.CompletedProcess:
    """Runs a command and prints it.

    Args:
        cmd: List[str]: Command in list form
        echo: bool: Echo command
        **kwargs: Same as subprocess.call

    Returns:
        subprocess.CompletedProcess: Return value of the command
    """
    if echo:
        print(f"[CMD] {' '.join(map(shlex.quote, cmd))}")

    return subprocess.run(cmd, **kwargs)


def make_lib_filename(prefix: str) -> str:
    return f"{prefix}{sysconfig.get_config_var('SHLIB_SUFFIX')}"


def get_compile_info(echo=False):
    cmd = ["make", "-C", cmdstanpy.cmdstan_path(), "compile_info"]
    res = cmd_run_echoed(cmd=cmd, echo=echo, capture_output=True)
    if res.returncode != 0:
        print(f"Could not get compile info from makefile; exited with exit code {
              res.returncode}", file=sys.stderr)
        print(f"stderr:\n{res.stderr.decode('utf8')}")
        sys.exit(1)

    compile_info = res.stdout.decode("utf8")
    info_iter = iter(compile_info.split())

    compiler = next(info_iter)
    include_paths = list()
    linker_args = list()
    libraries = list()
    macros = list()

    for entry in info_iter:
        if entry.startswith("-I"):
            inc_path = next(info_iter)
            include_paths.append(
                f"-I{cmdstanpy.cmdstan_path()}/{inc_path}")
        elif entry.startswith("-Wl"):
            linker_args.append(entry.replace('"', ''))
        elif entry.startswith("-l"):
            libraries.append(entry)
        elif "lib" in entry:
            libraries.append(f"{cmdstanpy.cmdstan_path()}/{entry}")
        elif entry.startswith("-D"):
            macros.append(entry)
        elif echo:
            print(f"Ignored {entry}")

    return compiler, include_paths, linker_args, libraries, macros


def compile_shared_lib(code_file: str, shlib_file: str) -> bool:
    if os.path.exists(shlib_file) and os.path.getmtime(code_file) < os.path.getmtime(shlib_file):
        print(
            f"No compilation necessary because shared library ({shlib_file}) was modified after code ({code_file})")
        return True

    compiler, include_paths, linker_args, libraries, macros = get_compile_info()

    macros.extend(["-DBOOST_DISABLE_ASSERTS",
                   "-DBOOST_PHOENIX_NO_VARIADIC_EXPRESSION",
                   "-DSTAN_THREADS",
                   "-D_REENTRANT",
                   "-D_GLIBCXX_USE_CXX11_ABI=0",
                   "-D_HAS_AUTO_PTR_ETC=0",  # Needed to compile on MacOS
                   ])

    include_paths.append("-I./EMC2/")
    include_paths.append(f"-I{pybind11.get_include()}")
    include_paths.append(f"-I{sysconfig.get_path('include')}")

    extra_compile_args = [
        "-O3",
        "-g",
        "-std=c++20",
        "-Wno-sign-compare",
        # "-Wall",
        # "-Wextra",
        # "-pedantic",
        "-march=native",
    ]

    if platform.system() == "Darwin":
        extra_compile_args.extend(["-undefined", "dynamic_lookup",])

    cmd = [compiler]
    cmd.extend(extra_compile_args)

    cmd.extend(include_paths)

    cmd.extend(macros)

    cmd.append("-shared")
    cmd.append("-fPIC")
    cmd.extend(["-o", shlib_file])
    cmd.append(code_file)

    cmd.extend(linker_args)
    cmd.extend(libraries)

    t_begin = perf_counter()
    res = cmd_run_echoed(cmd=cmd, capture_output=True)
    if res.returncode != 0:
        print(
            f"Compilation exited with non-zero exit code:\n{res.stderr.decode('utf-8')}", file=sys.stderr)
        return False
    t_dur = perf_counter() - t_begin
    print(f"Compilation took {t_dur:.2f}s")

    return True


def create_cpp_code(stan_file: str, num_params: int, module_name: str) -> str:
    cpp_code = R"""
#include <iostream>
#include <sstream>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
<<STAN MODEL IMPLEMENTATION>>
#pragma GCC diagnostic pop

#include <Eigen/Dense>
#include <stan/io/json/json_data.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "OptimisticSearch.hpp"

using Float        = double;
constexpr int SIZE = <<NUMBER PARAMETERS>>;

std::vector<EB::Hypercube<Float, SIZE>>
find_hypercubes(const std::string& json_data_file,
                size_t num_starts,
                Float learning_rate,
                Float tol,
                size_t max_iter,
                const EB::Hypercube<Float, SIZE>& boundaries,
                Float max_factor = 0.05,
                Float min_factor = 0.01) {
  std::ifstream data_in(json_data_file);
  if (!data_in) {
    std::stringstream sstream{};
    sstream << "Could not open " << json_data_file << ": " << std::strerror(errno);
    throw std::runtime_error(sstream.str());
  }

  stan_model model = [&] {
    try {
      stan::json::json_data var_context{data_in};
      return stan_model{var_context};
    } catch (const std::exception& e) {
      std::stringstream sstream{};
      sstream << "Could not create Stan model: " << e.what();
      throw std::runtime_error(sstream.str());
    }
  }();

  std::cout << "----------------------------------------------------------------------\n";
  std::cout << "num_starts    = " << num_starts << '\n';
  std::cout << "learning_rate = " << learning_rate << '\n';
  std::cout << "tol           = " << tol << '\n';
  std::cout << "max_iter      = " << max_iter << '\n';
  std::cout << "boundaries    = " << boundaries << '\n';
  std::cout << "max_factor    = " << max_factor << '\n';
  std::cout << "min_factor    = " << min_factor << '\n';
  std::cout << "----------------------------------------------------------------------\n";

  auto t_begin = std::chrono::high_resolution_clock::now();
  const auto maxima =
      EB::optimistic_search(model, num_starts, learning_rate, boundaries, tol, max_iter);
  auto t_dur =
      std::chrono::duration<Float>(std::chrono::high_resolution_clock::now() - t_begin);
  std::cout << "Optimistic search took " << t_dur.count() << "s\n";
  std::cout << "Found " << maxima.size() << " maxima\n";

  t_begin = std::chrono::high_resolution_clock::now();
  const auto hypercubes = EB::get_hypercubes(maxima, boundaries, max_factor, min_factor);
  t_dur = std::chrono::duration<Float>(std::chrono::high_resolution_clock::now() - t_begin);
  std::cout << "Obtaining hypercubes took " << t_dur.count() << "s\n";
  std::cout << "Found " << hypercubes.size() << " hypercubes\n";

  t_begin = std::chrono::high_resolution_clock::now();
  const auto merged_hypercubes = EB::merge_hypercubes(hypercubes);
  t_dur = std::chrono::duration<Float>(std::chrono::high_resolution_clock::now() - t_begin);
  std::cout << "Merging hypercubes took " << t_dur.count() << "s\n";
  std::cout << "Found " << merged_hypercubes.size() << " merged hypercubes\n";

  return merged_hypercubes;
}

namespace py = pybind11;

PYBIND11_MODULE(<<MODULE NAME>>, m) {
  py::class_<EB::Interval<Float>>(m, "Interval")
      .def(py::init<>())
      .def(py::init<Float, Float>())
      .def_readwrite("min", &EB::Interval<Float>::min)
      .def_readwrite("max", &EB::Interval<Float>::max)
      .def("__repr__",
           [](const EB::Interval<Float>& self) {
             std::stringstream sstream{};
             sstream << "{ .min=" << self.min << ", .max=" << self.max << "}";
             return sstream.str();
           })
      .def(py::pickle([](const EB::Interval<Float>& i) { return py::make_tuple(i.min, i.max); },
                      [](py::tuple t) {
                        assert(t.size() == 2);
                        return EB::Interval<Float>{t[0].cast<Float>(), t[1].cast<Float>()};
                      }));

  py::class_<EB::Hypercube<Float, SIZE>>(m, "Hypercube")
      .def(py::init<>())
      .def("__len__", [](const EB::Hypercube<Float, SIZE>& self) { return SIZE; })
      .def("__getitem__",
           [](EB::Hypercube<Float, SIZE>& self, size_t idx) -> EB::Interval<Float>& {
             return self[idx];
           })
      .def("__setitem__",
           [](EB::Hypercube<Float, SIZE>& self, size_t idx, EB::Interval<Float> interval) {
             self[idx] = std::move(interval);
           })
      .def("__repr__",
           [](const EB::Hypercube<Float, SIZE>& self) {
             std::stringstream sstream{};
             sstream << "{ ";
             for (auto [min, max] : self) {
               sstream << "[" << min << ", " << max << "], ";
             }
             sstream << "}";
             return sstream.str();
           })
      .def(py::pickle(
          [](const EB::Hypercube<Float, SIZE>& hc) {
            py::tuple t(2 * SIZE);
            for (size_t i = 0; i < SIZE; ++i) {
              t[2 * i + 0] = hc[i].min;
              t[2 * i + 1] = hc[i].max;
            }
            return t;
          },
          [](py::tuple t) {
            assert(t.size() == 2 * SIZE);
            EB::Hypercube<Float, SIZE> hc{};
            for (size_t i = 0; i < SIZE; ++i) {
              hc[i] = EB::Interval<Float>{t[2 * i + 0].cast<Float>(), t[2 * i + 1].cast<Float>()};
            }
            return hc;
          }));

  m.def("find_hypercubes",
        &find_hypercubes,
        "Get the hypercube for the model using the optimistic search algorithm.",
        py::arg("json_data_file"),
        py::arg("num_starts"),
        py::arg("learning_rate"),
        py::arg("tol"),
        py::arg("max_iter"),
        py::arg("boundaries"),
        py::arg("max_factor") = 0.05,
        py::arg("min_factor") = 0.01);
}
"""
    cmd = list()
    cmd.append(os.path.join(cmdstanpy.cmdstan_path(),
               'bin', 'stanc' + cmdstanpy.utils.EXTENSION))
    cmd.append("--O1")
    cmd.append("--print-cpp")
    cmd.append(stan_file)

    res = cmd_run_echoed(cmd, capture_output=True)
    if res.returncode != 0:
        print(
            f"Could not transpile Stan code to C++, stanc exited with exit code {res.returncode}:\n{res.stderr.decode('utf8')}", file=sys.stderr)
        sys.exit(1)

    stan_cpp_code = res.stdout.decode("utf8")
    cpp_code = cpp_code.replace("<<STAN MODEL IMPLEMENTATION>>", stan_cpp_code)

    cpp_code = cpp_code.replace("<<NUMBER PARAMETERS>>", str(num_params))
    cpp_code = cpp_code.replace("<<MODULE NAME>>", module_name)

    return cpp_code
