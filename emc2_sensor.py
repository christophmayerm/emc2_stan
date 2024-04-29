import sys
import os
import json
import multiprocessing
import tempfile
from itertools import repeat

import cmdstanpy

from EMC2 import make_lib_filename, create_cpp_code, compile_shared_lib


def add_constraint(stan_code, hypercube):
    assert (len(hypercube) == 8)

    model_section_str = "model {"
    begin_model_section = str.find(stan_code, "model {")
    if begin_model_section == -1:
        print("Invalid stan code", file=sys.stderr)
        return None

    idx = begin_model_section + len(model_section_str)
    stan_code = f"{stan_code[:idx]}\n{stan_code[idx:]}"
    idx += 1

    seperator = "    // - Auto generated: Self-informed prior constraint on parameters ------------\n"
    stan_code = f"{stan_code[:idx]}{seperator}{stan_code[idx:]}"
    idx += len(seperator)

    for i in range(len(hypercube)):
        row = i % 4
        col = i // 4
        new_prior = f"    x_unknown[{row+1}, {col+1}] ~ uniform({hypercube[i].min}, {hypercube[i].max});\n"
        stan_code = f"{stan_code[:idx]}{new_prior}{stan_code[idx:]}"
        idx += len(new_prior)

    stan_code = f"{stan_code[:idx]}{seperator}{stan_code[idx:]}"
    idx += len(seperator)

    return stan_code


def run_constrained_model(hypercube, stan_code, stan_data):
    new_stan_code = add_constraint(stan_code, hypercube)

    with tempfile.NamedTemporaryFile(suffix=".stan", mode="w") as tmpfile:
        print(new_stan_code, file=tmpfile.file, end="")
        tmpfile.seek(0, 0)

        init = {"x_unknown": [[None, None],
                              [None, None],
                              [None, None],
                              [None, None]]}

        min_extend = hypercube[0].max - hypercube[0].min
        for i in range(len(hypercube)):
            row = i % 4
            col = i // 4
            init["x_unknown"][row][col] = (
                hypercube[i].min + hypercube[i].max) / 2

            min_extend = min(min_extend, hypercube[i].max - hypercube[i].min)

        model = cmdstanpy.CmdStanModel(stan_file=tmpfile.name)
        return model.sample(data=stan_data,
                            chains=2,
                            inits=init,
                            step_size=0.01 * min_extend,
                            iter_warmup=100,
                            iter_sampling=2_000,
                            show_progress=False,
                            show_console=False)


def main():
    num_params = 8
    stan_file = "./stan_code/sensor.stan"
    cpp_file = "./stan_code/sensor.cpp"
    module_name = "sensor"
    lib_file = f"./{make_lib_filename(module_name)}"

    if os.path.exists(cpp_file) and os.path.getmtime(stan_file) < os.path.getmtime(cpp_file):
        print(
            f"No compilation necessary because c++ file ({cpp_file}) was modified after stan file ({stan_file})")
    else:
        cpp_code = create_cpp_code(
            stan_file=stan_file, num_params=num_params, module_name=module_name)
        with open(cpp_file, "w") as f:
            print(cpp_code, file=f, end='')

    if not compile_shared_lib(cpp_file, lib_file):
        print("Could not compile shared library")
        sys.exit(1)

    import sensor

    stan_data_file = "./data/sensor_data.json"
    num_starts = 50_000
    learning_rate = 1e-2
    tol = 75e-2
    max_iter = 10_000
    max_factor = 0.15
    boundaries = sensor.Hypercube()
    for i in range(num_params):
        boundaries[i] = sensor.Interval(-1.0, 2.0)

    try:
        hypercubes = sensor.find_hypercubes(
            stan_data_file, num_starts, learning_rate, tol, max_iter, boundaries, max_factor=max_factor)
    except Exception as e:
        print(f"Could not find hypercubes: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        hc_file = "output/emc2_sensor_hypercubes.txt"
        with open(hc_file, "w") as f:
            for i, hc in enumerate(hypercubes):
                print(f"{i}: {hc}", file=f)
    except IOError as err:
        print(err, file=sys.stderr)
        sys.exit(1)
    print(f"Wrote hypercubes to {hc_file}")

    with open(stan_data_file, "r") as f:
        stan_data = json.load(f)

    with open(stan_file, "r") as f:
        stan_code = f.read()

    num_procs = os.cpu_count() if os.cpu_count() is not None else 4
    with multiprocessing.Pool(processes=num_procs) as pool:
        fits = pool.starmap(run_constrained_model, zip(
            hypercubes, repeat(stan_code), repeat(stan_data)))

    for i, fit in enumerate(fits):
        #     print(fit.draws_pd().describe())
        fit.draws_pd().to_csv(f"output/emc2_sensor_results_{i}.csv")


if __name__ == "__main__":
    main()
