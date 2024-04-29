import cmdstanpy


def main():
    stan_file = "./stan_code/sensor.stan"
    stan_data = "./data/sensor_data.json"

    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    fit = model.sample(data=stan_data, chains=9,
                       iter_warmup=1_000, iter_sampling=4_000)

    fit.draws_pd().to_csv("output/vanilla_sensor_results.csv")


if __name__ == "__main__":
    main()
