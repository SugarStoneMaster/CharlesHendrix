import pandas as pd
from codecarbon import OfflineEmissionsTracker  # We import the emission tracker

from src.genetic_algorithm.genetic_algorithm import genetic_algorithm


def main():
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")
    tracker.start()

    genetic_algorithm(inference=True)

    tracker.stop()
    emissions_csv = pd.read_csv("emissions.csv")
    last_emissions = emissions_csv.tail(1)

    emissions = last_emissions["emissions"] * 1000
    energy = last_emissions["energy_consumed"]
    print(f"{emissions} Grams of CO2-equivalents")
    print(f"{energy} Sum of cpu_energy, gpu_energy and ram_energy (kWh)")


if __name__ == "__main__":
    main()