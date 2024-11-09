import pandas as pd
from codecarbon import OfflineEmissionsTracker  # We import the emission tracker


def main():
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")

    tracker.start()

    #data processing, models' testing


    tracker.stop()





    emissions_csv = pd.read_csv("emissions.csv")
    last_emissions = emissions_csv.tail(1)

    emissions = last_emissions["emissions"] * 1000
    energy = last_emissions["energy_consumed"]
    cpu = last_emissions["cpu_energy"]
    gpu = last_emissions["gpu_energy"]
    ram = last_emissions["ram_energy"]

    print(f"{emissions} Grams of CO2-equivalents")
    print(f"{energy} Sum of cpu_energy, gpu_energy and ram_energy (kWh)")
    print(f"{cpu} Energy used per CPU (kWh)")
    print(f"{gpu} Energy used per GPU (kWh)")
    print(f"{ram} Energy used per RAM (kWh)")
